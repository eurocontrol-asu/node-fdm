from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Protocol

import polars as pl
from pydantic import BaseModel

from node_fdm_pipeline.commands._day_assemble import assemble_partition
from node_fdm_pipeline.commands._day_bounds import (
    ResourceBudget,
    register_parent_process,
    require_parent_process,
    run_bounded_slices,
)
from node_fdm_pipeline.commands._day_cleanup import cleanup_day
from node_fdm_pipeline.commands._day_commit import DayCommit, DaySnapshot, commit_day
from node_fdm_pipeline.commands._day_plan import (
    DayPartitionKey,
    DayPlan,
    build_day_plan,
)
from node_fdm_pipeline.commands._day_publish import (
    StepHook,
    load_partition_resume_state,
    partition_resume_state,
    publish_day,
    resume_publication,
)
from node_fdm_pipeline.commands._day_weather import DayGridCache
from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan
from node_fdm_pipeline.commands._science_profile import (
    ScienceProfile,
    profile_manifest,
)

__all__ = [
    "DayRunReport",
    "PreviousDayCleanupFailed",
    "admit_day",
    "assert_day_admissible",
    "load_day_journal_snapshot",
    "partitions_to_assemble",
    "run_selection_day",
]

type DayJournalSnapshot = tuple[dict[str, object], ...]
type CleanupStepHook = Callable[[str, str], None]


class PreviousDayCleanupFailed(RuntimeError):  # noqa: N818
    """Raised when a day's durable predecessor still has failed cleanup."""


class DayRunReport(BaseModel, frozen=True):
    """Immutable observations from one complete selection-day run."""

    external_access_pids: frozenset[int]
    worker_pids: tuple[int, ...]
    grid_opens: dict[tuple[str, str], int]


class _OpenSkyReader(Protocol):
    def __call__(self, source_day: str, selection_ids: frozenset[str]) -> pl.DataFrame:
        """Read locally recorded OpenSky rows for one UTC source day."""


class _GridOpener(Protocol):
    def __call__(self, source_day: str, field_name: str, /) -> object:
        """Open one ERA5 field for one UTC source day."""


class _GridCloser(Protocol):
    def __call__(self, handle: object) -> None:
        """Close an ERA5 handle."""


class _LeaseHandle(Protocol):
    def acquire(self) -> None:
        """Acquire the lease."""

    def heartbeat(self) -> None:
        """Renew the lease."""

    def assert_owned(self) -> None:
        """Verify possession of the lease."""

    def release(self) -> None:
        """Release the lease."""


@dataclass(frozen=True)
class _AssemblySlice:
    slice_id: str
    slice_gib: float
    frame: pl.DataFrame
    plan: DayPlan
    key: DayPartitionKey
    profile: ScienceProfile
    versions: dict[str, str]
    grid_handles: dict[str, object]


@dataclass(frozen=True)
class _AssemblyResult:
    key: DayPartitionKey
    frame: pl.DataFrame
    pid: int


def _assemble_slice(slice_: _AssemblySlice) -> _AssemblyResult:
    frame = assemble_partition(
        slice_.frame,
        slice_.plan,
        slice_.key,
        slice_.profile,
        slice_.versions,
        grid_handles=slice_.grid_handles,
    )
    return _AssemblyResult(key=slice_.key, frame=frame, pid=os.getpid())


def _previous_day(day: str) -> str:
    compact = "-" not in day
    format_ = "%Y%m%d" if compact else "%Y-%m-%d"
    parsed = datetime.strptime(day, format_).date()
    return (parsed - timedelta(days=1)).strftime(format_)


def _event_day(event: Mapping[str, object]) -> str | None:
    value = event.get("day", event.get("meta_selection_day"))
    return value if isinstance(value, str) else None


def assert_day_admissible(snapshot: Iterable[Mapping[str, object]], day: str) -> None:
    """Reject a day when its predecessor's latest cleanup outcome is failed."""
    previous_day = _previous_day(day)
    last_cleanup_state: str | None = None
    for event in snapshot:
        name = event.get("event")
        if _event_day(event) == previous_day and name in {"cleanup_failed", "cleanup_completed"}:
            last_cleanup_state = str(name)
    if last_cleanup_state == "cleanup_failed":
        raise PreviousDayCleanupFailed(
            f"previous day {previous_day} cleanup failed; resume cleanup before starting {day}"
        )


def load_day_journal_snapshot(journal_path: Path) -> DayJournalSnapshot:
    """Load the durable events used to decide day admission."""
    if not journal_path.exists():
        return ()
    return tuple(read_events(journal_path))


def admit_day(day: str, journal_path: Path) -> DayJournalSnapshot:
    """Recompute and enforce admission from the durable journal."""
    snapshot = load_day_journal_snapshot(journal_path)
    assert_day_admissible(snapshot, day)
    return snapshot


def _parent_call[ResultT](
    pids: set[int],
    action: str,
    operation: Callable[[], ResultT],
) -> ResultT:
    require_parent_process(action)
    pids.add(os.getpid())
    return operation()


def _read_opensky_sources(
    plan: DayPlan,
    reader: _OpenSkyReader,
    external_pids: set[int],
) -> pl.DataFrame:
    frames = [
        _parent_call(
            external_pids,
            f"opensky_read:{source_day}",
            partial(reader, source_day, plan.selection_ids),
        )
        for source_day in plan.source_days
    ]
    if not frames:
        raise ValueError(f"day {plan.meta_selection_day} has no OpenSky source day")
    return pl.concat(frames, how="vertical_relaxed")


@dataclass
class _OpenedDayGrids:
    handles: dict[str, object]
    opens: dict[tuple[str, str], int]
    cache: DayGridCache[object]

    def close(self) -> None:
        for source_day in self.handles:
            self.cache.evict(source_day)


def _open_shared_grids(
    plan: DayPlan,
    fields: tuple[str, ...],
    opener: _GridOpener,
    closer: _GridCloser,
    external_pids: set[int],
) -> _OpenedDayGrids:
    def guarded_opener(source_day: str, field_name: str) -> object:
        return _parent_call(
            external_pids,
            f"era5_open:{source_day}:{field_name}",
            lambda: opener(source_day, field_name),
        )

    if plan.source_days and not fields:
        raise ValueError("at least one ERA5 field is required")

    cache = DayGridCache[object](opener=guarded_opener, closer=closer)
    handles: dict[str, object] = {}
    field_name = fields[0] if fields else ""
    for source_day in plan.source_days:
        handle = cache.acquire(source_day, field_name)
        handles[source_day] = handle
        cache.release(source_day, field_name)
        if not (hasattr(handle, "interpolate") or hasattr(handle, "data_vars")):
            for legacy_field in fields[1:]:
                cache.acquire(source_day, legacy_field)
                cache.release(source_day, legacy_field)
    return _OpenedDayGrids(
        handles=handles,
        opens=cache.stats().opens,
        cache=cache,
    )


def _assemble_partitions(  # noqa: PLR0913
    frame: pl.DataFrame,
    plan: DayPlan,
    profile: ScienceProfile,
    versions: Mapping[str, str],
    local_workers: int,
    max_resident_gib: float,
    *,
    grid_handles: Mapping[str, object] | None = None,
) -> tuple[dict[DayPartitionKey, pl.DataFrame], tuple[int, ...]]:
    slices = tuple(
        _AssemblySlice(
            slice_id=f"{key.cohort}:{key.meta_selection_day}",
            slice_gib=1.0,
            frame=frame,
            plan=plan,
            key=key,
            profile=profile,
            versions=dict(versions),
            grid_handles=dict(grid_handles or {}),
        )
        for key in plan.partition_keys
    )
    report = run_bounded_slices(
        slices=slices,
        budget=ResourceBudget(
            local_workers=local_workers,
            max_resident_gib=max_resident_gib,
        ),
        fn=_assemble_slice,
    )
    results: list[_AssemblyResult] = []
    for result in report.results:
        if not isinstance(result, _AssemblyResult):
            raise TypeError("day assembly returned an unexpected result")
        results.append(result)
    partitions = {result.key: result.frame for result in results}
    worker_pids = tuple(sorted({result.pid for result in results}))
    return partitions, worker_pids


def _cleanup_hook(value: object | None) -> CleanupStepHook | None:
    if value is None:
        return None
    if not callable(value):
        raise TypeError("cleanup_step_hook must be callable")

    def invoke(step: str, artifact_id: str) -> None:
        value(step, artifact_id)

    return invoke


def _committed_snapshot(plan: DayPlan, profile: ScienceProfile) -> DaySnapshot:
    marker = DayCommit(
        meta_selection_day=plan.meta_selection_day,
        partition_keys=plan.partition_keys,
        profile_manifest=profile_manifest(profile),
    )
    return DaySnapshot(
        meta_selection_day=plan.meta_selection_day,
        published_keys=plan.partition_keys,
        day_committed=marker,
    )


def _run_parent_acquisition(  # noqa: PLR0913
    lease: _LeaseHandle,
    plan: DayPlan,
    opensky_reader: _OpenSkyReader,
    grid_opener: _GridOpener,
    grid_closer: _GridCloser,
    era5_fields: tuple[str, ...],
    external_pids: set[int],
) -> tuple[pl.DataFrame, _OpenedDayGrids]:
    _parent_call(external_pids, "trino_lease_acquire", lease.acquire)
    try:
        _parent_call(external_pids, "trino_lease_heartbeat", lease.heartbeat)
        _parent_call(external_pids, "trino_lease_possession", lease.assert_owned)
        frame = _read_opensky_sources(plan, opensky_reader, external_pids)
        shared_grids = _open_shared_grids(
            plan,
            era5_fields,
            grid_opener,
            grid_closer,
            external_pids,
        )
        return frame, shared_grids
    finally:
        _parent_call(external_pids, "trino_lease_release", lease.release)


def partitions_to_assemble(
    plan: DayPlan,
    resume_state: Mapping[DayPartitionKey, str],
) -> tuple[DayPartitionKey, ...]:
    """Return pending partition keys in their canonical plan order."""
    return tuple(key for key in plan.partition_keys if resume_state[key] == "pending")


def _partition_subplan(
    plan: DayPlan,
    keys: tuple[DayPartitionKey, ...],
) -> DayPlan:
    selection_ids_by_key = {key: plan.selection_ids_by_key[key] for key in keys}
    return DayPlan(
        meta_selection_day=plan.meta_selection_day,
        partition_keys=keys,
        selection_ids=frozenset(
            selection_id for key in keys for selection_id in selection_ids_by_key[key]
        ),
        selection_ids_by_key=selection_ids_by_key,
        source_days=plan.source_days,
    )


def run_selection_day(  # noqa: PLR0913
    day: str,
    *,
    selection: SelectionPlan,
    opensky_reader: _OpenSkyReader,
    grid_opener: _GridOpener,
    grid_closer: _GridCloser,
    era5_fields: tuple[str, ...],
    lease: _LeaseHandle,
    staging_root: Path,
    published_root: Path,
    journal_path: Path,
    counters_path: Path,
    cleanup_artifacts: Mapping[str, Path],
    cleanup_decrements: Iterable[str],
    science_profile: ScienceProfile,
    versions: Mapping[str, str],
    local_workers: int,
    max_resident_gib: float,
    step_hook: StepHook | None = None,
    cleanup_step_hook: object | None = None,
) -> DayRunReport:
    """Run one admitted selection day through assembly, publication, commit, and cleanup."""
    admit_day(day, journal_path)
    register_parent_process()
    append_event(journal_path, {"event": "day_started", "meta_selection_day": day})
    plan = build_day_plan(selection, day)
    resume_state = (
        load_partition_resume_state(journal_path, plan)
        if journal_path.exists()
        else partition_resume_state((), plan)
    )
    pending_keys = partitions_to_assemble(plan, resume_state)
    pending_plan = _partition_subplan(plan, pending_keys)
    external_pids: set[int] = set()
    if pending_keys:
        frame, shared_grids = _run_parent_acquisition(
            lease,
            pending_plan,
            opensky_reader,
            grid_opener,
            grid_closer,
            era5_fields,
            external_pids,
        )
        try:
            partitions, worker_pids = _assemble_partitions(
                frame,
                pending_plan,
                science_profile,
                versions,
                local_workers,
                max_resident_gib,
                grid_handles=shared_grids.handles,
            )
        finally:
            shared_grids.close()
        grid_opens = shared_grids.opens
    else:
        partitions = {}
        worker_pids = ()
        grid_opens = {}
    if all(state == "pending" for state in resume_state.values()):
        publish_day(
            partitions=partitions,
            plan=plan,
            staging_root=staging_root,
            published_root=published_root,
            event_log=journal_path,
            step_hook=step_hook,
        )
    else:
        resume_publication(
            partitions=partitions,
            plan=plan,
            staging_root=staging_root,
            published_root=published_root,
            event_log=journal_path,
        )
    commit_day(
        plan,
        published_root,
        journal_path,
        science_profile,
        step_hook=step_hook,
    )
    cleanup_day(
        _committed_snapshot(plan, science_profile),
        journal_path=journal_path,
        counters_path=counters_path,
        artifacts=cleanup_artifacts,
        decrements=cleanup_decrements,
        step_hook=_cleanup_hook(cleanup_step_hook),
    )
    return DayRunReport(
        external_access_pids=frozenset(external_pids),
        worker_pids=worker_pids,
        grid_opens=grid_opens,
    )
