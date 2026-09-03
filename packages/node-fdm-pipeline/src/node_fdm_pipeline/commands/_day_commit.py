from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict

from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan
from node_fdm_pipeline.commands._day_publish import validate_partition
from node_fdm_pipeline.commands._fleet_manifest import (
    ManifestCorruption,
    append_event,
    read_events,
)
from node_fdm_pipeline.commands._science_profile import (
    ScienceProfile,
    profile_manifest,
    resolve_science_profile,
)

__all__ = [
    "DayCommit",
    "DayIncomplete",
    "DaySnapshot",
    "PurgeForbiddenBeforeCommit",
    "assert_purge_allowed",
    "commit_day",
    "load_day_snapshot",
    "missing_expected_keys",
    "reconcile_commit",
]

type StepHook = Callable[[str], None]

_PARTITION_KEY_PARTS = 2


class DayIncomplete(RuntimeError):  # noqa: N818
    """Raised when a day cannot be committed because a planned partition is absent."""


class PurgeForbiddenBeforeCommit(RuntimeError):  # noqa: N818
    """Raised when destructive retention is requested before the durable day marker."""


class DayCommit(BaseModel):
    """Validated durable marker closing one selection day."""

    model_config = ConfigDict(frozen=True)

    meta_selection_day: str
    partition_keys: tuple[DayPartitionKey, ...]
    profile_manifest: dict[str, str]


class DaySnapshot(BaseModel):
    """Immutable state reconstructed from one day's complete journal events."""

    model_config = ConfigDict(frozen=True)

    meta_selection_day: str
    published_keys: tuple[DayPartitionKey, ...]
    day_committed: DayCommit | None = None


class _PurgeSnapshot(Protocol):
    meta_selection_day: str
    published_keys: tuple[DayPartitionKey, ...]
    day_committed: object | None


def missing_expected_keys(
    plan: DayPlan,
    published_keys: Collection[DayPartitionKey],
) -> tuple[DayPartitionKey, ...]:
    """Return absent planned keys without losing the plan's deterministic order."""
    published = set(published_keys)
    return tuple(key for key in plan.partition_keys if key not in published)


def _partition_path(root: Path, key: DayPartitionKey) -> Path:
    return (
        root
        / f"cohort={key.cohort}"
        / f"meta_selection_day={key.meta_selection_day}"
        / "data.parquet"
    )


def _profile_digest(profile: ScienceProfile) -> str:
    payload = json.dumps(
        profile_manifest(profile),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_profile(
    frame: pl.DataFrame,
    key: DayPartitionKey,
    profile: ScienceProfile,
) -> None:
    expected_digest = _profile_digest(profile)
    profile_ids = frame.get_column("profile_id").unique().to_list()
    profile_digests = frame.get_column("profile_digest").unique().to_list()
    if profile_ids != [profile.profile_id] or profile_digests != [expected_digest]:
        raise DayIncomplete(f"partition {tuple(key)!r} has an invalid science profile")


def _validate_complete_day(
    plan: DayPlan,
    published_root: Path,
    profile: ScienceProfile,
) -> None:
    published_keys = tuple(
        key for key in plan.partition_keys if _partition_path(published_root, key).is_file()
    )
    missing = missing_expected_keys(plan, published_keys)
    if missing:
        raise DayIncomplete(f"missing partition key {tuple(missing[0])!r}")

    for key in plan.partition_keys:
        frame = pl.read_parquet(_partition_path(published_root, key))
        validate_partition(frame=frame, plan=plan, key=key)
        _validate_profile(frame, key, profile)


def _event_marker(event: dict[str, object]) -> DayCommit | None:
    if event.get("event") != "day_committed":
        return None

    day = event.get("meta_selection_day")
    raw_keys = event.get("partition_keys")
    raw_profile = event.get("profile_manifest")
    if (
        not isinstance(day, str)
        or not isinstance(raw_keys, list)
        or not isinstance(raw_profile, dict)
    ):
        raise ManifestCorruption("invalid day_committed event")

    keys: list[DayPartitionKey] = []
    for raw_key in raw_keys:
        if (
            not isinstance(raw_key, list)
            or len(raw_key) != _PARTITION_KEY_PARTS
            or not all(isinstance(part, str) for part in raw_key)
        ):
            raise ManifestCorruption("invalid partition key in day_committed event")
        cohort, selection_day = raw_key
        if not isinstance(cohort, str) or not isinstance(selection_day, str):
            raise ManifestCorruption("invalid partition key in day_committed event")
        keys.append(DayPartitionKey(cohort, selection_day))

    manifest: dict[str, str] = {}
    for name, value in raw_profile.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ManifestCorruption("invalid profile manifest in day_committed event")
        manifest[name] = value

    return DayCommit(
        meta_selection_day=day,
        partition_keys=tuple(keys),
        profile_manifest=manifest,
    )


def _markers(events: list[dict[str, object]], day: str) -> tuple[DayCommit, ...]:
    markers = tuple(
        marker
        for event in events
        if (marker := _event_marker(event)) is not None and marker.meta_selection_day == day
    )
    if len(markers) > 1:
        raise ManifestCorruption(f"duplicate day_committed events for {day}")
    return markers


def _marker_payload(
    plan: DayPlan,
    profile: ScienceProfile,
) -> dict[str, object]:
    return {
        "event": "day_committed",
        "meta_selection_day": plan.meta_selection_day,
        "partition_keys": [list(key) for key in plan.partition_keys],
        "profile_manifest": profile_manifest(profile),
    }


def _assert_matching_marker(
    marker: DayCommit,
    plan: DayPlan,
    profile: ScienceProfile,
) -> None:
    expected = DayCommit(
        meta_selection_day=plan.meta_selection_day,
        partition_keys=plan.partition_keys,
        profile_manifest=profile_manifest(profile),
    )
    if marker != expected:
        raise ManifestCorruption(
            f"day_committed event for {plan.meta_selection_day} conflicts with current day"
        )


def commit_day(
    plan: DayPlan,
    published_root: Path,
    journal_path: Path,
    profile: ScienceProfile,
    *,
    step_hook: StepHook | None = None,
) -> None:
    """Validate and durably close a complete day exactly once."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        events = read_events(journal_path)
        existing = _markers(events, plan.meta_selection_day)
        if existing:
            _assert_matching_marker(existing[0], plan, profile)
            return

        _validate_complete_day(plan, published_root, profile)
        append_event(journal_path, _marker_payload(plan, profile))
        if step_hook is not None:
            step_hook("day_committed")


def _published_key(event: dict[str, object]) -> DayPartitionKey | None:
    if event.get("event") != "publish_partition":
        return None
    cohort = event.get("cohort")
    day = event.get("meta_selection_day")
    if not isinstance(cohort, str) or not isinstance(day, str):
        raise ManifestCorruption("invalid publish_partition event")
    return DayPartitionKey(cohort, day)


def load_day_snapshot(journal_path: Path) -> DaySnapshot:
    """Reconstruct one day's publication and commit state from its journal."""
    events = read_events(journal_path)
    published = tuple(key for event in events if (key := _published_key(event)) is not None)
    markers = tuple(marker for event in events if (marker := _event_marker(event)) is not None)
    days = {key.meta_selection_day for key in published}
    days.update(marker.meta_selection_day for marker in markers)
    if len(days) != 1:
        raise ManifestCorruption("day journal must describe exactly one selection day")

    day = next(iter(days))
    day_markers = tuple(marker for marker in markers if marker.meta_selection_day == day)
    if len(day_markers) > 1:
        raise ManifestCorruption(f"duplicate day_committed events for {day}")
    committed = day_markers[0] if day_markers else None
    published_keys = tuple(dict.fromkeys(published))
    if not published_keys and committed is not None:
        published_keys = committed.partition_keys
    return DaySnapshot(
        meta_selection_day=day,
        published_keys=published_keys,
        day_committed=committed,
    )


def assert_purge_allowed(snapshot: _PurgeSnapshot) -> None:
    """Refuse every purge until the durable day commit marker exists."""
    if snapshot.day_committed is None:
        raise PurgeForbiddenBeforeCommit(
            f"purge forbidden before day_committed for {snapshot.meta_selection_day}"
        )


def _discover_partitions(
    published_root: Path,
) -> dict[str, tuple[DayPartitionKey, ...]]:
    keys_by_day: dict[str, list[DayPartitionKey]] = {}
    for path in sorted(published_root.glob("cohort=*/meta_selection_day=*/data.parquet")):
        cohort_part = path.parent.parent.name
        day_part = path.parent.name
        if not cohort_part.startswith("cohort=") or not day_part.startswith("meta_selection_day="):
            continue
        key = DayPartitionKey(
            cohort_part.removeprefix("cohort="),
            day_part.removeprefix("meta_selection_day="),
        )
        keys_by_day.setdefault(key.meta_selection_day, []).append(key)
    return {day: tuple(sorted(keys)) for day, keys in keys_by_day.items()}


def _recovery_plan(
    published_root: Path,
    day: str,
    keys: tuple[DayPartitionKey, ...],
) -> tuple[DayPlan, ScienceProfile]:
    selection_ids_by_key: dict[DayPartitionKey, frozenset[str]] = {}
    profile_ids: set[str] = set()
    source_days: set[str] = set()
    for key in keys:
        frame = pl.read_parquet(_partition_path(published_root, key))
        selection_ids_by_key[key] = frozenset(
            str(value) for value in frame.get_column("selection_id").unique().to_list()
        )
        profile_ids.update(
            str(value) for value in frame.get_column("profile_id").unique().to_list()
        )
        source_days.update(
            str(value) for value in frame.get_column("meta_source_day").unique().to_list()
        )

    if len(profile_ids) != 1:
        raise ManifestCorruption(f"published partitions for {day} disagree on profile")
    profile = resolve_science_profile(next(iter(profile_ids)))
    plan = DayPlan(
        meta_selection_day=day,
        partition_keys=keys,
        selection_ids=frozenset().union(*selection_ids_by_key.values()),
        selection_ids_by_key=selection_ids_by_key,
        source_days=tuple(sorted(source_days)),
    )
    _validate_complete_day(plan, published_root, profile)
    return plan, profile


def reconcile_commit(journal_path: Path, published_root: Path) -> None:
    """Recover an interrupted commit marker without rewriting published partitions."""
    discovered = _discover_partitions(published_root)
    if not discovered:
        raise DayIncomplete("no published partitions available for reconciliation")

    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        events = read_events(journal_path)
        for day, keys in discovered.items():
            existing = _markers(events, day)
            if existing:
                continue
            plan, profile = _recovery_plan(published_root, day, keys)
            append_event(journal_path, _marker_payload(plan, profile))
            events = read_events(journal_path)
