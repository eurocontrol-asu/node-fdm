from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import polars as pl
import pytest
import xarray as xr

from _daily_corpus import build_raw_opensky_day
from _local_era5_grid import write_local_era5_grid
from node_fdm_pipeline.commands import _day_commit, _day_publish, _day_runner
from node_fdm_pipeline.commands._day_plan import DayPartitionKey, build_day_plan
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan
from node_fdm_pipeline.commands._science_profile import resolve_science_profile

pytestmark = pytest.mark.integration

DAY = "20200101"
PROFILE_ID = "opensky26-exp03-v1"
VERSIONS = {"node-fdm-pipeline": "test-code-v1"}
ERA5_FIELDS = (
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
)


@dataclass
class _Reader:
    root: Path

    def __call__(self, source_day: str, selection_ids: frozenset[str]) -> pl.DataFrame:
        return pl.read_parquet(self.root / f"{source_day}.parquet").filter(
            pl.col("selection_id").is_in(selection_ids)
        )


@dataclass
class _GridOpener:
    stores: dict[str, Path]

    def __call__(self, source_day: str, field_name: str) -> object:
        del field_name
        return cast("xr.Dataset", xr.open_zarr(str(self.stores[source_day]), chunks=None))


def _close_grid(handle: object) -> None:
    assert isinstance(handle, xr.Dataset)
    handle.close()


@dataclass
class _Lease:
    calls: list[str] = field(default_factory=list)

    def acquire(self) -> None:
        self.calls.append("acquire")

    def heartbeat(self) -> None:
        self.calls.append("heartbeat")

    def assert_owned(self) -> None:
        self.calls.append("assert_owned")

    def release(self) -> None:
        self.calls.append("release")


@dataclass(frozen=True)
class _RunPaths:
    raw_root: Path
    stores: dict[str, Path]
    staging_root: Path
    published_root: Path
    journal_path: Path
    counters_path: Path
    artifact: Path


def _selection() -> SelectionPlan:
    return SelectionPlan(
        flights=(
            SelectedFlight(
                icao24="39a20a",
                callsign="A20N1",
                firstseen=1_577_872_800,
                lastseen=1_577_873_100,
                msn="A20N-001",
                split="train",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-a20n",
                utc_days=(DAY,),
                acquisition_key="a20n-20200101",
            ),
            SelectedFlight(
                icao24="4bb738",
                callsign="B7381",
                firstseen=1_577_880_000,
                lastseen=1_577_880_300,
                msn="B738-001",
                split="train",
                cohorts=frozenset({"B738"}),
                selection_id="sel-b738",
                utc_days=(DAY,),
                acquisition_key="b738-20200101",
            ),
        ),
        digest="two-cohort-boundary-test",
    )


def _prepare_run(tmp_path: Path) -> _RunPaths:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    build_raw_opensky_day(DAY).with_columns(pl.lit("train").alias("meta_split")).write_parquet(
        raw_root / f"{DAY}.parquet"
    )
    artifact = tmp_path / "cleanup-token"
    artifact.write_text("input", encoding="utf-8")
    counters_path = tmp_path / "counters.json"
    counters_path.write_text(json.dumps({"cleanup-token": 0}) + "\n", encoding="utf-8")
    return _RunPaths(
        raw_root=raw_root,
        stores=write_local_era5_grid(tmp_path / "era5"),
        staging_root=tmp_path / "staging",
        published_root=tmp_path / "published",
        journal_path=tmp_path / "journal.jsonl",
        counters_path=counters_path,
        artifact=artifact,
    )


def _run_day(paths: _RunPaths, step_hook: Callable[[str], None]) -> None:
    _day_runner.run_selection_day(
        DAY,
        selection=_selection(),
        opensky_reader=_Reader(paths.raw_root),
        grid_opener=_GridOpener(paths.stores),
        grid_closer=_close_grid,
        era5_fields=ERA5_FIELDS,
        lease=_Lease(),
        staging_root=paths.staging_root,
        published_root=paths.published_root,
        journal_path=paths.journal_path,
        counters_path=paths.counters_path,
        cleanup_artifacts={"cleanup-token": paths.artifact},
        cleanup_decrements=(),
        science_profile=resolve_science_profile(PROFILE_ID),
        versions=VERSIONS,
        local_workers=2,
        max_resident_gib=8.0,
        step_hook=step_hook,
    )


def _partition_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.parquet")))


def _byte_digests(paths: tuple[Path, ...]) -> dict[Path, str]:
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _staged_frames(paths: _RunPaths) -> dict[DayPartitionKey, pl.DataFrame]:
    frames: dict[DayPartitionKey, pl.DataFrame] = {}
    for path in _partition_files(paths.staging_root):
        frame = pl.read_parquet(path)
        key = DayPartitionKey(
            str(frame["cohort"][0]),
            str(frame["meta_selection_day"][0]),
        )
        frames[key] = frame
    return frames


def _committed_events(journal_path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        event for event in read_events(journal_path) if event.get("event") == "day_committed"
    )


def test_run_selection_day_forwards_hook_to_every_boundary(tmp_path: Path) -> None:
    """AC1: the composed daily route exposes every durable boundary in order."""
    paths = _prepare_run(tmp_path)
    boundaries: list[str] = []

    _run_day(paths, boundaries.append)

    assert boundaries == [
        "stage",
        "validate",
        "publish",
        "stage",
        "validate",
        "publish",
        "day_committed",
    ]


class _InjectedBoundaryError(RuntimeError):
    """Fault raised by the daily boundary hook."""


def test_publish_fault_resumes_only_missing_partition(tmp_path: Path) -> None:
    """AC2: a publish fault preserves prior bytes and resume publishes only the missing key."""
    paths = _prepare_run(tmp_path)
    publish_calls = 0

    def fail_second_publish(boundary: str) -> None:
        nonlocal publish_calls
        if boundary == "publish":
            publish_calls += 1
            if publish_calls == 2:
                raise _InjectedBoundaryError(boundary)

    with pytest.raises(_InjectedBoundaryError):
        _run_day(paths, fail_second_publish)

    plan = build_day_plan(_selection(), DAY)
    a20n = next(key for key in plan.partition_keys if key.cohort == "A20N")
    b738 = next(key for key in plan.partition_keys if key.cohort == "B738")
    assert _day_publish.missing_partitions(plan, paths.published_root) == (b738,)
    assert _committed_events(paths.journal_path) == ()

    published_before = _partition_files(paths.published_root)
    assert len(published_before) == 1
    digest_before = _byte_digests(published_before)

    outcome = _day_publish.resume_publication(
        partitions=_staged_frames(paths),
        plan=plan,
        staging_root=paths.staging_root,
        published_root=paths.published_root,
        event_log=paths.journal_path,
    )

    assert outcome.published_now == (b738,)
    assert a20n not in outcome.published_now
    assert _day_publish.missing_partitions(plan, paths.published_root) == ()
    assert _byte_digests(published_before) == digest_before


def test_day_committed_fault_reconciles_exactly_once(tmp_path: Path) -> None:
    """AC3: an interrupted day marker reconciles once without republishing any partition."""
    paths = _prepare_run(tmp_path)

    def fail_before_day_marker(boundary: str) -> None:
        if boundary == "day_committed":
            raise _InjectedBoundaryError(boundary)

    with pytest.raises(_InjectedBoundaryError):
        _run_day(paths, fail_before_day_marker)

    assert _committed_events(paths.journal_path) == ()
    published = _partition_files(paths.published_root)
    assert len(published) == 2
    digests_before = _byte_digests(published)

    _day_commit.reconcile_commit(paths.journal_path, paths.published_root)
    journal_after_first = paths.journal_path.read_bytes()

    assert len(_committed_events(paths.journal_path)) == 1
    assert _byte_digests(published) == digests_before

    _day_commit.reconcile_commit(paths.journal_path, paths.published_root)

    assert paths.journal_path.read_bytes() == journal_after_first
    assert len(_committed_events(paths.journal_path)) == 1
    assert _byte_digests(published) == digests_before
