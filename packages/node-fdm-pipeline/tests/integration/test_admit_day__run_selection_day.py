from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from node_fdm_pipeline.commands import _day_cleanup, _day_commit
from node_fdm_pipeline.commands._day_runner import DayRunReport, run_selection_day
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan
from node_fdm_pipeline.commands._science_profile import resolve_science_profile

pytestmark = pytest.mark.integration

ERA5_FIELDS = ("temperature", "u_component_of_wind")
VERSIONS = {"node-fdm-pipeline": "test-code-v1"}
PROFILE_ID = "opensky26-exp03-v1"


@dataclass
class _RecordedOpenSkyReader:
    frame: pl.DataFrame
    pids: list[int] = field(default_factory=list)

    def __call__(self, source_day: str, selection_ids: frozenset[str]) -> pl.DataFrame:
        self.pids.append(os.getpid())
        return self.frame.filter(
            (pl.col("meta_source_day") == source_day) & pl.col("selection_id").is_in(selection_ids)
        )


@dataclass
class _RecordedGridOpener:
    pids: list[int] = field(default_factory=list)

    def __call__(self, source_day: str, field_name: str) -> tuple[str, str]:
        self.pids.append(os.getpid())
        return source_day, field_name


@dataclass
class _RecordedLease:
    calls: list[tuple[str, int]] = field(default_factory=list)

    def acquire(self) -> None:
        self.calls.append(("acquire", os.getpid()))

    def heartbeat(self) -> None:
        self.calls.append(("heartbeat", os.getpid()))

    def assert_owned(self) -> None:
        self.calls.append(("possession", os.getpid()))

    def release(self) -> None:
        self.calls.append(("release", os.getpid()))


@dataclass
class _DayFixture:
    selection: SelectionPlan
    reader: _RecordedOpenSkyReader
    grid_opener: _RecordedGridOpener
    staging_root: Path
    published_root: Path
    journal_path: Path
    counters_path: Path
    artifacts: dict[str, Path]
    decrements: tuple[str, ...]


def _runner() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_runner")


def _selection() -> SelectionPlan:
    return SelectionPlan(
        flights=(
            SelectedFlight(
                icao24="a00001",
                callsign="A20N1",
                firstseen=1_577_880_000,
                lastseen=1_577_883_600,
                msn="A20N-001",
                split="train",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-a20n",
                utc_days=("20200101",),
                acquisition_key="a20n-20200101",
            ),
            SelectedFlight(
                icao24="a00002",
                callsign="A20N2",
                firstseen=1_577_835_600,
                lastseen=1_577_838_000,
                msn="A20N-002",
                split="test",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-a20n-midnight",
                utc_days=("20191231", "20200101"),
                acquisition_key="a20n-midnight",
            ),
            SelectedFlight(
                icao24="b00001",
                callsign="B7381",
                firstseen=1_577_890_000,
                lastseen=1_577_891_000,
                msn="B738-001",
                split="train",
                cohorts=frozenset({"B738"}),
                selection_id="sel-b738",
                utc_days=("20200101",),
                acquisition_key="b738-20200101",
            ),
            SelectedFlight(
                icao24="b00002",
                callsign="B7382",
                firstseen=1_577_970_000,
                lastseen=1_577_971_000,
                msn="B738-002",
                split="test",
                cohorts=frozenset({"B738"}),
                selection_id="sel-next-day",
                utc_days=("20200102",),
                acquisition_key="b738-20200102",
            ),
        ),
        digest="recorded-two-day-selection",
    )


def _recorded_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "selection_id": [
                "sel-a20n-midnight",
                "sel-a20n",
                "sel-b738",
                "sel-a20n-midnight",
                "sel-a20n",
                "sel-a20n-midnight",
                "sel-a20n-midnight",
                "sel-next-day",
            ],
            "cohort": ["A20N", "A20N", "B738", "A20N", "A20N", "A20N", "A20N", "B738"],
            "meta_source_day": [
                "20200101",
                "20200101",
                "20200101",
                "20191231",
                "20200101",
                "20191231",
                "20200101",
                "20200102",
            ],
            "raw_timestamp": [
                1_577_837_400,
                1_577_880_000,
                1_577_890_000,
                1_577_835_600,
                1_577_883_600,
                1_577_836_500,
                1_577_838_000,
                1_577_970_000,
            ],
            "raw_icao24": [
                "a00002",
                "a00001",
                "b00001",
                "a00002",
                "a00001",
                "a00002",
                "a00002",
                "b00002",
            ],
            "meta_split": [
                "test",
                "train",
                "train",
                "test",
                "train",
                "test",
                "test",
                "test",
            ],
            "altitude": [120.0, 1_000.0, 2_000.0, 80.0, 1_200.0, 100.0, 150.0, 900.0],
        }
    )


@pytest.fixture
def day_fixture(tmp_path: Path) -> _DayFixture:
    artifact = tmp_path / "inputs" / "cleanup-token"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("recorded input", encoding="utf-8")
    counters_path = tmp_path / "dependency-counters.json"
    counters_path.write_text(json.dumps({"cleanup-token": 0}) + "\n", encoding="utf-8")
    return _DayFixture(
        selection=_selection(),
        reader=_RecordedOpenSkyReader(_recorded_frame()),
        grid_opener=_RecordedGridOpener(),
        staging_root=tmp_path / "staging",
        published_root=tmp_path / "published",
        journal_path=tmp_path / "day-journal.jsonl",
        counters_path=counters_path,
        artifacts={"cleanup-token": artifact},
        decrements=(),
    )


def _run_day(
    fixture: _DayFixture,
    day: str,
    lease: _RecordedLease,
    *,
    cleanup_step_hook: object | None = None,
) -> DayRunReport:
    return run_selection_day(
        day,
        selection=fixture.selection,
        opensky_reader=fixture.reader,
        grid_opener=fixture.grid_opener,
        grid_closer=lambda _handle: None,
        era5_fields=ERA5_FIELDS,
        lease=lease,
        staging_root=fixture.staging_root,
        published_root=fixture.published_root,
        journal_path=fixture.journal_path,
        counters_path=fixture.counters_path,
        cleanup_artifacts=fixture.artifacts,
        cleanup_decrements=fixture.decrements,
        science_profile=resolve_science_profile(PROFILE_ID),
        versions=VERSIONS,
        local_workers=2,
        max_resident_gib=8.0,
        cleanup_step_hook=cleanup_step_hook,
    )


def _event_day(event: dict[str, object]) -> object:
    return event.get("meta_selection_day", event.get("day"))


def _partition_frames(root: Path) -> tuple[pl.DataFrame, ...]:
    return tuple(pl.read_parquet(path) for path in sorted(root.rglob("*.parquet")))


def _fail_after_cleanup_deletion(step: str, _artifact_id: str) -> None:
    if step == "artifact_deleted":
        raise RuntimeError("injected cleanup failure")


def _leave_failed_cleanup(fixture: _DayFixture) -> None:
    with pytest.raises(RuntimeError, match="injected cleanup failure"):
        _run_day(
            fixture,
            "20200101",
            _RecordedLease(),
            cleanup_step_hook=_fail_after_cleanup_deletion,
        )


def _resume_first_day_cleanup(fixture: _DayFixture) -> None:
    snapshot = _day_commit.load_day_snapshot(fixture.journal_path)
    _day_cleanup.resume_cleanup(
        snapshot,
        journal_path=fixture.journal_path,
        counters_path=fixture.counters_path,
        artifacts=fixture.artifacts,
        decrements=fixture.decrements,
    )


def test_next_day_is_refused_before_its_first_step(day_fixture: _DayFixture) -> None:
    """AC2: an independent next-day call cannot start or stage after cleanup_failed."""
    _leave_failed_cleanup(day_fixture)
    reads_before = len(day_fixture.reader.pids)
    opens_before = len(day_fixture.grid_opener.pids)
    refused_lease = _RecordedLease()

    with pytest.raises(_runner().PreviousDayCleanupFailed, match="20200101"):
        _run_day(day_fixture, "20200102", refused_lease)

    events = read_events(day_fixture.journal_path)
    assert not any(
        event.get("event") == "day_started" and _event_day(event) == "20200102" for event in events
    )
    assert not any(
        frame.height and frame["meta_selection_day"][0] == "20200102"
        for frame in _partition_frames(day_fixture.staging_root)
    )
    assert len(day_fixture.reader.pids) == reads_before
    assert len(day_fixture.grid_opener.pids) == opens_before
    assert refused_lease.calls == []


def test_resumed_cleanup_admits_only_the_next_day(day_fixture: _DayFixture) -> None:
    """AC3: resuming 20200101 cleanup admits 20200102 without replaying prior work."""
    _leave_failed_cleanup(day_fixture)
    _resume_first_day_cleanup(day_fixture)
    resume_boundary = len(read_events(day_fixture.journal_path))

    _run_day(day_fixture, "20200102", _RecordedLease())

    events = read_events(day_fixture.journal_path)
    assert any(
        event.get("event") == "day_committed" and _event_day(event) == "20200102"
        for event in events
    )
    forbidden_prefixes = ("acquisition", "publication", "partition_")
    assert not any(
        _event_day(event) == "20200101"
        and str(event.get("event", "")).startswith(forbidden_prefixes)
        for event in events[resume_boundary:]
    )


def test_day_publishes_only_its_planned_partitions(day_fixture: _DayFixture) -> None:
    """AC4: the recorded day publishes exactly its two keys and three selection ids."""
    _run_day(day_fixture, "20200101", _RecordedLease())

    frames = _partition_frames(day_fixture.published_root)
    keys = tuple(
        sorted((str(frame["cohort"][0]), str(frame["meta_selection_day"][0])) for frame in frames)
    )
    selection_ids = {
        str(selection_id) for frame in frames for selection_id in frame["selection_id"].to_list()
    }
    assert keys == (("A20N", "20200101"), ("B738", "20200101"))
    assert selection_ids == {"sel-a20n", "sel-a20n-midnight", "sel-b738"}


def test_external_access_stays_in_parent_process(day_fixture: _DayFixture) -> None:
    """AC5: OpenSky and every lease operation stay in the parent under two workers."""
    lease = _RecordedLease()

    report = _run_day(day_fixture, "20200101", lease)

    parent_pid = os.getpid()
    worker_pids = tuple(report.worker_pids)
    assert set(report.external_access_pids) == {parent_pid}
    assert set(day_fixture.reader.pids) == {parent_pid}
    assert {"acquire", "heartbeat", "possession", "release"} <= {
        operation for operation, _pid in lease.calls
    }
    assert {pid for _operation, pid in lease.calls} == {parent_pid}
    assert worker_pids
    assert len(worker_pids) == len(set(worker_pids))
    assert parent_pid not in worker_pids


def test_shared_era5_grids_open_once_per_day_and_field(day_fixture: _DayFixture) -> None:
    """AC6: two cohorts share one ERA5 open for every UTC source-day field."""
    report = _run_day(day_fixture, "20200101", _RecordedLease())

    expected = {
        (source_day, field_name): 1
        for source_day in ("20191231", "20200101")
        for field_name in ERA5_FIELDS
    }
    assert report.grid_opens == expected
