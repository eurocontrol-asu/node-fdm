from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import polars as pl
import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan
from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events
from node_fdm_pipeline.commands._science_profile import (
    ScienceProfile,
    profile_manifest,
    resolve_science_profile,
)

DAY = "20200102"
FORBIDDEN_PARTITION_EVENTS = {
    "partition_staged",
    "partition_published",
    "stage_partition",
    "publish_partition",
}


def _day_runner() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_runner")


def _day_commit() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_commit")


def _plan() -> DayPlan:
    keys = tuple(
        DayPartitionKey(cohort, DAY) for cohort in ("A319", "A320", "A321", "A332", "A359")
    )
    selection_ids_by_key = {key: frozenset({f"selection-{key.cohort}"}) for key in keys}
    return DayPlan(
        meta_selection_day=DAY,
        partition_keys=keys,
        selection_ids=frozenset().union(*selection_ids_by_key.values()),
        selection_ids_by_key=selection_ids_by_key,
        source_days=(DAY,),
    )


def _profile_digest(profile: ScienceProfile) -> str:
    payload = json.dumps(
        profile_manifest(profile),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _partition_path(root: Path, key: DayPartitionKey) -> Path:
    return (
        root
        / f"cohort={key.cohort}"
        / f"meta_selection_day={key.meta_selection_day}"
        / "data.parquet"
    )


def _publish_partitions(
    published_root: Path,
    journal: Path,
    plan: DayPlan,
    profile: ScienceProfile,
    *,
    journal_events: bool = True,
) -> None:
    profile_digest = _profile_digest(profile)
    for key in plan.partition_keys:
        path = _partition_path(published_root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "cohort": [key.cohort],
                "meta_selection_day": [key.meta_selection_day],
                "meta_source_day": [key.meta_selection_day],
                "selection_id": [next(iter(plan.selection_ids_by_key[key]))],
                "msn": [f"MSN-{key.cohort}"],
                "split": ["train"],
                "profile_id": [profile.profile_id],
                "profile_digest": [profile_digest],
                "code_version": ["test-version"],
            }
        ).write_parquet(path)
        if journal_events:
            append_event(
                journal,
                {
                    "event": "publish_partition",
                    "cohort": key.cohort,
                    "meta_selection_day": key.meta_selection_day,
                    "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "row_count": 1,
                },
            )


def _run_day(  # noqa: PLR0913
    day_runner: ModuleType,
    mocker: MockerFixture,
    tmp_path: Path,
    plan: DayPlan,
    profile: ScienceProfile,
    journal: Path,
) -> object:
    counters_path = tmp_path / "counters.json"
    counters_path.write_text("{}\n", encoding="utf-8")
    mocker.patch.object(day_runner, "build_day_plan", return_value=plan)
    lease = SimpleNamespace(
        acquire=lambda: None,
        heartbeat=lambda: None,
        assert_owned=lambda: None,
        release=lambda: None,
    )
    return day_runner.run_selection_day(
        DAY,
        selection=mocker.sentinel.selection,
        opensky_reader=lambda _day, _ids: pl.DataFrame(),
        grid_opener=lambda _source_day, _field: object(),
        grid_closer=lambda _handle: None,
        era5_fields=("temperature",),
        lease=lease,
        staging_root=tmp_path / "staging",
        published_root=tmp_path / "published",
        journal_path=journal,
        counters_path=counters_path,
        cleanup_artifacts={},
        cleanup_decrements=(),
        science_profile=profile,
        versions={"node-fdm-pipeline": "test-version"},
        local_workers=1,
        max_resident_gib=1.0,
    )


def _partition_event_count(events: list[dict[str, object]]) -> int:
    return sum(event.get("event") in FORBIDDEN_PARTITION_EVENTS for event in events)


@pytest.fixture
def durable_day(
    tmp_path: Path,
) -> tuple[DayPlan, ScienceProfile, Path, Path]:
    plan = _plan()
    profile = resolve_science_profile("opensky26-exp03-v1")
    published_root = tmp_path / "published"
    journal = tmp_path / "journal.jsonl"
    journal.touch()
    return plan, profile, published_root, journal


@pytest.mark.integration
def test_fully_published_day_resumes_directly_at_commit(
    durable_day: tuple[DayPlan, ScienceProfile, Path, Path],
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC2: a fully published day goes directly to commit without republishing."""
    day_runner = _day_runner()
    day_commit = _day_commit()
    plan, profile, published_root, journal = durable_day
    _publish_partitions(published_root, journal, plan, profile)
    before = read_events(journal)
    resume_publication = mocker.spy(day_runner, "resume_publication")

    _run_day(day_runner, mocker, tmp_path, plan, profile, journal)

    after = read_events(journal)
    assert day_commit.load_day_snapshot(journal).day_committed is not None
    assert _partition_event_count(after) == _partition_event_count(before)
    resume_publication.assert_not_called()


@pytest.mark.integration
def test_committed_day_skips_acquisition_and_assembly(
    durable_day: tuple[DayPlan, ScienceProfile, Path, Path],
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC3: a complete commit marker returns a report without acquisition or assembly."""
    day_runner = _day_runner()
    day_commit = _day_commit()
    plan, profile, published_root, journal = durable_day
    _publish_partitions(
        published_root,
        journal,
        plan,
        profile,
        journal_events=False,
    )
    day_commit.commit_day(plan, published_root, journal, profile)
    before = read_events(journal)
    opened_grids = SimpleNamespace(handles={}, opens={}, close=lambda: None)
    acquisition = mocker.patch.object(
        day_runner,
        "_run_parent_acquisition",
        return_value=(pl.DataFrame(), opened_grids),
    )
    assembly = mocker.patch.object(
        day_runner,
        "_assemble_partitions",
        return_value=({}, ()),
    )
    mocker.patch.object(day_runner, "publish_day")

    report = _run_day(day_runner, mocker, tmp_path, plan, profile, journal)

    after = read_events(journal)
    assert isinstance(report, day_runner.DayRunReport)
    assert _partition_event_count(after) == _partition_event_count(before)
    acquisition.assert_not_called()
    assembly.assert_not_called()


@pytest.mark.integration
def test_torn_commit_marker_is_reconciled_during_resume(
    durable_day: tuple[DayPlan, ScienceProfile, Path, Path],
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC4: resume reconciles a torn marker to one complete published snapshot."""
    day_runner = _day_runner()
    day_commit = _day_commit()
    plan, profile, published_root, journal = durable_day
    _publish_partitions(published_root, journal, plan, profile)
    published_snapshot = day_commit.load_day_snapshot(journal)
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"event":"day_committed","meta_selection_day":"20200102"')

    _run_day(day_runner, mocker, tmp_path, plan, profile, journal)

    events = read_events(journal)
    markers = [event for event in events if event.get("event") == "day_committed"]
    snapshot = day_commit.load_day_snapshot(journal)
    assert len(markers) == 1
    assert snapshot.day_committed is not None
    assert snapshot.day_committed.partition_keys == published_snapshot.published_keys
    assert snapshot.day_committed.profile_manifest == profile_manifest(profile)
