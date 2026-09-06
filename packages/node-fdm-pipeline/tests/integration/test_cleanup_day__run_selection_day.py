from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
import pytest_mock

from node_fdm_pipeline.commands import _day_runner as day_runner
from node_fdm_pipeline.commands._day_cleanup import cleanup_day, resume_cleanup
from node_fdm_pipeline.commands._day_commit import DayCommit, DaySnapshot
from node_fdm_pipeline.commands._day_plan import DayPartitionKey
from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan
from node_fdm_pipeline.commands._science_profile import ScienceProfile

pytestmark = pytest.mark.integration


def _committed_snapshot() -> DaySnapshot:
    key = DayPartitionKey(cohort="A20N", meta_selection_day="20200101")
    return DaySnapshot(
        meta_selection_day="20200101",
        published_keys=(key,),
        day_committed=DayCommit(
            meta_selection_day="20200101",
            partition_keys=(key,),
            profile_manifest={"profile_id": "test"},
        ),
    )


def test_delete_fault_resumes_without_double_decrement_or_collateral_deletion(
    tmp_path: Path,
) -> None:
    """AC2: a delete fault resumes atomically without replaying durable decrements."""
    journal_path = tmp_path / "day-journal.jsonl"
    counters_path = tmp_path / "dependency-counters.json"
    counters_path.write_text(
        json.dumps(
            {
                "era5/referenced": 2,
                "raw/referenced": 2,
                "era5/zero": 0,
                "raw/zero": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {
        artifact_id: tmp_path / "inputs" / artifact_id
        for artifact_id in (
            "era5/referenced",
            "raw/referenced",
            "era5/zero",
            "raw/zero",
        )
    }
    for artifact in artifacts.values():
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"source-artifact")

    committed_output = tmp_path / "published" / "20200101" / "part.parquet"
    committed_output.parent.mkdir(parents=True)
    committed_output.write_bytes(b"immutable-committed-output")
    committed_bytes = committed_output.read_bytes()

    def fail_on_first_delete(step: str, artifact_id: str) -> None:
        if step == "delete" and artifact_id == "era5/zero":
            raise RuntimeError("injected delete-boundary failure")

    with pytest.raises(RuntimeError, match="delete-boundary"):
        cleanup_day(
            _committed_snapshot(),
            journal_path=journal_path,
            counters_path=counters_path,
            artifacts=artifacts,
            decrements=("era5/referenced", "raw/referenced"),
            step_hook=fail_on_first_delete,
        )

    assert artifacts["era5/referenced"].exists()
    assert artifacts["raw/referenced"].exists()
    assert not artifacts["era5/zero"].exists()
    assert artifacts["raw/zero"].exists()
    assert committed_output.read_bytes() == committed_bytes

    events_before_resume = read_events(journal_path)
    decrement_counts_before = {
        artifact_id: sum(
            event.get("event") == "counter_decremented" and event.get("artifact_id") == artifact_id
            for event in events_before_resume
        )
        for artifact_id in ("era5/referenced", "raw/referenced")
    }
    resumed_boundaries: list[tuple[str, str]] = []

    resume_cleanup(
        _committed_snapshot(),
        journal_path=journal_path,
        counters_path=counters_path,
        artifacts=artifacts,
        decrements=("era5/referenced", "raw/referenced"),
        step_hook=lambda step, artifact_id: resumed_boundaries.append((step, artifact_id)),
    )

    assert artifacts["era5/referenced"].exists()
    assert artifacts["raw/referenced"].exists()
    assert not artifacts["era5/zero"].exists()
    assert not artifacts["raw/zero"].exists()
    assert committed_output.read_bytes() == committed_bytes
    assert resumed_boundaries == [("delete", "raw/zero")]

    events_after_resume = read_events(journal_path)
    decrement_counts_after = {
        artifact_id: sum(
            event.get("event") == "counter_decremented" and event.get("artifact_id") == artifact_id
            for event in events_after_resume
        )
        for artifact_id in ("era5/referenced", "raw/referenced")
    }
    assert (
        decrement_counts_before
        == decrement_counts_after
        == {
            "era5/referenced": 1,
            "raw/referenced": 1,
        }
    )


@dataclass(frozen=True)
class _CleanupResumeScenario:
    journal_path: Path
    artifacts: dict[str, Path]
    resumed_counters: dict[str, int]
    reference_counters: dict[str, int]
    events_before_resume: int
    resume_calls: int


def _append_commit_marker(journal_path: Path) -> None:
    append_event(
        journal_path,
        {
            "event": "day_committed",
            "meta_selection_day": "20200101",
            "partition_keys": [["A20N", "20200101"]],
            "profile_manifest": {"profile_id": "test"},
        },
    )


def _write_cleanup_inputs(root: Path) -> tuple[Path, dict[str, Path]]:
    counters_path = root / "dependency-counters.json"
    artifact_ids = (
        "era5/pressure",
        "era5/temperature",
        "raw/flight",
        "raw/state-vectors",
        "raw/tracks",
    )
    counters_path.parent.mkdir(parents=True, exist_ok=True)
    counters_path.write_text(
        json.dumps(dict.fromkeys(artifact_ids, 1)) + "\n",
        encoding="utf-8",
    )
    artifacts = {artifact_id: root / "inputs" / artifact_id for artifact_id in artifact_ids}
    for artifact in artifacts.values():
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"source-artifact")
    return counters_path, artifacts


def _unexpected_external_access(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("cleanup-only resume attempted external acquisition")


def _unexpected_opensky_access(
    source_day: str,
    selection_ids: frozenset[str],
) -> pl.DataFrame:
    raise AssertionError("cleanup-only resume attempted OpenSky acquisition")


def _unexpected_grid_close(handle: object) -> None:
    raise AssertionError("cleanup-only resume attempted to close an ERA5 grid")


class _UnexpectedLease:
    def acquire(self) -> None:
        raise AssertionError("cleanup-only resume attempted to acquire a lease")

    def heartbeat(self) -> None:
        raise AssertionError("cleanup-only resume attempted to renew a lease")

    def assert_owned(self) -> None:
        raise AssertionError("cleanup-only resume attempted to inspect a lease")

    def release(self) -> None:
        raise AssertionError("cleanup-only resume attempted to release a lease")


@pytest.fixture
def cleanup_resume_scenario(
    tmp_path: Path,
    mocker: pytest_mock.MockerFixture,
) -> _CleanupResumeScenario:
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"
    reference_journal = reference_root / "day-journal.jsonl"
    resumed_journal = resumed_root / "day-journal.jsonl"
    reference_counters_path, reference_artifacts = _write_cleanup_inputs(reference_root)
    resumed_counters_path, resumed_artifacts = _write_cleanup_inputs(resumed_root)
    _append_commit_marker(reference_journal)
    _append_commit_marker(resumed_journal)

    artifact_ids = tuple(resumed_artifacts)
    cleanup_day(
        _committed_snapshot(),
        journal_path=reference_journal,
        counters_path=reference_counters_path,
        artifacts=reference_artifacts,
        decrements=artifact_ids,
    )

    deleted_count = 0

    def fail_after_second_delete(step: str, _artifact_id: str) -> None:
        nonlocal deleted_count
        if step != "delete":
            return
        deleted_count += 1
        if deleted_count == 2:
            raise RuntimeError("injected failure after two deletions")

    with pytest.raises(RuntimeError, match="after two deletions"):
        cleanup_day(
            _committed_snapshot(),
            journal_path=resumed_journal,
            counters_path=resumed_counters_path,
            artifacts=resumed_artifacts,
            decrements=artifact_ids,
            step_hook=fail_after_second_delete,
        )

    assert deleted_count == 2
    assert sum(not path.exists() for path in resumed_artifacts.values()) == 2
    events_before_resume = len(read_events(resumed_journal))
    resume_spy = mocker.patch.object(
        day_runner,
        "resume_cleanup",
        wraps=resume_cleanup,
        create=True,
    )
    # The daily runner must select the explicit durable-resume boundary.
    # The existing cleanup implementation remains live so the assertions also
    # verify the resulting files, counters, and journal.

    day_runner.run_selection_day(
        "20200101",
        selection=SelectionPlan(flights=(), digest="empty-selection"),
        opensky_reader=_unexpected_opensky_access,
        grid_opener=_unexpected_external_access,
        grid_closer=_unexpected_grid_close,
        era5_fields=(),
        lease=_UnexpectedLease(),
        staging_root=resumed_root / "staging",
        published_root=resumed_root / "published",
        journal_path=resumed_journal,
        counters_path=resumed_counters_path,
        cleanup_artifacts=resumed_artifacts,
        cleanup_decrements=artifact_ids,
        science_profile=ScienceProfile(
            profile_id="test",
            source_commit="test-commit",
            artifact_digests={
                "retained.csv": "retained",
                "grids.json": "grids",
                "metrics.yaml": "metrics",
            },
        ),
        versions={},
        local_workers=1,
        max_resident_gib=1.0,
    )

    return _CleanupResumeScenario(
        journal_path=resumed_journal,
        artifacts=resumed_artifacts,
        resumed_counters=json.loads(resumed_counters_path.read_text(encoding="utf-8")),
        reference_counters=json.loads(reference_counters_path.read_text(encoding="utf-8")),
        events_before_resume=events_before_resume,
        resume_calls=resume_spy.call_count,
    )


def test_run_selection_day_resumes_only_remaining_artifact_deletions(
    cleanup_resume_scenario: _CleanupResumeScenario,
) -> None:
    """AC1: a restarted day deletes only the three artifacts still pending."""
    events = read_events(cleanup_resume_scenario.journal_path)
    deletion_counts = Counter(
        event["artifact_id"] for event in events if event.get("event") == "artifact_deleted"
    )

    assert cleanup_resume_scenario.resume_calls == 1
    assert all(not path.exists() for path in cleanup_resume_scenario.artifacts.values())
    assert deletion_counts == Counter(dict.fromkeys(cleanup_resume_scenario.artifacts, 1))


def test_run_selection_day_does_not_decrement_counters_twice(
    cleanup_resume_scenario: _CleanupResumeScenario,
) -> None:
    """AC2: durable decrements survive the fault and match a fault-free cleanup."""
    assert cleanup_resume_scenario.resume_calls == 1
    assert (
        cleanup_resume_scenario.resumed_counters
        == cleanup_resume_scenario.reference_counters
        == dict.fromkeys(cleanup_resume_scenario.artifacts, 0)
    )


def test_cleanup_only_resume_skips_acquisition_and_publication(
    cleanup_resume_scenario: _CleanupResumeScenario,
) -> None:
    """AC3: cleanup-only restart neither acquires inputs nor republishes partitions."""
    resumed_events = read_events(cleanup_resume_scenario.journal_path)[
        cleanup_resume_scenario.events_before_resume :
    ]

    assert cleanup_resume_scenario.resume_calls == 1
    assert not {event.get("event") for event in resumed_events} & {
        "partition_staged",
        "partition_published",
    }
