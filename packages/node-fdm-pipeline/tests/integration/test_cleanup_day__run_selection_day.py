from __future__ import annotations

import json
from pathlib import Path

import pytest

from node_fdm_pipeline.commands._day_cleanup import cleanup_day, resume_cleanup
from node_fdm_pipeline.commands._day_commit import DayCommit, DaySnapshot
from node_fdm_pipeline.commands._day_plan import DayPartitionKey
from node_fdm_pipeline.commands._fleet_manifest import read_events

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
