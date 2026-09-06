from __future__ import annotations

import importlib
from pathlib import Path

import polars as pl
import pytest


def _frame(cohort: str, selection_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cohort": [cohort],
            "meta_selection_day": ["20200101"],
            "meta_source_day": ["20200101"],
            "selection_id": [selection_id],
            "msn": ["10001"],
            "split": ["train"],
            "profile_id": ["science-v1"],
            "profile_digest": ["profile-digest"],
            "code_version": ["test-version"],
            "raw_timestamp": [1_577_836_900],
        }
    )


@pytest.mark.integration
def test_load_partition_resume_state_reads_real_partition_journal(tmp_path: Path) -> None:
    """AC4: real stage, validate, and publish events restore per-partition state."""
    day_publish = importlib.import_module("node_fdm_pipeline.commands._day_publish")
    day_plan = importlib.import_module("node_fdm_pipeline.commands._day_plan")
    published_key = day_plan.DayPartitionKey("A20N", "20200101")
    staged_key = day_plan.DayPartitionKey("B738", "20200101")
    plan = day_plan.DayPlan(
        meta_selection_day="20200101",
        partition_keys=(published_key, staged_key),
        selection_ids=frozenset({"sel-a20n", "sel-b738"}),
        selection_ids_by_key={
            published_key: frozenset({"sel-a20n"}),
            staged_key: frozenset({"sel-b738"}),
        },
        source_days=("20200101",),
    )
    event_log = tmp_path / "journal.jsonl"
    staging_root = tmp_path / "staging"
    published_root = tmp_path / "published"
    published_frame = _frame(published_key.cohort, "sel-a20n")
    staged_frame = _frame(staged_key.cohort, "sel-b738")

    staged_published = day_publish.stage_partition(
        frame=published_frame,
        key=published_key,
        staging_root=staging_root,
        event_log=event_log,
    )
    day_publish.validate_partition(
        frame=published_frame,
        plan=plan,
        key=published_key,
        event_log=event_log,
    )
    day_publish.publish_partition(
        staged=staged_published,
        published_root=published_root,
        event_log=event_log,
    )
    day_publish.stage_partition(
        frame=staged_frame,
        key=staged_key,
        staging_root=staging_root,
        event_log=event_log,
    )

    result = day_publish.load_partition_resume_state(event_log, plan)

    assert result == {
        published_key: "published",
        staged_key: "staged",
    }
