from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan, build_day_plan
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan


def test_validate_partition_rejects_unauthorised_selection_id() -> None:
    """AC4: validation names a selection id that is absent from the active DayPlan."""
    flight = SelectedFlight(
        icao24="abc001",
        callsign="AXM001",
        firstseen=1_577_836_800,
        lastseen=1_577_837_100,
        msn="10001",
        split="train",
        cohorts=frozenset({"A20N"}),
        selection_id="sel-e100",
        utc_days=("20200101",),
        acquisition_key="acq-e100",
    )
    plan = build_day_plan(
        SelectionPlan(flights=(flight,), digest="selection-digest"),
        "20200101",
    )
    key = plan.partition_keys[0]
    staged_frame = pl.DataFrame(
        {
            "cohort": ["A20N"],
            "meta_selection_day": ["20200101"],
            "meta_source_day": ["20200101"],
            "selection_id": ["sel-e190"],
            "msn": ["19001"],
            "split": ["train"],
            "profile_id": ["science-v1"],
            "profile_digest": ["profile-digest"],
            "code_version": ["test-version"],
            "raw_timestamp": [1_577_836_900],
        }
    )
    day_publish = importlib.import_module("node_fdm_pipeline.commands._day_publish")

    with pytest.raises(day_publish.PartitionValidationError) as excinfo:
        day_publish.validate_partition(frame=staged_frame, plan=plan, key=key)

    assert "sel-e190" in str(excinfo.value)


def test_publish_day_reports_stable_boundaries_once_per_partition(
    mocker: MockerFixture,
) -> None:
    """AC1: publication reports stage, validate, and publish once per partition in order."""
    day_publish = importlib.import_module("node_fdm_pipeline.commands._day_publish")
    day_plan = importlib.import_module("node_fdm_pipeline.commands._day_plan")
    keys = (
        day_plan.DayPartitionKey("A20N", "20200101"),
        day_plan.DayPartitionKey("B738", "20200101"),
    )
    plan = day_plan.DayPlan(
        meta_selection_day="20200101",
        partition_keys=keys,
        selection_ids=frozenset({"sel-a20n", "sel-b738"}),
        selection_ids_by_key={
            keys[0]: frozenset({"sel-a20n"}),
            keys[1]: frozenset({"sel-b738"}),
        },
        source_days=("20200101",),
    )
    frames = {key: pl.DataFrame({"cohort": [key.cohort]}) for key in keys}
    staged = {key: SimpleNamespace(key=key, path=Path(f"{key.cohort}.parquet")) for key in keys}
    boundaries: list[str] = []

    mocker.patch.object(day_publish, "missing_partitions", return_value=keys)
    mocker.patch.object(
        day_publish,
        "stage_partition",
        side_effect=lambda *, key, **_kwargs: staged[key],
    )
    mocker.patch.object(
        day_publish.pl,
        "read_parquet",
        side_effect=[frames[key] for key in keys],
    )
    validate = mocker.patch.object(day_publish, "validate_partition")
    publish = mocker.patch.object(day_publish, "publish_partition")

    outcome = day_publish.publish_day(
        partitions=frames,
        plan=plan,
        staging_root=Path("staging"),
        published_root=Path("published"),
        event_log=Path("journal.jsonl"),
        step_hook=boundaries.append,
    )

    assert outcome.published_now == keys
    assert boundaries == ["stage", "validate", "publish"] * 2
    assert validate.call_count == len(keys)
    assert publish.call_count == len(keys)


def _resume_plan() -> DayPlan:
    keys = tuple(
        DayPartitionKey(cohort, "20200101") for cohort in ("A20N", "B738", "B739", "E190", "E195")
    )
    selection_ids_by_key = {key: frozenset({f"sel-{key.cohort.lower()}"}) for key in keys}
    return DayPlan(
        meta_selection_day="20200101",
        partition_keys=keys,
        selection_ids=frozenset().union(*selection_ids_by_key.values()),
        selection_ids_by_key=selection_ids_by_key,
        source_days=("20200101",),
    )


def _resume_event(
    event: str,
    key: DayPartitionKey,
    *,
    selection_id: str | None = None,
    digest: str = "digest",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": event,
        "cohort": key.cohort,
        "meta_selection_day": key.meta_selection_day,
        "digest": digest,
        "row_count": 1,
    }
    if selection_id is not None:
        payload["selection_id"] = selection_id
    return payload


def test_partition_resume_state_returns_highest_boundary_by_partition() -> None:
    """AC1: each planned partition receives its highest durable resume boundary."""
    day_publish = importlib.import_module("node_fdm_pipeline.commands._day_publish")
    plan = _resume_plan()
    staged, validated, published, pending_a, pending_b = plan.partition_keys
    events = [
        _resume_event("stage_partition", staged),
        _resume_event("stage_partition", validated),
        _resume_event("validate_partition", validated),
        _resume_event("publish_partition", published),
        _resume_event("stage_partition", published),
        _resume_event("validate_partition", published),
    ]

    result = day_publish.partition_resume_state(events, plan)

    assert result == {
        staged: "staged",
        validated: "validated",
        published: "published",
        pending_a: "pending",
        pending_b: "pending",
    }


def test_partition_resume_state_ignores_out_of_scope_events() -> None:
    """AC2: events from another UTC day or selection stay outside the active plan."""
    day_publish = importlib.import_module("node_fdm_pipeline.commands._day_publish")
    plan = _resume_plan()
    wrong_day_key, wrong_selection_key = plan.partition_keys[:2]
    wrong_day_event = _resume_event("stage_partition", wrong_day_key)
    wrong_day_event["meta_selection_day"] = "20200102"
    wrong_selection_event = _resume_event(
        "stage_partition",
        wrong_selection_key,
        selection_id="sel-outside-plan",
    )

    result = day_publish.partition_resume_state(
        [wrong_day_event, wrong_selection_event],
        plan,
    )

    assert result[wrong_day_key] == "pending"
    assert result[wrong_selection_key] == "pending"


def test_partition_resume_state_ignores_incomplete_terminal_event() -> None:
    """AC3: an incomplete tail event cannot advance a complete durable boundary."""
    day_publish = importlib.import_module("node_fdm_pipeline.commands._day_publish")
    plan = _resume_plan()
    key = plan.partition_keys[0]
    incomplete_validated = _resume_event("validate_partition", key)
    incomplete_validated.pop("digest")

    result = day_publish.partition_resume_state(
        [_resume_event("stage_partition", key), incomplete_validated],
        plan,
    )

    assert result[key] == "staged"
