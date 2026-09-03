from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands._day_plan import build_day_plan
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
