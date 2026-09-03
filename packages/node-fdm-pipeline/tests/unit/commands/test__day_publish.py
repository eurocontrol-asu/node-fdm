from __future__ import annotations

import importlib

import polars as pl
import pytest

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
