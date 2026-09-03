from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace

import pytest

from node_fdm_pipeline.commands._day_plan import DayPartitionKey, build_day_plan
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan


def _day_commit() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_commit")


def _selection_plan() -> SelectionPlan:
    flight = SelectedFlight(
        icao24="abc123",
        callsign="TEST01",
        firstseen=1_577_923_200,
        lastseen=1_577_926_800,
        msn="MSN-1",
        split="train",
        cohorts=frozenset({"CL35", "CRJ9", "E190"}),
        selection_id="selection-1",
        utc_days=("20200102",),
        acquisition_key="acquisition-1",
    )
    return SelectionPlan(flights=(flight,), digest="selection-digest")


def test_purge_is_forbidden_while_day_is_not_committed() -> None:
    """AC4: a fully published day remains purge-locked until its commit marker exists."""
    day_commit = _day_commit()
    published_keys = tuple(
        DayPartitionKey(cohort, "20200102") for cohort in ("A319", "A320", "A321", "A332", "A359")
    )
    snapshot = SimpleNamespace(
        meta_selection_day="20200102",
        published_keys=published_keys,
        day_committed=None,
    )

    with pytest.raises(day_commit.PurgeForbiddenBeforeCommit) as excinfo:
        day_commit.assert_purge_allowed(snapshot)

    assert "20200102" in str(excinfo.value)


def test_missing_expected_keys_are_reported_in_plan_order() -> None:
    """AC5: missing expected keys retain the deterministic order declared by the plan."""
    day_commit = _day_commit()
    plan = build_day_plan(_selection_plan(), "20200102")
    published_keys = {DayPartitionKey("E190", "20200102")}

    result = day_commit.missing_expected_keys(plan, published_keys)

    assert result == (
        DayPartitionKey("CL35", "20200102"),
        DayPartitionKey("CRJ9", "20200102"),
    )
