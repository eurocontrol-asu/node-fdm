"""Unit tests for the selection-day execution boundary."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

from node_fdm_pipeline.commands._fleet_selection import (
    SelectionPlan,
    compile_selection_csv,
)

_SELECTION_CSV = (
    "selection_id,icao24,callsign,firstseen,lastseen,msn,split,cohorts,utc_days\n"
    "sel-a20n,a20a20,A20N01,1577872800,1577876400,A20-1,train,A20N,\n"
    "sel-a20n-midnight,a20a21,A20N02,1577835600,1577838000,A20-2,train,A20N,\n"
    "sel-b738,b73b38,B73801,1577880000,1577883600,B738-1,test,B738,\n"
    "sel-e190,e19e90,E19001,1577959200,1577962800,E190-1,train,E190,\n"
    "sel-crj9,c9c9c9,CRJ901,1577966400,1577970000,CRJ9-1,test,CRJ9,\n"
    "sel-cl35,c13535,CL3501,1577973600,1577977200,CL35-1,test,CL35,\n"
)


def _load_day_plan() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_plan")


@pytest.fixture
def selection() -> SelectionPlan:
    """Compile the six-flight, two-day selection entirely in memory."""
    return compile_selection_csv(_SELECTION_CSV)


def test_build_day_plan_holds_20200101_partitions(selection: SelectionPlan) -> None:
    """AC1: the 20200101 plan contains exactly its partitions and selection ids."""
    day_plan = _load_day_plan()

    plan = day_plan.build_day_plan(selection, "20200101")

    assert plan.partition_keys == (("A20N", "20200101"), ("B738", "20200101"))
    assert plan.selection_ids == {
        "sel-a20n",
        "sel-a20n-midnight",
        "sel-b738",
    }


def test_build_day_plan_holds_20200102_partitions(selection: SelectionPlan) -> None:
    """AC2: the 20200102 plan contains exactly its three cohort partitions."""
    day_plan = _load_day_plan()

    plan = day_plan.build_day_plan(selection, "20200102")

    assert plan.partition_keys == (
        ("CL35", "20200102"),
        ("CRJ9", "20200102"),
        ("E190", "20200102"),
    )
    assert plan.selection_ids == {"sel-e190", "sel-crj9", "sel-cl35"}


def test_build_day_plan_orders_and_deduplicates_source_days(
    selection: SelectionPlan,
) -> None:
    """AC3: a midnight-crossing selection contributes both UTC source days once."""
    day_plan = _load_day_plan()

    plan = day_plan.build_day_plan(selection, "20200101")

    assert plan.source_days == ("20191231", "20200101")


def test_require_in_day_scope_refuses_foreign_partition(
    selection: SelectionPlan,
) -> None:
    """AC4: a cohort/day key outside the active plan is rejected and identified."""
    day_plan = _load_day_plan()
    plan = day_plan.build_day_plan(selection, "20200101")

    with pytest.raises(day_plan.DayScopeViolation) as excinfo:
        day_plan.require_in_day_scope(
            plan,
            cohort="E190",
            meta_selection_day="20200102",
        )

    message = str(excinfo.value)
    assert "E190" in message
    assert "20200102" in message


def test_require_selection_in_scope_refuses_foreign_selection(
    selection: SelectionPlan,
) -> None:
    """AC5: a selection id outside the active plan is rejected and identified."""
    day_plan = _load_day_plan()
    plan = day_plan.build_day_plan(selection, "20200101")

    with pytest.raises(day_plan.DayScopeViolation) as excinfo:
        day_plan.require_selection_in_scope(plan, "sel-e190")

    assert "sel-e190" in str(excinfo.value)
