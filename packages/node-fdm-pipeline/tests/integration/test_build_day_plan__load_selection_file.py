"""Integration test for building a day plan from a selection file."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from node_fdm_pipeline.commands._fleet_selection import load_selection_file

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


@pytest.mark.integration
def test_build_day_plan_from_loaded_selection_file(tmp_path: Path) -> None:
    """AC1: a file-backed selection produces the exact 20200101 day boundary."""
    selection_path = tmp_path / "selection.csv"
    selection_path.write_text(_SELECTION_CSV)
    selection_source = load_selection_file(selection_path)
    assert selection_source.plan is not None
    day_plan = _load_day_plan()

    plan = day_plan.build_day_plan(selection_source.plan, "20200101")

    assert plan.partition_keys == (("A20N", "20200101"), ("B738", "20200101"))
    assert plan.selection_ids == {
        "sel-a20n",
        "sel-a20n-midnight",
        "sel-b738",
    }
