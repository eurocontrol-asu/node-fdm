from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from node_fdm_pipeline.commands._day_plan import build_day_plan
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan
from node_fdm_pipeline.commands._science_profile import resolve_science_profile

VERSIONS = {"node-fdm-pipeline": "test-code-v1"}
PROFILE_ID = "opensky26-exp03-v1"


def _day_assemble_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_assemble")


@pytest.mark.integration
def test_a20n_partition_holds_exactly_two_selected_trajectories(tmp_path: Path) -> None:
    """AC1: the persisted A20N input yields exactly its two selected trajectories."""
    selection = SelectionPlan(
        flights=(
            SelectedFlight(
                icao24="a00001",
                callsign="A20N1",
                firstseen=1_577_880_000,
                lastseen=1_577_883_600,
                msn="A20N-001",
                split="train",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-a20n",
                utc_days=("20200101",),
                acquisition_key="a20n-20200101",
            ),
            SelectedFlight(
                icao24="a00002",
                callsign="A20N2",
                firstseen=1_577_835_600,
                lastseen=1_577_838_000,
                msn="A20N-002",
                split="test",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-a20n-midnight",
                utc_days=("20191231", "20200101"),
                acquisition_key="a20n-midnight",
            ),
            SelectedFlight(
                icao24="b00001",
                callsign="B7381",
                firstseen=1_577_890_000,
                lastseen=1_577_891_000,
                msn="B738-001",
                split="train",
                cohorts=frozenset({"B738"}),
                selection_id="sel-b738",
                utc_days=("20200101",),
                acquisition_key="b738-20200101",
            ),
        ),
        digest="selection-plan-test-digest",
    )
    recorded = pl.DataFrame(
        {
            "selection_id": [
                "sel-a20n-midnight",
                "sel-a20n",
                "sel-b738",
                "sel-a20n-midnight",
                "sel-a20n",
                "sel-a20n-midnight",
                "sel-a20n-midnight",
            ],
            "cohort": ["A20N", "A20N", "B738", "A20N", "A20N", "A20N", "A20N"],
            "meta_source_day": [
                "20200101",
                "20200101",
                "20200101",
                "20191231",
                "20200101",
                "20191231",
                "20200101",
            ],
            "raw_timestamp": [
                1_577_837_400,
                1_577_880_000,
                1_577_890_000,
                1_577_835_600,
                1_577_883_600,
                1_577_836_500,
                1_577_838_000,
            ],
            "raw_icao24": ["a00002", "a00001", "b00001", "a00002", "a00001", "a00002", "a00002"],
            "meta_split": ["test", "train", "train", "test", "train", "test", "test"],
            "altitude": [120.0, 1_000.0, 2_000.0, 80.0, 1_200.0, 100.0, 150.0],
        }
    )
    parquet_path = tmp_path / "recorded-day.parquet"
    recorded.write_parquet(parquet_path)
    restored = pl.read_parquet(parquet_path)
    module = _day_assemble_module()
    result = module.assemble_partition(
        restored,
        build_day_plan(selection, "20200101"),
        ("A20N", "20200101"),
        resolve_science_profile(PROFILE_ID),
        VERSIONS,
    )

    assert set(result["selection_id"]) == {"sel-a20n", "sel-a20n-midnight"}
    assert result.height == 6
