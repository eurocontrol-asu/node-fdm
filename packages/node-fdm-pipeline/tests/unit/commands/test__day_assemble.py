from __future__ import annotations

import importlib
from types import ModuleType
from typing import cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from _daily_corpus import build_raw_opensky_day
from node_fdm_pipeline.commands._day_plan import DayScopeViolation, build_day_plan
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan
from node_fdm_pipeline.commands._science_profile import resolve_science_profile

VERSIONS = {"node-fdm-pipeline": "test-code-v1"}
PROFILE_ID = "opensky26-exp03-v1"


def _day_assemble_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_assemble")


@pytest.fixture
def selection_plan() -> SelectionPlan:
    return SelectionPlan(
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
                firstseen=1_577_923_080,
                lastseen=1_577_923_320,
                msn="A20N-002",
                split="test",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-a20n-midnight",
                utc_days=("20200101", "20200102"),
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
            SelectedFlight(
                icao24="e00001",
                callsign="E1901",
                firstseen=1_577_966_400,
                lastseen=1_577_967_000,
                msn="E190-001",
                split="validation",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-e190",
                utc_days=("20200102",),
                acquisition_key="e190-20200102",
            ),
        ),
        digest="selection-plan-test-digest",
    )


@pytest.fixture
def recorded_frame() -> pl.DataFrame:
    return pl.DataFrame(
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
                "20200102",
                "20200101",
                "20200101",
                "20200101",
                "20200101",
                "20200101",
                "20200102",
            ],
            "raw_timestamp": [
                1_577_923_260,
                1_577_880_000,
                1_577_890_000,
                1_577_923_080,
                1_577_883_600,
                1_577_923_140,
                1_577_923_320,
            ],
            "raw_icao24": ["a00002", "a00001", "b00001", "a00002", "a00001", "a00002", "a00002"],
            "meta_split": ["test", "train", "train", "test", "train", "test", "test"],
            "altitude": [120.0, 1_000.0, 2_000.0, 80.0, 1_200.0, 100.0, 150.0],
        }
    )


def _assemble(
    frame: pl.DataFrame,
    selection_plan: SelectionPlan,
) -> pl.DataFrame:
    module = _day_assemble_module()
    plan = build_day_plan(selection_plan, "20200101")
    profile = resolve_science_profile(PROFILE_ID)
    return cast(
        "pl.DataFrame",
        module.assemble_partition(frame, plan, ("A20N", "20200101"), profile, VERSIONS),
    )


def test_every_published_row_carries_full_durable_identity(
    recorded_frame: pl.DataFrame,
    selection_plan: SelectionPlan,
) -> None:
    """AC2: every published row carries complete, non-null durable identity."""
    module = _day_assemble_module()
    result = _assemble(recorded_frame, selection_plan)

    assert set(module.PARTITION_IDENTITY_COLUMNS) <= set(result.columns)
    null_counts = result.select(module.PARTITION_IDENTITY_COLUMNS).null_count()
    assert sum(null_counts.row(0)) == 0


def test_scientific_numeric_boundary_casts_nullable_decoder_strings() -> None:
    """Decoder placeholder strings cannot reach scientific arithmetic."""
    module = _day_assemble_module()
    frame = pl.DataFrame(
        {
            "raw_alt_ft": ["12000.0", None],
            "bds_mcp_alt_sel_ft": [None, None],
            "selection_id": ["sel-a", "sel-a"],
        }
    )

    result = module._coerce_scientific_numeric_columns(frame)

    assert result.schema["raw_alt_ft"] == pl.Float64
    assert result.schema["bds_mcp_alt_sel_ft"] == pl.Float64
    assert result["selection_id"].to_list() == ["sel-a", "sel-a"]


def test_midnight_trajectory_is_ordered_union_of_source_days(
    recorded_frame: pl.DataFrame,
    selection_plan: SelectionPlan,
) -> None:
    """AC3: the midnight trajectory is a complete ordered source-day union."""
    result = _assemble(recorded_frame, selection_plan)
    midnight = result.filter(pl.col("selection_id") == "sel-a20n-midnight")
    pairs = list(midnight.select("meta_source_day", "raw_timestamp").iter_rows())

    assert pairs == [
        ("20200101", 1_577_923_080),
        ("20200101", 1_577_923_140),
        ("20200102", 1_577_923_260),
        ("20200102", 1_577_923_320),
    ]
    assert midnight["raw_timestamp"].n_unique() == midnight.height


def test_bounded_and_full_history_assembly_agree_row_by_row(
    recorded_frame: pl.DataFrame,
    selection_plan: SelectionPlan,
) -> None:
    """AC4: bounded source-day inputs and full history have row-for-row parity."""
    full_result = _assemble(recorded_frame, selection_plan)
    bounded_inputs = (
        recorded_frame.filter(pl.col("meta_source_day") == source_day)
        for source_day in ("20200101", "20200102")
    )
    bounded_result = _assemble(pl.concat(bounded_inputs), selection_plan)
    common_columns = [column for column in full_result.columns if column in bounded_result.columns]

    assert_frame_equal(
        bounded_result.select(common_columns),
        full_result.select(common_columns),
        check_row_order=True,
        check_column_order=True,
    )


def test_selection_id_outside_day_plan_is_refused(
    selection_plan: SelectionPlan,
) -> None:
    """AC5: a row otherwise landing in the key cannot escape DayPlan scope."""
    frame = pl.DataFrame(
        {
            "selection_id": ["sel-e190"],
            "cohort": ["A20N"],
            "meta_source_day": ["20200101"],
            "raw_timestamp": [1_577_880_500],
            "raw_icao24": ["e00001"],
            "meta_split": ["validation"],
            "altitude": [900.0],
        }
    )

    with pytest.raises(DayScopeViolation) as excinfo:
        _assemble(frame, selection_plan)

    assert "sel-e190" in str(excinfo.value)


@pytest.fixture
def next_day_midnight_selection_plan() -> SelectionPlan:
    return SelectionPlan(
        flights=(
            SelectedFlight(
                icao24="39a20f",
                callsign="A20N2",
                firstseen=1_577_923_080,
                lastseen=1_577_923_320,
                msn="A20N-002",
                split="test",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-a20n-midnight",
                utc_days=("20200101", "20200102"),
                acquisition_key="a20n-midnight-20200101",
            ),
        ),
        digest="next-day-midnight-selection",
    )


def test_candidate_rows_union_next_day_without_moving_partition_anchor(
    next_day_midnight_selection_plan: SelectionPlan,
) -> None:
    """AC1: candidate rows form one ordered union anchored on selection day 20200101."""
    corpus = (
        pl.concat(build_raw_opensky_day(day) for day in ("20200101", "20200102"))
        .filter(pl.col("selection_id") == "sel-a20n-midnight")
        .select(
            "selection_id",
            "cohort",
            "meta_selection_day",
            "meta_source_day",
            "raw_timestamp",
            "raw_icao24",
        )
        .with_columns(pl.lit("test").alias("meta_split"))
    )
    duplicate = corpus.filter(
        (pl.col("selection_id") == "sel-a20n-midnight") & (pl.col("meta_source_day") == "20200102")
    ).head(1)
    recorded = pl.concat((corpus, duplicate))
    result = _assemble(recorded, next_day_midnight_selection_plan).filter(
        pl.col("selection_id") == "sel-a20n-midnight"
    )
    expected = (
        corpus.filter(pl.col("selection_id") == "sel-a20n-midnight")
        .unique(subset=["selection_id", "raw_timestamp"], keep="first")
        .sort("raw_timestamp")
    )
    actual_pairs = list(result.select("meta_source_day", "raw_timestamp").iter_rows())
    expected_pairs = list(expected.select("meta_source_day", "raw_timestamp").iter_rows())
    timestamps = result["raw_timestamp"].to_list()

    assert actual_pairs == expected_pairs
    assert timestamps == sorted(set(timestamps))
    assert result["meta_selection_day"].unique().to_list() == ["20200101"]
    assert build_day_plan(next_day_midnight_selection_plan, "20200102").partition_keys == ()
