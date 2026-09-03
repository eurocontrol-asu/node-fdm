from __future__ import annotations

from datetime import UTC, datetime
from importlib import import_module
from itertools import pairwise
from types import ModuleType

import polars as pl


def _daily_corpus() -> ModuleType:
    return import_module("_daily_corpus")


def test_20200101_frame_is_prederive_and_holds_expected_selections_and_cohorts() -> None:
    """AC1: the first source day is pre-derive and spans its expected cohorts."""
    corpus = _daily_corpus()
    frame: pl.DataFrame = corpus.build_raw_opensky_day("20200101")

    selection_ids: set[str] = {str(value) for value in frame.get_column("selection_id").to_list()}
    aircraft_types: set[str] = {
        str(value) for value in frame.get_column("meta_aircraft_type").to_list()
    }
    cohorts: set[str] = {str(value) for value in frame.get_column("cohort").to_list()}

    assert not any(name.startswith("fdm_") for name in frame.columns)
    assert selection_ids == {"sel-a20n", "sel-a20n-midnight", "sel-b738"}
    assert aircraft_types == cohorts == {"A20N", "B738"}


def test_20200102_own_day_rows_hold_expected_selections_and_typecodes() -> None:
    """AC2: filtering by selection day isolates the second day's three cohorts."""
    corpus = _daily_corpus()
    frame: pl.DataFrame = corpus.build_raw_opensky_day("20200102")
    own_day: pl.DataFrame = frame.filter(pl.col("meta_selection_day") == "20200102")

    selection_ids: set[str] = {
        str(value) for value in own_day.get_column("selection_id").to_list()
    }
    aircraft_types: set[str] = {
        str(value) for value in own_day.get_column("meta_aircraft_type").to_list()
    }

    assert selection_ids == {"sel-e190", "sel-crj9", "sel-cl35"}
    assert aircraft_types == {"E190", "CRJ9", "CL35"}


def test_midnight_trajectory_spans_both_source_days_in_order() -> None:
    """AC3: one selection crosses midnight while retaining its selection day."""
    corpus = _daily_corpus()
    first_frame: pl.DataFrame = corpus.build_raw_opensky_day("20200101")
    second_frame: pl.DataFrame = corpus.build_raw_opensky_day("20200102")
    first_slice: pl.DataFrame = first_frame.filter(pl.col("selection_id") == "sel-a20n-midnight")
    second_slice: pl.DataFrame = second_frame.filter(pl.col("selection_id") == "sel-a20n-midnight")

    assert first_slice.height > 0
    assert second_slice.height > 0

    combined: pl.DataFrame = pl.concat([first_slice, second_slice], how="vertical")
    timestamp_values: list[object] = combined.get_column("raw_timestamp").to_list()
    timestamps: list[datetime] = [
        value for value in timestamp_values if isinstance(value, datetime)
    ]
    midnight = datetime(2020, 1, 2, tzinfo=UTC)
    first_source_days: set[str] = {
        str(value) for value in first_slice.get_column("meta_source_day").to_list()
    }
    second_source_days: set[str] = {
        str(value) for value in second_slice.get_column("meta_source_day").to_list()
    }
    selection_days: set[str] = {
        str(value) for value in combined.get_column("meta_selection_day").to_list()
    }

    assert len(timestamps) == combined.height
    assert all(later > earlier for earlier, later in pairwise(timestamps))
    assert timestamps[0] < midnight < timestamps[-1]
    assert first_source_days == {"20200101"}
    assert second_source_days == {"20200102"}
    assert selection_days == {"20200101"}
