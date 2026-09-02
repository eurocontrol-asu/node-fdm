"""Integration coverage for projecting compiled selections into fleet plans."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from node_fdm_pipeline.commands._fleet_plan import FleetPlan, build_fleet_plan
from node_fdm_pipeline.commands._fleet_selection import (
    SelectedFlight,
    SelectionPlan,
    compile_selection,
)


@pytest.fixture
def alpha_selection_inputs(
    tmp_path: Path, make_config: Callable[..., Path]
) -> tuple[list[tuple[str, Path, Path]], SelectionPlan]:
    """Build one real cohort selection whose flight crosses a UTC day boundary."""
    config_path = tmp_path / "config.yaml"
    selection_path = tmp_path / "selection_C1.csv"
    make_config(config_path, tmp_path / "data" / "C1")
    row = {
        "selection_id": "sel-alpha",
        "icao24": "a00001",
        "callsign": "ALPHA1",
        "firstseen": 1577835000,
        "lastseen": 1577838600,
        "msn": "M1",
        "split": "train",
        "cohort": "C1",
        "day": "2020-01-01",
    }
    pl.DataFrame([row]).write_csv(selection_path)
    return [("C1", config_path, selection_path)], compile_selection([row])


@pytest.mark.integration
def test_plan_dates_come_from_selection_identity_utc_days(
    alpha_selection_inputs: tuple[list[tuple[str, Path, Path]], SelectionPlan],
) -> None:
    """AC1: fleet dates are projected from each selection identity's utc_days."""
    triples, selection = alpha_selection_inputs

    plan = build_fleet_plan(triples, selection=selection)

    assert plan.dates == {
        "20191231": ["a00001"],
        "20200101": ["a00001"],
    }


@pytest.mark.integration
def test_plan_resolves_selected_flight_by_aircraft_and_day(
    alpha_selection_inputs: tuple[list[tuple[str, Path, Path]], SelectionPlan],
) -> None:
    """AC2: an aircraft-day address resolves to the unchanged SelectedFlight."""
    triples, selection = alpha_selection_inputs
    selected = selection.flights[0]

    plan: FleetPlan = build_fleet_plan(triples, selection=selection)
    resolved = plan.resolve("a00001", "20191231")

    assert isinstance(resolved, SelectedFlight)
    assert resolved is selected
    assert (
        resolved.selection_id,
        resolved.acquisition_key,
        resolved.msn,
        resolved.split,
        resolved.firstseen,
        resolved.lastseen,
    ) == (
        "sel-alpha",
        selected.acquisition_key,
        selected.msn,
        selected.split,
        selected.firstseen,
        selected.lastseen,
    )
