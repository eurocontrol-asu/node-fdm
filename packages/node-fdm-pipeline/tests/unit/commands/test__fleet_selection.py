"""Unit tests for deterministic in-memory fleet-selection compilation."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import pytest


def _load_fleet_selection() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_selection")


@pytest.fixture
def alpha_beta_selection() -> list[Mapping[str, Any]]:
    """Three source rows: two cohort assignments for alpha and one for beta."""
    alpha = {
        "icao24": "a00001",
        "callsign": "ALPHA1",
        "firstseen": 1577835000,
        "lastseen": 1577838600,
        "msn": "M1",
        "split": "train",
    }
    beta = {
        "icao24": "b00002",
        "callsign": "BETA2",
        "firstseen": 1577923200,
        "lastseen": 1577926800,
        "msn": "M2",
        "split": "test",
    }
    return [
        {**alpha, "cohort": "C1", "day": "2019-12-31"},
        {**alpha, "cohort": "C2", "day": "2020-01-01"},
        {**beta, "cohort": "C3", "day": "2020-01-02"},
    ]


def test_compile_selection_unites_cohorts_by_flight_identity(
    alpha_beta_selection: list[Mapping[str, Any]],
) -> None:
    """AC1: duplicate identities become one flight without losing cohort assignments."""
    fleet_selection = _load_fleet_selection()
    plan = fleet_selection.compile_selection(alpha_beta_selection)

    assert len(plan.flights) == 2
    alpha, beta = plan.flights
    assert alpha.cohorts == frozenset({"C1", "C2"})
    assert beta.cohorts == frozenset({"C3"})


def test_compile_selection_preserves_every_identity_field(
    alpha_beta_selection: list[Mapping[str, Any]],
) -> None:
    """AC2: compiled flights preserve every acquisition identity field."""
    fleet_selection = _load_fleet_selection()
    plan = fleet_selection.compile_selection(alpha_beta_selection)

    assert all(isinstance(flight, fleet_selection.SelectedFlight) for flight in plan.flights)
    assert [
        (
            flight.icao24,
            flight.callsign,
            flight.firstseen,
            flight.lastseen,
            flight.msn,
            flight.split,
        )
        for flight in plan.flights
    ] == [
        ("a00001", "ALPHA1", 1577835000, 1577838600, "M1", "train"),
        ("b00002", "BETA2", 1577923200, 1577926800, "M2", "test"),
    ]


def test_compile_selection_mutualises_acquisition_key(
    alpha_beta_selection: list[Mapping[str, Any]],
) -> None:
    """AC3: cohort fan-out does not duplicate the acquisition identity."""
    fleet_selection = _load_fleet_selection()
    plan = fleet_selection.compile_selection(alpha_beta_selection)

    assert len({flight.acquisition_key for flight in plan.flights}) == 2
    alpha_flights = [flight for flight in plan.flights if flight.icao24 == "a00001"]
    assert len(alpha_flights) == 1
    assert alpha_flights[0].cohorts == frozenset({"C1", "C2"})


def test_compile_selection_derives_all_closed_interval_utc_days(
    alpha_beta_selection: list[Mapping[str, Any]],
) -> None:
    """AC4: UTC acquisition dependencies cover both endpoints of each interval."""
    fleet_selection = _load_fleet_selection()
    plan = fleet_selection.compile_selection(alpha_beta_selection)
    alpha, beta = plan.flights

    assert alpha.utc_days == ("2019-12-31", "2020-01-01")
    assert beta.utc_days == ("2020-01-02",)
    assert fleet_selection.utc_days_for_interval(1577835000, 1577838600) == (
        "2019-12-31",
        "2020-01-01",
    )


def test_compile_selection_is_order_independent(
    alpha_beta_selection: list[Mapping[str, Any]],
) -> None:
    """AC5: canonical flights, UTC days and digest are independent of input order."""
    fleet_selection = _load_fleet_selection()
    plan_a = fleet_selection.compile_selection(alpha_beta_selection)
    plan_b = fleet_selection.compile_selection(list(reversed(alpha_beta_selection)))

    assert plan_a.flights == plan_b.flights
    assert tuple(flight.utc_days for flight in plan_a.flights) == tuple(
        flight.utc_days for flight in plan_b.flights
    )
    assert plan_a.digest == plan_b.digest
