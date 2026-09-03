"""Integration coverage for projecting compiled selections into fleet plans."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from node_fdm_pipeline.commands import _fleet_plan
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
    triples: list[tuple[str, Path, Path]] = []
    rows: list[dict[str, object]] = []
    for name in ("C1", "C2"):
        cohort_dir = tmp_path / "types" / name
        selection_path = cohort_dir / "results" / f"selection_{name}.csv"
        selection_path.parent.mkdir(parents=True)
        config_path = cohort_dir / "config.yaml"
        make_config(config_path, tmp_path / "data" / name)
        pl.DataFrame({"icao24": ["b00003"], "day": ["2020-01-05"]}).write_csv(selection_path)
        rows.append(
            {
                "selection_id": "sel-alpha",
                "icao24": "a00001",
                "callsign": "ALPHA1",
                "firstseen": 1577835000,
                "lastseen": 1577838600,
                "msn": "M1",
                "split": "train",
                "cohort": name,
                "day": "2020-01-01",
            }
        )
        triples.append((name, config_path, selection_path))
    return _fleet_plan.discover_cohorts(tmp_path), compile_selection(rows)


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


@pytest.mark.integration
def test_selection_overrides_per_cohort_files(
    alpha_selection_inputs: tuple[list[tuple[str, Path, Path]], SelectionPlan],
) -> None:
    """AC1: compiled identities and UTC days override stale per-cohort files."""
    triples, selection = alpha_selection_inputs

    plan = build_fleet_plan(triples, selection=selection)
    cohorts = {cohort.name: cohort for cohort in plan.cohorts}

    assert cohorts["C1"].icao24 == frozenset({"a00001"})
    assert cohorts["C2"].icao24 == frozenset({"a00001"})
    assert plan.dates == {"20191231": ["a00001"], "20200101": ["a00001"]}
    assert "b00003" not in plan.owner
    assert "20200105" not in plan.dates
    assert {entry.utc_day for entry in plan.shared_acquisitions} == {
        "20191231",
        "20200101",
    }


@pytest.mark.integration
def test_single_cohort_selection_does_not_leak_to_other_cohort(
    alpha_selection_inputs: tuple[list[tuple[str, Path, Path]], SelectionPlan],
) -> None:
    """AC3: a C1-only flight contributes neither identity nor UTC day to C2."""
    triples, _ = alpha_selection_inputs
    rows = [
        {
            "selection_id": "sel-alpha",
            "icao24": "a00001",
            "callsign": "ALPHA1",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
            "msn": "M1",
            "split": "train",
            "cohort": cohort,
            "day": "2020-01-01",
        }
        for cohort in ("C1", "C2")
    ]
    rows.append(
        {
            "selection_id": "sel-beta",
            "icao24": "b00004",
            "callsign": "BETA1",
            "firstseen": 1577925000,
            "lastseen": 1577925000,
            "msn": "M2",
            "split": "test",
            "cohort": "C1",
            "day": "2020-01-02",
        }
    )

    plan = build_fleet_plan(triples, selection=compile_selection(rows))
    cohorts = {cohort.name: cohort for cohort in plan.cohorts}
    beta_acquisitions = tuple(
        entry for entry in plan.shared_acquisitions if entry.utc_day == "20200102"
    )

    assert cohorts["C1"].icao24 == frozenset({"a00001", "b00004"})
    assert cohorts["C2"].icao24 == frozenset({"a00001"})
    assert beta_acquisitions
    assert {entry.owners for entry in beta_acquisitions} == {("C1",)}


@pytest.mark.integration
def test_fleet_plan_exposes_shared_acquisitions(
    alpha_selection_inputs: tuple[list[tuple[str, Path, Path]], SelectionPlan],
) -> None:
    """AC2: FleetPlan exposes the pure shared-acquisition projection unchanged."""
    triples, selection = alpha_selection_inputs

    plan = build_fleet_plan(triples, selection=selection)

    assert plan.shared_acquisitions == _fleet_plan.plan_shared_acquisitions(selection)
