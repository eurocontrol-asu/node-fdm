"""Integration coverage for projecting compiled selections into fleet plans."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from node_fdm_pipeline.commands import _fleet_plan
from node_fdm_pipeline.commands._fleet_plan import FleetPlan, build_fleet_plan
from node_fdm_pipeline.commands._fleet_selection import (
    SelectedFlight,
    SelectionFormatError,
    SelectionPlan,
    compile_selection,
    load_selection_file,
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


def _campaign_rows() -> list[dict[str, object]]:
    return [
        {
            "selection_id": "sel-alpha",
            "icao24": "a00001",
            "callsign": "ALPHA1",
            "firstseen": 1577836800,
            "lastseen": 1577840400,
            "msn": "M1",
            "split": "train",
            "cohort": "C1",
            "utc_days": ["20200110"],
        },
        {
            "selection_id": "sel-zulu",
            "icao24": "b00002",
            "callsign": "ZULU1",
            "firstseen": 1577923200,
            "lastseen": 1577926800,
            "msn": "M2",
            "split": "test",
            "cohort": "C2",
            "utc_days": ["20200111", "20200112"],
        },
    ]


def _write_selection_csv(path: Path, rows: list[dict[str, object]]) -> None:
    csv_rows: list[dict[str, object]] = []
    for row in rows:
        csv_row = dict(row)
        csv_row["cohorts"] = csv_row.pop("cohort")
        utc_days = csv_row["utc_days"]
        assert isinstance(utc_days, list)
        csv_row["utc_days"] = "|".join(str(day) for day in utc_days)
        csv_rows.append(csv_row)
    pl.DataFrame(csv_rows).write_csv(path)


def _write_selection_json(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(rows))


def _campaign_triples(
    tmp_path: Path,
    make_config: Callable[..., Path],
    selection_path: Path,
) -> list[tuple[str, Path, Path]]:
    triples: list[tuple[str, Path, Path]] = []
    for index, name in enumerate(("C1", "C2"), start=1):
        cohort_dir = tmp_path / "types" / name
        historical_path = cohort_dir / "results" / f"selection_{name}.csv"
        historical_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "icao24": [f"dead0{index}"],
                "day": [f"2020-02-0{index}"],
            }
        ).write_csv(historical_path)
        config_path = cohort_dir / "config.yaml"
        make_config(config_path, tmp_path / "data" / name)
        triples.append((name, config_path, selection_path))
    return triples


def _plan_projection(
    plan: FleetPlan,
) -> tuple[
    dict[str, list[str]],
    dict[str, tuple[frozenset[str], frozenset[str]]],
]:
    return (
        plan.dates,
        {cohort.name: (cohort.icao24, cohort.utc_days) for cohort in plan.cohorts},
    )


@pytest.mark.integration
def test_csv_selection_overrides_contradictory_cohort_files(
    tmp_path: Path,
    make_config: Callable[..., Path],
) -> None:
    """AC1: the header CSV alone determines dates and aircraft."""
    selection_path = tmp_path / "selection.csv"
    rows = _campaign_rows()
    _write_selection_csv(selection_path, rows)
    triples = _campaign_triples(tmp_path, make_config, selection_path)

    loaded = load_selection_file(selection_path)
    plan = build_fleet_plan(triples)

    assert loaded.plan is not None
    assert plan.dates == {
        "20200110": ["a00001"],
        "20200111": ["b00002"],
        "20200112": ["b00002"],
    }
    assert {cohort.name: (cohort.icao24, cohort.utc_days) for cohort in plan.cohorts} == {
        "C1": (frozenset({"a00001"}), frozenset({"20200110"})),
        "C2": (
            frozenset({"b00002"}),
            frozenset({"20200111", "20200112"}),
        ),
    }
    assert not {"dead01", "dead02"} & set(plan.owner)


@pytest.mark.integration
def test_equivalent_csv_and_json_selections_build_equal_plans(
    tmp_path: Path,
    make_config: Callable[..., Path],
) -> None:
    """AC2: equivalent CSV and JSON selections yield equal plan projections."""
    rows = _campaign_rows()
    csv_path = tmp_path / "selection.csv"
    json_path = tmp_path / "selection.json"
    _write_selection_csv(csv_path, rows)
    _write_selection_json(json_path, rows)

    csv_plan = build_fleet_plan(_campaign_triples(tmp_path, make_config, csv_path))
    json_plan = build_fleet_plan(_campaign_triples(tmp_path, make_config, json_path))

    assert _plan_projection(csv_plan) == _plan_projection(json_plan)


@pytest.mark.integration
def test_editing_only_csv_utc_days_changes_plan_dates(
    tmp_path: Path,
    make_config: Callable[..., Path],
) -> None:
    """AC3: changing only utc_days changes the fleet plan dates."""
    selection_path = tmp_path / "selection.csv"
    rows = _campaign_rows()
    _write_selection_csv(selection_path, rows)
    triples = _campaign_triples(tmp_path, make_config, selection_path)
    initial_plan = build_fleet_plan(triples)

    edited_rows = [dict(row) for row in rows]
    edited_rows[0]["utc_days"] = ["20200301", "20200302"]
    edited_rows[1]["utc_days"] = ["20200303"]
    _write_selection_csv(selection_path, edited_rows)
    edited_plan = build_fleet_plan(triples)

    assert initial_plan.dates != edited_plan.dates
    assert edited_plan.dates == {
        "20200301": ["a00001"],
        "20200302": ["a00001"],
        "20200303": ["b00002"],
    }


@pytest.mark.integration
def test_csv_without_selection_id_fails_without_cohort_fallback(
    tmp_path: Path,
    make_config: Callable[..., Path],
) -> None:
    """AC4: a malformed CSV names selection_id instead of using cohort files."""
    selection_path = tmp_path / "selection.csv"
    malformed_rows = [dict(row) for row in _campaign_rows()]
    for row in malformed_rows:
        row.pop("selection_id")
    _write_selection_csv(selection_path, malformed_rows)
    triples = _campaign_triples(tmp_path, make_config, selection_path)

    with pytest.raises(SelectionFormatError) as excinfo:
        build_fleet_plan(triples)

    assert "selection_id" in str(excinfo.value)
