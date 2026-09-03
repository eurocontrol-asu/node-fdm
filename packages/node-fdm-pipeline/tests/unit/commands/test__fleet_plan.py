"""Unit tests for the download-plan date parsing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest


def _write_plan(tmp_path: Path, column: str, values: list[str]) -> Path:
    path = tmp_path / "plan.csv"
    pl.DataFrame({"icao24": [f"a{i:05x}" for i in range(len(values))], column: values}).write_csv(
        path
    )
    return path


@pytest.mark.parametrize(
    "value",
    [
        "2025-02-17",
        "2025-02-17 08:31:00",
        "2025-02-17T09:02:11Z",
        "2025-02-17T09:02:11.123456+00:00",
    ],
)
def test_plan_day_accepts_every_timestamp_shape(tmp_path: Path, value: str) -> None:
    """A day, a space-separated timestamp and an ISO-8601 one all give one day."""
    from node_fdm_pipeline.commands.data import _read_flight_plan

    plan = _read_flight_plan(_write_plan(tmp_path, "day", [value]))

    assert list(plan) == ["20250217"]


def test_plan_day_tolerates_mixed_shapes_in_one_column(tmp_path: Path) -> None:
    """Mixed shapes group under the same day.

    Format inference reads the leading rows, picks one format and then fails on
    any row that does not match it — which a real ``firstseen`` column contains
    routinely. Grouping must depend on the calendar day alone.
    """
    from node_fdm_pipeline.commands.data import _read_flight_plan

    plan = _read_flight_plan(
        _write_plan(
            tmp_path,
            "firstseen",
            ["2025-02-17", "2025-02-17 08:31:00", "2025-02-17T09:02:11Z"],
        )
    )

    assert list(plan) == ["20250217"]
    assert len(plan["20250217"]) == 3


def test_plan_groups_aircraft_by_day(tmp_path: Path) -> None:
    """Each day carries exactly the aircraft listed against it."""
    from node_fdm_pipeline.commands.data import _read_flight_plan

    path = tmp_path / "plan.csv"
    pl.DataFrame(
        {
            "icao24": ["aaa111", "bbb222", "aaa111"],
            "day": ["2025-02-17", "2025-02-17", "2025-05-21"],
        }
    ).write_csv(path)

    plan = _read_flight_plan(path)

    assert sorted(plan) == ["20250217", "20250521"]
    assert sorted(plan["20250217"]) == ["aaa111", "bbb222"]
    assert plan["20250521"] == ["aaa111"]


def test_build_fleet_plan_keeps_every_cross_cohort_owner(
    tmp_path: Path, make_config: Callable[..., Path]
) -> None:
    """One Mode-S identity cannot silently dispatch into two model silos."""
    from node_fdm_pipeline.commands._fleet_plan import build_fleet_plan

    triples: list[tuple[str, Path, Path]] = []
    for name in ("A320neo", "A321neo"):
        config = make_config(tmp_path / f"{name}.yaml", tmp_path / name)
        selection = tmp_path / f"selection_{name}.csv"
        pl.DataFrame({"icao24": ["424461"], "day": ["2020-01-01"]}).write_csv(selection)
        triples.append((name, config, selection))

    plan = build_fleet_plan(triples)

    assert isinstance(plan.owner["424461"], tuple)
    assert tuple(cohort.name for cohort in plan.owner["424461"]) == ("A320neo", "A321neo")


def test_plan_without_a_date_column_is_rejected(tmp_path: Path) -> None:
    """A plan needs a 'day' or a 'firstseen' column."""
    from node_fdm_pipeline.commands.data import _read_flight_plan

    path = tmp_path / "plan.csv"
    pl.DataFrame({"icao24": ["aaa111"], "when": ["2025-02-17"]}).write_csv(path)

    with pytest.raises(SystemExit):
        _read_flight_plan(path)


def test_plan_shared_acquisitions_deduplicates_day_kind_pairs() -> None:
    """AC2: each UTC-day/kind pair is emitted once with both sorted owners."""
    from typing import Literal

    from node_fdm_pipeline.commands import _fleet_plan
    from node_fdm_pipeline.commands._fleet_selection import compile_selection

    rows = [
        {
            "selection_id": selection_id,
            "icao24": icao24,
            "callsign": callsign,
            "firstseen": 1577835000,
            "lastseen": 1577838600,
            "msn": msn,
            "split": "train",
            "cohort": cohort,
            "day": "2020-01-01",
        }
        for selection_id, icao24, callsign, msn in (
            ("sel-alpha", "a00001", "ALPHA1", "M1"),
            ("sel-zulu", "z00002", "ZULU1", "M2"),
        )
        for cohort in ("C2", "C1")
    ]
    selection = compile_selection(rows)

    acquisitions = _fleet_plan.plan_shared_acquisitions(selection)
    kinds: tuple[Literal["history", "extended", "flightlist"], ...] = (
        "history",
        "extended",
        "flightlist",
    )

    assert acquisitions == tuple(
        _fleet_plan.SharedAcquisition(
            utc_day=utc_day,
            kind=kind,
            owners=("C1", "C2"),
        )
        for utc_day in ("20191231", "20200101")
        for kind in kinds
    )
