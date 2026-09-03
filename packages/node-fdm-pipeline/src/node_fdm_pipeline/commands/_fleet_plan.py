"""Cross-cohort download plan: mutualise aircraft per date, parallelise dates.

``fdm download --flight-plan`` fetches one cohort at a time, and within a day it
asks for that cohort's aircraft only — measured at ~1.2 aircraft per request.
Because all 22 cohorts span the same 2019-2025 window, the same calendar day is
then visited once per cohort: 35,000 requests for 2,545 distinct days.

This module builds the other plan. For a given date it names *every* aircraft
any cohort wants that day (~38 on average), so the day is fetched once, and the
rows are routed back to whichever silo asked for each aircraft. Dates are
independent units of work, so a small pool of workers runs several concurrently;
the pool size is deliberately low because the OpenSky Trino cluster caps how many
queries one account may run at once.

Routing is a total function ``icao24 -> cohort``, which is what makes the
dispatch a plain lookup rather than a fan-out: the 22 selections are **strictly
disjoint in icao24** (verified: zero aircraft appears in two cohorts, over all
97,658 aircraft-days). :func:`build_fleet_plan` asserts this rather than assuming
it, because if it ever stopped holding the writes would silently land in one
silo and leave a hole in another.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan

if TYPE_CHECKING:
    import polars as pl

    from node_fdm_pipeline.config import PipelineConfig

__all__ = [
    "Cohort",
    "FleetPlan",
    "SharedAcquisition",
    "build_fleet_plan",
    "discover_cohorts",
    "plan_shared_acquisitions",
]


@dataclass(frozen=True)
class Cohort:
    """One silo: its name, its config, and the aircraft it owns."""

    name: str
    config_path: Path
    selection_path: Path
    cfg: PipelineConfig
    icao24: frozenset[str]
    utc_days: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FleetPlan:
    """A mutualised download plan.

    Attributes:
        dates: ``{YYYYMMDD: [icao24, ...]}`` — every aircraft wanted that day,
            across all cohorts, sorted for reproducible request ordering.
        owner: ``{icao24: (Cohort, ...)}`` — every silo receiving the aircraft's rows.
        cohorts: The cohorts this plan covers.
        selection: The compiled selection projected into this plan, when available.
        shared_acquisitions: Date/kind staging units and their owning cohorts.
    """

    dates: dict[str, list[str]]
    owner: dict[str, tuple[Cohort, ...]]
    cohorts: tuple[Cohort, ...]
    selection: SelectionPlan | None = None
    shared_acquisitions: tuple[SharedAcquisition, ...] = ()

    def resolve(self, icao24: str, day: str) -> SelectedFlight | None:
        """Return the selected flight that requested an aircraft-day pair."""
        if self.selection is None:
            return None
        return next(
            (
                flight
                for flight in self.selection.flights
                if flight.icao24 == icao24 and day in flight.utc_days
            ),
            None,
        )

    @property
    def aircraft_days(self) -> int:
        """Total (date, icao24) pairs — the invariant volume, unchanged by batching."""
        return sum(len(v) for v in self.dates.values())

    def requests_saved(self) -> tuple[int, int]:
        """``(per_cohort_requests, mutualised_requests)`` for the same payload."""
        per_cohort = sum(
            len({cohort.name for aircraft in date_ac for cohort in self.owner[aircraft]})
            for date_ac in self.dates.values()
        )
        return per_cohort, len(self.dates)


def _parse_plan_day(day: pl.Expr) -> pl.Expr:
    """Parse a plan's date column, read as text, into a datetime.

    ``infer_schema=False`` reads every column as text, so the date arrives as a
    string and a bare ``cast(pl.Datetime)`` does not parse it: on a plain ISO
    date — ``2025-02-17``, the format this module's own docstrings advertise —
    the cast failed on every row.

    Only the calendar day is ever used (the result is immediately formatted as
    ``%Y%m%d``), so the time part is noise to be discarded rather than data to
    be parsed. Taking the first 10 characters and reading them as a date makes
    the parse independent of what follows: a bare ``2025-02-17``, a
    space-separated ``2025-02-17 08:31:00`` and an ISO-8601
    ``2025-02-17T09:02:11Z`` all yield the same day. Inference cannot do this --
    it picks one format from the leading rows and then fails on any row that
    does not match it, which a real ``firstseen`` column routinely contains.
    """
    return day.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=True)


def _read_selection_days(path: Path) -> dict[str, set[str]]:
    """Read one selection CSV into ``{YYYYMMDD: {icao24}}``.

    Mirrors ``_read_flight_plan``'s reading rules on purpose: ``infer_schema=False``
    because a selection carries serial-number columns that are numeric for most
    rows and text for airframes whose register publishes none, and only the date
    column is ever cast.
    """
    import polars as pl

    frame = pl.read_csv(path, infer_schema=False)
    if "day" in frame.columns:
        day = pl.col("day")
    elif "firstseen" in frame.columns:
        day = pl.col("firstseen")
    else:
        raise SystemExit(f"{path}: needs a 'day' or 'firstseen' column to build a plan")

    grouped = (
        frame.with_columns(_parse_plan_day(day).dt.strftime("%Y%m%d").alias("_day"))
        .group_by("_day")
        .agg(pl.col("icao24").unique().alias("_icao24"))
    )
    return {row["_day"]: set(row["_icao24"]) for row in grouped.iter_rows(named=True)}


def discover_cohorts(fleet_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find ``(name, config, selection)`` triples under a fleet directory.

    The pairing rule is the one the fleet's own tooling uses: ``config.yaml`` goes
    with ``selection_{type}.csv``, and ``config.{variant}.yaml`` with
    ``selection_{type}__{variant}.csv``. A config whose selection is missing is
    an error rather than a skip — a silently absent cohort is how a fleet-wide
    download quietly covers 21 of 22.
    """
    out: list[tuple[str, Path, Path]] = []
    for config_path in sorted(fleet_dir.glob("types/*/config*.yaml")):
        designation = config_path.parent.name
        stem = config_path.name.removeprefix("config").removesuffix(".yaml")
        variant = stem.lstrip(".") or None
        name = f"{designation}__{variant}" if variant else designation
        selection = config_path.parent / "results" / f"selection_{name}.csv"
        if not selection.exists():
            raise SystemExit(f"{config_path}: no selection at {selection}")
        out.append((name, config_path, selection))
    return out


@dataclass(frozen=True)
class SharedAcquisition:
    """One mutualised staging unit shared by every owning cohort."""

    utc_day: str
    kind: Literal["history", "extended", "flightlist"]
    owners: tuple[str, ...]


def plan_shared_acquisitions(selection: SelectionPlan) -> tuple[SharedAcquisition, ...]:
    """Project a selection into unique, deterministically ordered staging units."""
    owners_by_day: dict[str, set[str]] = defaultdict(set)
    for flight in selection.flights:
        for utc_day in flight.utc_days:
            owners_by_day[utc_day].update(flight.cohorts)

    kinds: tuple[Literal["history", "extended", "flightlist"], ...] = (
        "history",
        "extended",
        "flightlist",
    )
    return tuple(
        SharedAcquisition(utc_day=utc_day, kind=kind, owners=tuple(sorted(owners)))
        for utc_day, owners in sorted(owners_by_day.items())
        for kind in kinds
    )


def build_fleet_plan(
    triples: list[tuple[str, Path, Path]],
    *,
    data_root: Path | None = None,
    selection: SelectionPlan | None = None,
) -> FleetPlan:
    """Invert per-cohort selections into one date-major, mutualised plan.

    Shared aircraft retain every owning cohort in declaration order so one
    mutualised acquisition can fan out into every relevant silo.
    """
    from node_fdm_pipeline.config import PipelineConfig

    cohorts: list[Cohort] = []
    owner: dict[str, list[Cohort]] = {}
    dates: dict[str, set[str]] = defaultdict(set)

    for name, config_path, selection_path in triples:
        selected_flights: tuple[SelectedFlight, ...] = ()
        if selection is None:
            per_day = _read_selection_days(selection_path)
            icao = {aircraft for day_set in per_day.values() for aircraft in day_set}
            utc_days = frozenset(per_day)
        else:
            per_day = {}
            selected_flights = tuple(
                flight for flight in selection.flights if name in flight.cohorts
            )
            icao = {flight.icao24 for flight in selected_flights}
            utc_days = frozenset(day for flight in selected_flights for day in flight.utc_days)
        cohort = Cohort(
            name=name,
            config_path=config_path,
            selection_path=selection_path,
            cfg=PipelineConfig.from_yaml(config_path, data_root=data_root),
            icao24=frozenset(icao),
            utc_days=utc_days,
        )
        cohorts.append(cohort)

        for aircraft in icao:
            owner.setdefault(aircraft, []).append(cohort)

        if selection is None:
            for day, day_set in per_day.items():
                dates[day] |= day_set
        else:
            for flight in selected_flights:
                for day in flight.utc_days:
                    dates[day].add(flight.icao24)

    return FleetPlan(
        dates={day: sorted(v) for day, v in sorted(dates.items())},
        owner={aircraft: tuple(owners) for aircraft, owners in owner.items()},
        cohorts=tuple(cohorts),
        selection=selection,
        shared_acquisitions=(plan_shared_acquisitions(selection) if selection is not None else ()),
    )
