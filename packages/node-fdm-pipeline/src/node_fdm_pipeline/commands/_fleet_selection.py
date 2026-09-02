"""Deterministic compilation of in-memory fleet-selection rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel, ConfigDict

__all__ = [
    "SelectedFlight",
    "SelectionPlan",
    "compile_selection",
    "utc_days_for_interval",
]

type FlightIdentity = tuple[str, str, int, int, str, str]


class _SelectionRow(BaseModel):
    """Validated shape consumed from one source selection row."""

    icao24: str
    callsign: str
    firstseen: int
    lastseen: int
    msn: str
    split: str
    cohort: str


class SelectedFlight(BaseModel):
    """One acquisition identity and all cohorts depending on it."""

    model_config = ConfigDict(frozen=True)

    icao24: str
    callsign: str
    firstseen: int
    lastseen: int
    msn: str
    split: str
    cohorts: frozenset[str]
    utc_days: tuple[str, ...]
    acquisition_key: str


class SelectionPlan(BaseModel):
    """Canonical selection result, stable across source-row orderings."""

    model_config = ConfigDict(frozen=True)

    flights: tuple[SelectedFlight, ...]
    digest: str


def utc_days_for_interval(firstseen: int, lastseen: int) -> tuple[str, ...]:
    """Return every UTC calendar day touched by the closed timestamp interval."""
    if lastseen < firstseen:
        raise ValueError("lastseen must be greater than or equal to firstseen")

    first_day = datetime.fromtimestamp(firstseen, tz=UTC).date()
    last_day = datetime.fromtimestamp(lastseen, tz=UTC).date()
    return tuple(_iso_days(first_day, last_day))


def _iso_days(first_day: date, last_day: date) -> Iterable[str]:
    day = first_day
    while day <= last_day:
        yield day.isoformat()
        day += timedelta(days=1)


def _identity(row: _SelectionRow | SelectedFlight) -> FlightIdentity:
    return (
        row.icao24,
        row.callsign,
        row.firstseen,
        row.lastseen,
        row.msn,
        row.split,
    )


def _identity_payload(identity: FlightIdentity) -> dict[str, str | int]:
    icao24, callsign, firstseen, lastseen, msn, split = identity
    return {
        "icao24": icao24,
        "callsign": callsign,
        "firstseen": firstseen,
        "lastseen": lastseen,
        "msn": msn,
        "split": split,
    }


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _acquisition_key(identity: FlightIdentity) -> str:
    canonical_identity = _canonical_json(_identity_payload(identity))
    return hashlib.sha256(canonical_identity.encode()).hexdigest()


def _cohorts_by_identity(
    rows: Iterable[Mapping[str, object]],
) -> dict[FlightIdentity, set[str]]:
    cohorts_by_identity: dict[FlightIdentity, set[str]] = {}
    for raw_row in rows:
        row = _SelectionRow.model_validate(raw_row)
        identity = _identity(row)
        cohorts_by_identity.setdefault(identity, set()).add(row.cohort)
    return cohorts_by_identity


def _selection_flights(
    cohorts_by_identity: dict[FlightIdentity, set[str]],
) -> tuple[SelectedFlight, ...]:
    return tuple(
        _selected_flight(identity, cohorts)
        for identity, cohorts in sorted(cohorts_by_identity.items())
    )


def _selection_digest(flights: tuple[SelectedFlight, ...]) -> str:
    payload = _canonical_json(_digest_payload(flights)).encode()
    return hashlib.sha256(payload).hexdigest()


def _selected_flight(identity: FlightIdentity, cohorts: set[str]) -> SelectedFlight:
    icao24, callsign, firstseen, lastseen, msn, split = identity
    return SelectedFlight(
        icao24=icao24,
        callsign=callsign,
        firstseen=firstseen,
        lastseen=lastseen,
        msn=msn,
        split=split,
        cohorts=frozenset(cohorts),
        utc_days=utc_days_for_interval(firstseen, lastseen),
        acquisition_key=_acquisition_key(identity),
    )


def _digest_payload(flights: tuple[SelectedFlight, ...]) -> list[dict[str, object]]:
    return [
        {
            **_identity_payload(_identity(flight)),
            "cohorts": sorted(flight.cohorts),
            "utc_days": list(flight.utc_days),
        }
        for flight in flights
    ]


def compile_selection(rows: Iterable[Mapping[str, object]]) -> SelectionPlan:
    """Mutualise identical flights while preserving every cohort assignment."""
    flights = _selection_flights(_cohorts_by_identity(rows))
    return SelectionPlan(flights=flights, digest=_selection_digest(flights))
