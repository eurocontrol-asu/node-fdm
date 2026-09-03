"""Deterministic compilation of in-memory fleet-selection rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import cast

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "SelectedFlight",
    "SelectionPlan",
    "compile_selection",
    "utc_days_for_interval",
]

type FlightIdentity = tuple[str, str, str, int, int, str, str]


class _SelectionRow(BaseModel):
    """Validated shape consumed from one source selection row."""

    icao24: str
    callsign: str
    firstseen: int
    lastseen: int
    msn: str
    split: str
    cohort: str
    selection_id: str
    utc_days: tuple[str, ...] | None = None

    @field_validator("selection_id")
    @classmethod
    def _validate_selection_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("selection_id must be a non-empty string")
        return value

    @field_validator("callsign")
    @classmethod
    def _normalize_callsign(cls, value: str) -> str:
        return value.strip().upper()


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
    selection_id: str
    utc_days: tuple[str, ...]
    acquisition_key: str


def compile_selection_text(raw: str) -> SelectionPlan | None:
    """Compile structured campaign rows while preserving opaque legacy digests."""
    try:
        parsed = cast("object", json.loads(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(isinstance(row, dict) for row in parsed):
        raise ValueError("campaign selection must be a JSON list of rows")
    return compile_selection(cast("list[dict[str, object]]", parsed))


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
        yield day.strftime("%Y%m%d")
        day += timedelta(days=1)


def _identity(row: _SelectionRow | SelectedFlight) -> FlightIdentity:
    return (
        row.selection_id,
        row.icao24,
        row.callsign,
        row.firstseen,
        row.lastseen,
        row.msn,
        row.split,
    )


def _identity_payload(identity: FlightIdentity) -> dict[str, str | int]:
    selection_id, icao24, callsign, firstseen, lastseen, msn, split = identity
    return {
        "selection_id": selection_id,
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
    identity_payload = _identity_payload(identity)
    acquisition_payload = {
        key: value for key, value in identity_payload.items() if key != "selection_id"
    }
    canonical_identity = _canonical_json(acquisition_payload)
    return hashlib.sha256(canonical_identity.encode()).hexdigest()


def _cohorts_by_identity(
    rows: Iterable[Mapping[str, object] | _SelectionRow],
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
    selection_id, icao24, callsign, firstseen, lastseen, msn, split = identity
    return SelectedFlight(
        selection_id=selection_id,
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
    validated_rows = tuple(_SelectionRow.model_validate(row) for row in rows)
    explicit_days: dict[FlightIdentity, tuple[str, ...]] = {}
    for row in validated_rows:
        if row.utc_days is None:
            continue
        identity = _identity(row)
        previous = explicit_days.setdefault(identity, row.utc_days)
        if previous != row.utc_days:
            msg = f"conflicting utc_days for selection {row.selection_id!r}"
            raise ValueError(msg)

    compiled = _selection_flights(_cohorts_by_identity(validated_rows))
    flights = tuple(
        flight.model_copy(
            update={"utc_days": explicit_days.get(_identity(flight), flight.utc_days)}
        )
        for flight in compiled
    )
    return SelectionPlan(flights=flights, digest=_selection_digest(flights))
