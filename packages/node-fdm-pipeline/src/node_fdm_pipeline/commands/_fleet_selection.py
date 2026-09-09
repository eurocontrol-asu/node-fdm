"""Deterministic compilation of in-memory fleet-selection rows."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "SelectedFlight",
    "SelectionFormatError",
    "SelectionPlan",
    "SelectionSource",
    "compile_selection",
    "compile_selection_csv",
    "load_selection_file",
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


class SelectionFormatError(ValueError):
    """Raised when an explicitly structured selection source is malformed."""


@dataclass(frozen=True)
class SelectionSource:
    """Raw selection evidence paired with its optional compiled plan."""

    raw: str
    plan: SelectionPlan | None


_REQUIRED_CSV_FIELDS = (
    "selection_id",
    "icao24",
    "callsign",
    "firstseen",
    "lastseen",
    "msn",
    "split",
    "cohorts",
    "utc_days",
)


def _selection_csv_row(row: Mapping[str, str | None]) -> dict[str, object]:
    normalized = _normalize_csv_identity(row)
    normalized["cohort"] = row["cohorts"] or ""
    normalized.pop("cohorts", None)
    raw_days = row["utc_days"] or ""
    normalized["utc_days"] = (
        tuple(day for value in raw_days.split("|") if (day := value.strip())) or None
    )
    return normalized


def _campaign_selection_csv_rows(
    row: Mapping[str, str | None],
) -> tuple[dict[str, object], ...]:
    normalized = _selection_csv_row(row)
    cohorts = tuple(value.strip() for value in (row["cohorts"] or "").split("|"))
    return tuple(normalized | {"cohort": cohort} for cohort in cohorts)


def compile_selection_csv(raw: str) -> SelectionPlan:
    """Compile a strict header-based CSV selection without partial rows."""
    fieldnames, rows = _read_selection_csv(raw)
    fields = set(fieldnames)
    if "cohort" in fields and "cohorts" in fields:
        raise SelectionFormatError("ambiguous CSV columns: cannot mix cohort and cohorts")
    required = (
        _COMPACT_CAMPAIGN_CSV_FIELDS
        if "start" in fields or "end" in fields
        else _REQUIRED_CSV_FIELDS
    )
    _require_csv_columns(fieldnames, required)
    return compile_selection(
        normalized for row in rows for normalized in _campaign_selection_csv_rows(row)
    )


def _compile_json_selection(raw: str) -> SelectionPlan:
    try:
        plan = compile_selection_text(raw)
    except ValueError as exc:
        raise SelectionFormatError(f"invalid JSON selection: {exc}") from exc
    if plan is None:
        raise SelectionFormatError("invalid JSON selection syntax")
    return plan


_COMPACT_CAMPAIGN_CSV_FIELDS = (
    "selection_id",
    "icao24",
    "callsign",
    "start",
    "end",
    "cohorts",
    "utc_days",
)
_HISTORICAL_CSV_FIELDS = (
    "selection_id",
    "icao24",
    "callsign",
    "start",
    "end",
    "cohort",
)
_RICH_HISTORICAL_CSV_FIELDS = (
    "selection_id",
    "icao24",
    "callsign",
    "firstseen",
    "lastseen",
    "msn",
    "split",
    "cohort",
)


def _read_selection_csv(
    raw: str,
) -> tuple[tuple[str, ...], Iterable[dict[str, str | None]]]:
    reader = csv.DictReader(StringIO(raw), strict=True)
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise SelectionFormatError("missing CSV header")
    try:
        for _row in reader:
            pass
    except csv.Error as exc:
        line_number = reader.line_num + 1
        raise SelectionFormatError(f"CSV syntax error on line {line_number}: {exc}") from exc

    stream_reader = csv.DictReader(StringIO(raw), strict=True)

    def rows() -> Iterable[dict[str, str | None]]:
        try:
            for row in stream_reader:
                yield dict(row)
        except csv.Error as exc:
            line_number = stream_reader.line_num + 1
            raise SelectionFormatError(f"CSV syntax error on line {line_number}: {exc}") from exc

    return tuple(fieldnames), rows()


def _require_csv_columns(fieldnames: tuple[str, ...], required: tuple[str, ...]) -> None:
    missing = [field for field in required if field not in fieldnames]
    if missing:
        columns = ", ".join(missing)
        raise SelectionFormatError(f"missing required CSV columns: {columns}")


def _normalize_csv_identity(row: Mapping[str, str | None]) -> dict[str, object]:
    normalized: dict[str, object] = dict(row)
    if "start" in row or "end" in row:
        normalized["firstseen"] = row.get("start") or ""
        normalized["lastseen"] = row.get("end") or ""
        normalized.setdefault("msn", "")
        normalized.setdefault("split", "")
        normalized.pop("start", None)
        normalized.pop("end", None)
    return normalized


def _historical_selection_csv_row(row: Mapping[str, str | None]) -> dict[str, object]:
    normalized = _normalize_csv_identity(row)
    firstseen = int(row.get("firstseen") or row.get("start") or "")
    lastseen = int(row.get("lastseen") or row.get("end") or "")
    normalized["utc_days"] = utc_days_for_interval(firstseen, lastseen)
    return normalized


def _compile_selection_source_csv(raw: str) -> SelectionPlan:
    fieldnames, rows = _read_selection_csv(raw)
    fields = set(fieldnames)
    if "cohort" in fields and "cohorts" in fields:
        raise SelectionFormatError("ambiguous CSV columns: cannot mix cohort and cohorts")
    if "cohort" in fields:
        required = (
            _HISTORICAL_CSV_FIELDS
            if "start" in fields or "end" in fields
            else _RICH_HISTORICAL_CSV_FIELDS
        )
        _require_csv_columns(fieldnames, required)
        return compile_selection(_historical_selection_csv_row(row) for row in rows)
    if "cohorts" in fields:
        return compile_selection_csv(raw)
    raise SelectionFormatError("CSV header must contain exactly one of cohort or cohorts")


def load_selection_file(path: Path) -> SelectionSource:
    """Load a structured selection by suffix or preserve a legacy opaque source."""
    raw = path.read_text()
    plan: SelectionPlan | None
    match path.suffix.lower():
        case ".csv":
            plan = _compile_selection_source_csv(raw)
        case ".json":
            plan = _compile_json_selection(raw)
        case _:
            plan = compile_selection_text(raw)
    return SelectionSource(raw=raw, plan=plan)


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
    explicit_days: dict[FlightIdentity, tuple[str, ...]] = {}
    cohorts_by_identity: dict[FlightIdentity, set[str]] = {}
    for raw_row in rows:
        row = _SelectionRow.model_validate(raw_row)
        identity = _identity(row)
        cohorts_by_identity.setdefault(identity, set()).add(row.cohort)
        if row.utc_days is None:
            continue
        previous = explicit_days.setdefault(identity, row.utc_days)
        if previous != row.utc_days:
            msg = f"conflicting utc_days for selection {row.selection_id!r}"
            raise ValueError(msg)

    compiled = _selection_flights(cohorts_by_identity)
    flights = tuple(
        flight.model_copy(
            update={"utc_days": explicit_days.get(_identity(flight), flight.utc_days)}
        )
        for flight in compiled
    )
    return SelectionPlan(flights=flights, digest=_selection_digest(flights))
