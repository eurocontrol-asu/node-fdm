"""Frozen, in-memory OpenSky corpus shared by pipeline tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import NamedTuple

import polars as pl

__all__ = ["build_raw_opensky_day"]


class _RawPoint(NamedTuple):
    selection_id: str
    cohort: str
    meta_selection_day: str
    meta_source_day: str
    meta_aircraft_type: str
    raw_icao24: str
    raw_timestamp: datetime
    raw_lat_deg: float
    raw_lon_deg: float
    raw_alt_ft: float
    raw_gs_kt: float
    raw_track_deg: float


_DAY_ROWS: dict[str, tuple[_RawPoint, ...]] = {
    "20200101": (
        _RawPoint(
            "sel-a20n",
            "A20N",
            "20200101",
            "20200101",
            "A20N",
            "39a20a",
            datetime(2020, 1, 1, 10, 0, tzinfo=UTC),
            48.85,
            2.35,
            12_000.0,
            280.0,
            90.0,
        ),
        _RawPoint(
            "sel-a20n",
            "A20N",
            "20200101",
            "20200101",
            "A20N",
            "39a20a",
            datetime(2020, 1, 1, 10, 1, tzinfo=UTC),
            48.86,
            2.42,
            12_500.0,
            285.0,
            91.0,
        ),
        _RawPoint(
            "sel-b738",
            "B738",
            "20200101",
            "20200101",
            "B738",
            "4bb738",
            datetime(2020, 1, 1, 12, 0, tzinfo=UTC),
            50.10,
            3.20,
            31_000.0,
            440.0,
            220.0,
        ),
        _RawPoint(
            "sel-b738",
            "B738",
            "20200101",
            "20200101",
            "B738",
            "4bb738",
            datetime(2020, 1, 1, 12, 1, tzinfo=UTC),
            50.04,
            3.10,
            30_800.0,
            438.0,
            221.0,
        ),
        _RawPoint(
            "sel-a20n-midnight",
            "A20N",
            "20200101",
            "20200101",
            "A20N",
            "39a20f",
            datetime(2020, 1, 1, 23, 58, tzinfo=UTC),
            47.90,
            1.80,
            28_000.0,
            410.0,
            45.0,
        ),
        _RawPoint(
            "sel-a20n-midnight",
            "A20N",
            "20200101",
            "20200101",
            "A20N",
            "39a20f",
            datetime(2020, 1, 1, 23, 59, tzinfo=UTC),
            47.95,
            1.88,
            27_500.0,
            405.0,
            46.0,
        ),
    ),
    "20200102": (
        _RawPoint(
            "sel-a20n-midnight",
            "A20N",
            "20200101",
            "20200102",
            "A20N",
            "39a20f",
            datetime(2020, 1, 2, 0, 1, tzinfo=UTC),
            48.05,
            2.04,
            26_500.0,
            395.0,
            48.0,
        ),
        _RawPoint(
            "sel-a20n-midnight",
            "A20N",
            "20200101",
            "20200102",
            "A20N",
            "39a20f",
            datetime(2020, 1, 2, 0, 2, tzinfo=UTC),
            48.10,
            2.12,
            26_000.0,
            390.0,
            49.0,
        ),
        _RawPoint(
            "sel-e190",
            "E190",
            "20200102",
            "20200102",
            "E190",
            "3ce190",
            datetime(2020, 1, 2, 8, 0, tzinfo=UTC),
            46.20,
            4.10,
            18_000.0,
            330.0,
            180.0,
        ),
        _RawPoint(
            "sel-crj9",
            "CRJ9",
            "20200102",
            "20200102",
            "CRJ9",
            "4cc9a0",
            datetime(2020, 1, 2, 9, 0, tzinfo=UTC),
            47.20,
            5.10,
            22_000.0,
            360.0,
            200.0,
        ),
        _RawPoint(
            "sel-cl35",
            "CL35",
            "20200102",
            "20200102",
            "CL35",
            "43c135",
            datetime(2020, 1, 2, 11, 0, tzinfo=UTC),
            45.80,
            1.20,
            35_000.0,
            455.0,
            270.0,
        ),
    ),
}


def build_raw_opensky_day(day: str) -> pl.DataFrame:
    """Return a fresh frame for one frozen source day without performing I/O."""
    try:
        rows = _DAY_ROWS[day]
    except KeyError as exc:
        msg = f"unsupported frozen OpenSky day: {day}"
        raise ValueError(msg) from exc

    return pl.DataFrame(rows, schema=list(_RawPoint._fields), orient="row")
