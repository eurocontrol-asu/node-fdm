"""Magnetic declination via the World Magnetic Model (WMM).

Provides per-sample magnetic declination in degrees, used to convert the
BDS magnetic heading to a true (geographic) heading for the lateral
channel.

The WMM coefficients are loaded once at module import (single GeoMag
instance — coefficient load dominates per-call cost).  Declination drifts
at most ~0.1 deg/yr, so a single decimal-year value derived from one
flight timestamp is sufficient for the whole flight.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import numpy.typing as npt
from pygeomag import GeoMag  # type: ignore[import-untyped]

from node_fdm_data.physics.wmm import declination_vec

__all__ = [
    "magnetic_declination",
]

# Module-level instance: WMM-2025 coefficients shipped with pygeomag.
# Reused across calls because the model-file load is the bulk of the cost.
_GEOMAG = GeoMag()

_FT_TO_KM: float = 0.3048 / 1000.0


def _decimal_year(ts: _dt.datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=_dt.UTC)
    year = ts.year
    start = _dt.datetime(year, 1, 1, tzinfo=_dt.UTC)
    end = _dt.datetime(year + 1, 1, 1, tzinfo=_dt.UTC)
    return year + (ts - start).total_seconds() / (end - start).total_seconds()


def magnetic_declination(
    lat_deg: npt.NDArray[np.floating],
    lon_deg: npt.NDArray[np.floating],
    alt_ft: npt.NDArray[np.floating],
    timestamp: _dt.datetime,
) -> npt.NDArray[np.float64]:
    """Per-sample magnetic declination in degrees via WMM (pygeomag).

    Args:
        lat_deg: Latitude in degrees, shape ``(n,)``.
        lon_deg: Longitude in degrees, shape ``(n,)``.
        alt_ft: Altitude in feet, shape ``(n,)``.  Converted internally to
            km.  NaN values fall back to 0 km (effect on declination is
            < 0.01° between 0 and FL400).
        timestamp: Any datetime within the flight.  Used to compute one
            decimal-year value reused for every sample (declination drifts
            < 0.1°/yr).

    Returns:
        Declination in degrees, shape ``(n,)``.  NaN where lat or lon is
        NaN, or where the WMM evaluation fails.
    """
    lat_arr = np.asarray(lat_deg, dtype=np.float64)
    lon_arr = np.asarray(lon_deg, dtype=np.float64)
    alt_arr = np.asarray(alt_ft, dtype=np.float64)

    n = lat_arr.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    bad = np.isnan(lat_arr) | np.isnan(lon_arr)
    if bool(bad.all()):
        return out

    alt_km = np.where(np.isnan(alt_arr), 0.0, alt_arr * _FT_TO_KM)
    decimal_year = _decimal_year(timestamp)

    # Mask NaN inputs to a safe value so the vectorized eval does not poison
    # neighbouring samples; results at masked indices are restored to NaN.
    lat_safe = np.where(bad, 0.0, lat_arr)
    lon_safe = np.where(bad, 0.0, lon_arr)
    alt_safe = np.where(bad, 0.0, alt_km)

    declination = declination_vec(_GEOMAG, lat_safe, lon_safe, alt_safe, decimal_year)
    return np.where(bad, np.nan, declination)
