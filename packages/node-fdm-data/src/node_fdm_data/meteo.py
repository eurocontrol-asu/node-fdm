"""Meteorological computations for flight data processing.

Core functions: great-circle distance (haversine), Mach / CAS derivation,
and TAS computation from wind + groundspeed.  All pure functions operating
on numpy arrays or Polars expressions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from node_fdm_data.physics.constants import A0, GAMMA_AIR, R
from node_fdm_data.physics.isa import isa_pressure

__all__ = [
    "compute_mach_and_cas",
    "compute_tas",
    "haversine",
]

EARTH_RADIUS_M: float = 6_371_000.0


def haversine(
    lat1: np.ndarray[Any, np.dtype[Any]],
    lon1: np.ndarray[Any, np.dtype[Any]],
    lat2: np.ndarray[Any, np.dtype[Any]],
    lon2: np.ndarray[Any, np.dtype[Any]],
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    """Great-circle distance between coordinate pairs (metres).

    All inputs are in **degrees**.

    Args:
        lat1: Latitudes of first points.
        lon1: Longitudes of first points.
        lat2: Latitudes of second points.
        lon2: Longitudes of second points.

    Returns:
        Array of distances in metres.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    result: np.ndarray[Any, np.dtype[np.floating[Any]]] = (
        2 * EARTH_RADIUS_M * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    )
    return result


def compute_mach_and_cas(
    tas_kt: np.ndarray[Any, np.dtype[Any]],
    alt_ft: np.ndarray[Any, np.dtype[Any]],
    temp_k: np.ndarray[Any, np.dtype[Any]],
) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """Compute Mach number and CAS from TAS, altitude, and temperature.

    Args:
        tas_kt: True airspeed in **knots**.
        alt_ft: Geometric altitude in **feet**.
        temp_k: Static air temperature in **Kelvin**.

    Returns:
        ``(mach, cas_kt)`` — Mach number (dimensionless) and calibrated
        airspeed in knots.
    """
    tas = np.asarray(tas_kt, dtype=np.float64) * 0.514444  # kt → m/s
    h = np.asarray(alt_ft, dtype=np.float64) * 0.3048  # ft → m

    a = np.sqrt(GAMMA_AIR * R * np.asarray(temp_k, dtype=np.float64))
    mach = tas / a

    p = isa_pressure(h)

    # Impact pressure ratio
    pt_over_p = (1 + (GAMMA_AIR - 1) / 2 * mach**2) ** (GAMMA_AIR / (GAMMA_AIR - 1))
    qc_p0 = (np.asarray(p) / 101_325.0) * (pt_over_p - 1)

    cas = A0 * np.sqrt(
        (2 / (GAMMA_AIR - 1)) * (((qc_p0 + 1) ** ((GAMMA_AIR - 1) / GAMMA_AIR)) - 1)
    )
    cas_kt = cas / 0.514444  # m/s → kt

    return mach, cas_kt


def compute_tas(
    gs_col: str = "raw_gs_kt",
    track_col: str = "raw_track_deg",
    u_wind_col: str = "u_component_of_wind",
    v_wind_col: str = "v_component_of_wind",
) -> pl.Expr:
    """Polars expression computing TAS from groundspeed and wind components.

    Wind components are expected in **m/s**, groundspeed and output in
    **knots**.

    Args:
        gs_col: Name of the groundspeed column (knots).
        track_col: Name of the track angle column (degrees).
        u_wind_col: Name of the u-wind column (m/s).
        v_wind_col: Name of the v-wind column (m/s).

    Returns:
        A ``pl.Expr`` evaluating to TAS in knots.
    """
    import polars as pl

    ms_to_kt = 1.94384
    u_wind_kt = pl.col(u_wind_col) * ms_to_kt
    v_wind_kt = pl.col(v_wind_col) * ms_to_kt

    track_rad = pl.col(track_col).radians()
    u_ground = pl.col(gs_col) * track_rad.sin()
    v_ground = pl.col(gs_col) * track_rad.cos()

    return ((u_ground - u_wind_kt).pow(2) + (v_ground - v_wind_kt).pow(2)).sqrt()
