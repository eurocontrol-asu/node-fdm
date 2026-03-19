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
    "enrich_era5",
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


_ERA5_RENAME: dict[str, str] = {
    "temperature": "era_temp_K",
    "u_component_of_wind": "era_u_wind_ms",
    "v_component_of_wind": "era_v_wind_ms",
}

# fastmeteo expects these column names
_FASTMETEO_INPUT_RENAME: dict[str, str] = {
    "raw_lat_deg": "latitude",
    "raw_lon_deg": "longitude",
    "raw_alt_ft": "altitude",
    "raw_timestamp": "timestamp",
}

_FASTMETEO_INPUT_RESTORE: dict[str, str] = {v: k for k, v in _FASTMETEO_INPUT_RENAME.items()}


def enrich_era5(
    df: pl.DataFrame,
    arco_grid: Any,
) -> pl.DataFrame:
    """Enrich a DataFrame with ERA5 weather data and derived airspeed columns.

    Calls fastmeteo to interpolate ERA5 weather variables, then computes
    ``era_tas_kt``, ``era_mach``, and ``era_cas_kt`` from the ERA5 wind
    and temperature fields.

    Existing ``bds_*`` columns are never modified.

    Args:
        df: DataFrame with ``raw_lat_deg``, ``raw_lon_deg``, ``raw_alt_ft``,
            ``raw_timestamp``, ``raw_gs_kt``, ``raw_track_deg`` columns.
        arco_grid: A ``fastmeteo.source.arco_era5.ArcoEra5`` instance
            (or any object with an ``interpolate(pd.DataFrame)`` method).

    Returns:
        DataFrame with ``era_temp_K``, ``era_u_wind_ms``, ``era_v_wind_ms``,
        ``era_tas_kt``, ``era_mach``, ``era_cas_kt`` columns added.
    """
    # Rename to fastmeteo convention
    df_fm = df.rename(_FASTMETEO_INPUT_RENAME)

    # Strip timezone if present (fastmeteo expects naive timestamps)
    ts_dtype = df_fm.schema["timestamp"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        df_fm = df_fm.with_columns(pl.col("timestamp").dt.replace_time_zone(None))

    # Call fastmeteo via pandas interop
    pd_df = df_fm.to_pandas()
    pd_df = arco_grid.interpolate(pd_df)
    df_fm = pl.from_pandas(pd_df)

    # Restore original column names and rename ERA5 outputs
    df_fm = df_fm.rename(_FASTMETEO_INPUT_RESTORE)
    df_fm = df_fm.rename(_ERA5_RENAME)

    # Compute era_tas_kt
    df_fm = df_fm.with_columns(
        compute_tas(
            gs_col="raw_gs_kt",
            track_col="raw_track_deg",
            u_wind_col="era_u_wind_ms",
            v_wind_col="era_v_wind_ms",
        ).alias("era_tas_kt"),
    )

    # Compute era_mach and era_cas_kt
    mach_arr, cas_arr = compute_mach_and_cas(
        tas_kt=df_fm["era_tas_kt"].to_numpy(),
        alt_ft=df_fm["raw_alt_ft"].to_numpy(),
        temp_k=df_fm["era_temp_K"].to_numpy(),
    )
    df_fm = df_fm.with_columns(
        pl.Series("era_mach", mach_arr),
        pl.Series("era_cas_kt", cas_arr),
    )

    return df_fm


def compute_tas(
    gs_col: str = "raw_gs_kt",
    track_col: str = "raw_track_deg",
    u_wind_col: str = "era_u_wind_ms",
    v_wind_col: str = "era_v_wind_ms",
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
