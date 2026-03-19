"""Derive physics columns (pipeline v3, étape 4).

Computes flight-path angle, longitudinal wind, altitude difference,
cumulative along-track distance, and great-circle distances to
departure/arrival airports.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_data.meteo import haversine
from node_fdm_data.physics.constants import FTMIN, KT

__all__ = [
    "derive_columns",
]

# Metres per nautical mile.
_M_PER_NM: float = 1852.0


def derive_columns(
    df: pl.DataFrame,
    *,
    airport_coords: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame:
    """Compute derived physics columns for étape 4.

    Adds ``fdm_gamma_rad``, ``fdm_long_wind_kt``, ``fdm_alt_diff_ft``,
    ``fdm_distance_cum_m``, ``fdm_adep_dist_nm``, and ``fdm_ades_dist_nm``.

    All per-flight computations are grouped by ``meta_flight_id``.

    Args:
        df: DataFrame with ``raw_*``, ``era_*``, ``bds_*``, ``meta_*`` columns.
        airport_coords: Optional mapping of ICAO code to ``(lat, lon)`` in
            degrees.  When provided, airport distance columns are computed;
            otherwise they are filled with ``NaN``.

    Returns:
        DataFrame with ``fdm_*`` derived columns added.
    """
    # --- Expression-based columns (vectorised, no grouping needed) ---
    vz_ms = pl.col("raw_vz_ftmin") * FTMIN
    tas_ms = pl.col("era_tas_kt") * KT
    ratio = (vz_ms / tas_ms.clip(lower_bound=1e-6)).clip(-1.0, 1.0)

    df = df.with_columns(
        ratio.arcsin().alias("fdm_gamma_rad"),
        (pl.col("era_tas_kt") - pl.col("raw_gs_kt")).alias("fdm_long_wind_kt"),
        (pl.col("bds_mcp_sel_alt_ft") - pl.col("raw_alt_ft")).alias("fdm_alt_diff_ft"),
    )

    # --- Cumulative distance (per flight) ---
    df = _cumulative_distance_per_flight(df)

    # --- Airport distances ---
    df = _airport_distances(df, airport_coords)

    return df


def _cumulative_distance_per_flight(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``fdm_distance_cum_m`` via per-flight haversine accumulation."""
    parts = df.partition_by("meta_flight_id", maintain_order=True)
    out: list[pl.DataFrame] = []
    for part in parts:
        lat = part["raw_lat_deg"].to_numpy()
        lon = part["raw_lon_deg"].to_numpy()

        if len(lat) < 2:  # noqa: PLR2004
            part = part.with_columns(pl.lit(0.0).alias("fdm_distance_cum_m"))
        else:
            d = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
            cum_d = np.concatenate(([0.0], np.cumsum(d)))
            part = part.with_columns(pl.Series("fdm_distance_cum_m", cum_d))
        out.append(part)

    return pl.concat(out, how="vertical_relaxed")


def _airport_distances(
    df: pl.DataFrame,
    airport_coords: dict[str, tuple[float, float]] | None,
) -> pl.DataFrame:
    """Add ``fdm_adep_dist_nm`` and ``fdm_ades_dist_nm``.

    When *airport_coords* is ``None`` or an airport ICAO code is missing,
    the corresponding column is filled with ``NaN``.
    """
    has_departure = "meta_departure" in df.columns
    has_arrival = "meta_arrival" in df.columns

    lat = df["raw_lat_deg"].to_numpy()
    lon = df["raw_lon_deg"].to_numpy()

    adep_dist = np.full(len(df), np.nan)
    ades_dist = np.full(len(df), np.nan)

    if airport_coords and has_departure:
        departures = df["meta_departure"].to_list()
        for icao in set(departures):
            if icao is None or icao not in airport_coords:
                continue
            ap_lat, ap_lon = airport_coords[icao]
            mask = np.array([d == icao for d in departures])
            n = mask.sum()
            d_m = haversine(lat[mask], lon[mask], np.full(n, ap_lat), np.full(n, ap_lon))
            adep_dist[mask] = d_m / _M_PER_NM

    if airport_coords and has_arrival:
        arrivals = df["meta_arrival"].to_list()
        for icao in set(arrivals):
            if icao is None or icao not in airport_coords:
                continue
            ap_lat, ap_lon = airport_coords[icao]
            mask = np.array([a == icao for a in arrivals])
            n = mask.sum()
            d_m = haversine(lat[mask], lon[mask], np.full(n, ap_lat), np.full(n, ap_lon))
            ades_dist[mask] = d_m / _M_PER_NM

    return df.with_columns(
        pl.Series("fdm_adep_dist_nm", adep_dist),
        pl.Series("fdm_ades_dist_nm", ades_dist),
    )
