"""Derive physics columns (pipeline v3, étape 4).

Computes flight-path angle, longitudinal wind, altitude difference,
cumulative along-track distance, and great-circle distances to
departure/arrival airports.
"""

from __future__ import annotations

import polars as pl

from node_fdm_data.meteo import haversine_expr
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
    # TAS source = bds_tas_from_cas_kt (clean BDS-derived TAS) instead of raw
    # era_tas_kt: keeps fdm_gamma_rad / fdm_long_wind_kt consistent with the
    # speeds produced by the clean-speeds stage.
    vz_ms = pl.col("raw_vz_ftmin") * FTMIN
    tas_ms = pl.col("bds_tas_from_cas_kt") * KT
    ratio = (vz_ms / tas_ms.clip(lower_bound=1e-6)).clip(-1.0, 1.0)

    df = df.with_columns(
        ratio.arcsin().alias("fdm_gamma_rad"),
        (pl.col("bds_tas_from_cas_kt") - pl.col("raw_gs_kt")).alias("fdm_long_wind_kt"),
        (pl.col("bds_mcp_sel_alt_ft") - pl.col("raw_alt_ft")).alias("fdm_alt_diff_ft"),
    )

    # --- Cumulative distance (per flight) ---
    df = _cumulative_distance_per_flight(df)

    # --- Airport distances ---
    df = _airport_distances(df, airport_coords)

    return df


def _cumulative_distance_per_flight(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``fdm_distance_cum_m`` via per-flight haversine accumulation."""
    df = df.with_columns(
        pl.col("raw_lat_deg").shift(1).over("meta_flight_id").alias("_prev_lat"),
        pl.col("raw_lon_deg").shift(1).over("meta_flight_id").alias("_prev_lon"),
    )
    df = df.with_columns(
        haversine_expr("_prev_lat", "_prev_lon", "raw_lat_deg", "raw_lon_deg")
        .fill_null(0.0)
        .cum_sum()
        .over("meta_flight_id")
        .alias("fdm_distance_cum_m"),
    )
    return df.drop("_prev_lat", "_prev_lon")


def _airport_distances(
    df: pl.DataFrame,
    airport_coords: dict[str, tuple[float, float]] | None,
) -> pl.DataFrame:
    """Add ``fdm_adep_dist_nm`` and ``fdm_ades_dist_nm``.

    When *airport_coords* is ``None`` or an airport ICAO code is missing,
    the corresponding column is filled with null.
    """
    has_departure = "meta_departure" in df.columns
    has_arrival = "meta_arrival" in df.columns

    if not airport_coords:
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("fdm_adep_dist_nm"),
            pl.lit(None, dtype=pl.Float64).alias("fdm_ades_dist_nm"),
        )

    ap_df = pl.DataFrame(
        {
            "_ap_icao": list(airport_coords.keys()),
            "_ap_lat": [c[0] for c in airport_coords.values()],
            "_ap_lon": [c[1] for c in airport_coords.values()],
        }
    )

    if has_departure:
        df = (
            df.cast({"meta_departure": pl.Utf8})
            .join(ap_df, left_on="meta_departure", right_on="_ap_icao", how="left")
            .with_columns(
                (
                    haversine_expr("raw_lat_deg", "raw_lon_deg", "_ap_lat", "_ap_lon") / _M_PER_NM
                ).alias("fdm_adep_dist_nm"),
            )
            .drop("_ap_lat", "_ap_lon")
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("fdm_adep_dist_nm"))

    if has_arrival:
        df = (
            df.cast({"meta_arrival": pl.Utf8})
            .join(ap_df, left_on="meta_arrival", right_on="_ap_icao", how="left")
            .with_columns(
                (
                    haversine_expr("raw_lat_deg", "raw_lon_deg", "_ap_lat", "_ap_lon") / _M_PER_NM
                ).alias("fdm_ades_dist_nm"),
            )
            .drop("_ap_lat", "_ap_lon")
        )
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("fdm_ades_dist_nm"))

    return df
