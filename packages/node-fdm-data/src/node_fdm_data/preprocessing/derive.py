"""Derive physics columns (pipeline v3, étape 4).

Computes flight-path angle, longitudinal wind, altitude difference,
cumulative along-track distance, and great-circle distances to
departure/arrival airports.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_data.lateral import augment_lateral
from node_fdm_data.meteo import haversine_expr
from node_fdm_data.physics.constants import FTMIN, KT
from node_fdm_data.physics.magnetic import magnetic_declination
from node_fdm_data.preprocessing.lateral_state import (
    clean_track_with_medfilt,
    coalesce_heading,
    compute_drift_from_wind,
    compute_wind_std,
)

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
    Also adds the lateral-channel columns: ``fdm_track_clean_deg``,
    ``fdm_declination_deg``, ``fdm_drift_deg``, ``fdm_wind_std_ms``,
    ``fdm_heading_deg``, ``fdm_heading_target_deg``, ``fdm_heading_known``,
    ``fdm_heading_target_known``, plus ``in_turn``, ``track_ortho``,
    ``track_sel_known`` (from :func:`augment_lateral`).

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
    # TAS source = fdm_tas_from_cas_kt (clean BDS-derived TAS) instead of raw
    # era_tas_kt: keeps fdm_gamma_rad / fdm_long_wind_kt consistent with the
    # speeds produced by the clean-speeds stage.
    vz_ms = pl.col("raw_vz_ftmin") * FTMIN
    tas_ms = pl.col("fdm_tas_from_cas_kt") * KT
    ratio = (vz_ms / tas_ms.clip(lower_bound=1e-6)).clip(-1.0, 1.0)

    df = df.with_columns(
        ratio.arcsin().alias("fdm_gamma_rad"),
        (pl.col("fdm_tas_from_cas_kt") - pl.col("raw_gs_kt")).alias("fdm_long_wind_kt"),
        (pl.col("bds_mcp_sel_alt_ft") - pl.col("raw_alt_ft")).alias("fdm_alt_diff_ft"),
    )

    # --- Cumulative distance (per flight) ---
    df = _cumulative_distance_per_flight(df)

    # --- Airport distances ---
    df = _airport_distances(df, airport_coords)

    # --- Lateral channel columns (per flight) ---
    df = _augment_lateral_per_flight(df)

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


def _augment_lateral_per_flight(df: pl.DataFrame) -> pl.DataFrame:
    """Add lateral-channel columns (``fdm_heading_*``, ``track_ortho``, ...).

    Operates per ``meta_flight_id`` (turn detection and reference track
    are inherently per-flight) and concatenates results.  When BDS heading
    or wind data are missing, ``fdm_heading_known`` is set to False and
    the heading falls back to NaN.
    """
    needed = {
        "raw_track_deg",
        "raw_lat_deg",
        "raw_lon_deg",
        "raw_alt_ft",
        "raw_timestamp",
        "bds_hdg_deg",
        "era_u_wind_ms",
        "era_v_wind_ms",
    }
    missing = needed - set(df.columns)
    if missing:
        # Fall back to NaN/False columns so the downstream schema does not
        # break.  Practical case: very early pipeline stages without full ERA
        # enrichment (caller is expected to ensure all columns are present).
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("fdm_track_clean_deg"),
            pl.lit(None, dtype=pl.Float64).alias("fdm_declination_deg"),
            pl.lit(None, dtype=pl.Float64).alias("fdm_drift_deg"),
            pl.lit(None, dtype=pl.Float64).alias("fdm_wind_std_ms"),
            pl.lit(None, dtype=pl.Float64).alias("fdm_heading_deg"),
            pl.lit(None, dtype=pl.Float64).alias("fdm_heading_target_deg"),
            pl.lit(False).alias("fdm_heading_known"),
            pl.lit(False).alias("fdm_heading_target_known"),
            pl.lit(True).alias("in_turn"),
            pl.lit(None, dtype=pl.Float64).alias("track_ortho"),
            pl.lit(False).alias("track_sel_known"),
        )

    # Pick a TAS source consistent with the longitudinal channel:
    # fdm_tas_from_cas_kt is the cleaned TAS used by clean-speeds; fallback
    # to era_tas_kt where the cleaned value is missing.
    tas_kt: pl.Expr
    if "fdm_tas_from_cas_kt" in df.columns:
        if "era_tas_kt" in df.columns:
            tas_kt = pl.col("fdm_tas_from_cas_kt").fill_null(pl.col("era_tas_kt"))
        else:
            tas_kt = pl.col("fdm_tas_from_cas_kt")
    elif "era_tas_kt" in df.columns:
        tas_kt = pl.col("era_tas_kt")
    else:
        tas_kt = pl.lit(None, dtype=pl.Float64)
    df = df.with_columns(tas_kt.alias("_lateral_tas_kt"))

    parts: list[pl.DataFrame] = []
    for (_flight_id,), flight_df in df.group_by("meta_flight_id", maintain_order=True):
        parts.append(_lateral_columns_for_flight(flight_df))

    if not parts:
        return df.drop("_lateral_tas_kt")

    out = pl.concat(parts, how="vertical").drop("_lateral_tas_kt")
    return out


def _lateral_columns_for_flight(flight: pl.DataFrame) -> pl.DataFrame:
    """Compute lateral columns for a single flight DataFrame."""
    n = flight.height
    if n == 0:
        return flight

    # 1. augment_lateral expects renamed columns; build a thin view, run it,
    #    pull the three output columns back, then attach to the original.
    renamed = flight.rename(
        {
            "raw_lat_deg": "latitude",
            "raw_lon_deg": "longitude",
            "raw_track_deg": "track",
            "bds_hdg_deg": "heading",
            "_lateral_tas_kt": "TAS",
        }
    )
    augmented = augment_lateral(renamed)
    in_turn = augmented["in_turn"].to_numpy()
    track_ortho = augmented["track_ortho"].to_numpy().astype(np.float64)
    track_sel_known = augmented["track_sel_known"].to_numpy()

    # 2. Track cleaning (median filter + Savgol).
    raw_track = flight["raw_track_deg"].to_numpy().astype(np.float64)
    track_clean = clean_track_with_medfilt(raw_track)

    # 3. Magnetic declination (one decimal-year value per flight).
    lat = flight["raw_lat_deg"].to_numpy().astype(np.float64)
    lon = flight["raw_lon_deg"].to_numpy().astype(np.float64)
    alt_ft = flight["raw_alt_ft"].to_numpy().astype(np.float64)
    ts_series = flight["raw_timestamp"]
    mid_ts = ts_series[ts_series.len() // 2]
    declination = magnetic_declination(lat, lon, alt_ft, mid_ts)

    # 4. Gather wind/TAS inputs for drift and heading coalescence.
    bds_hdg = flight["bds_hdg_deg"].to_numpy().astype(np.float64)
    tas_kt = flight["_lateral_tas_kt"].to_numpy().astype(np.float64)
    tas_ms = tas_kt * KT
    u_wind = flight["era_u_wind_ms"].to_numpy().astype(np.float64)
    v_wind = flight["era_v_wind_ms"].to_numpy().astype(np.float64)

    # 5. Coalesce heading: BDS+declination primary, track-based drift fallback.
    #    The fallback uses track_clean as a heading proxy inside the wind
    #    triangle (second-order error, see coalesce_heading docstring).
    heading_deg, heading_known = coalesce_heading(
        bds_hdg, declination, track_clean, tas_ms, u_wind, v_wind
    )

    # 6. Drift from the resolved heading — defined wherever heading is known.
    drift = compute_drift_from_wind(heading_deg, tas_ms, u_wind, v_wind)

    # 7. Wind std (diagnostic only, exposed via fdm_wind_std_ms).
    wind_std = compute_wind_std(u_wind, v_wind)

    # 8. Heading target = wrap(track_ortho - drift) — undefined where
    #    track_ortho is NaN (head/tail of flight, in-turn samples).
    heading_target_deg = (track_ortho - drift) % 360.0

    target_known = heading_known & track_sel_known & np.isfinite(heading_target_deg)
    # NaN-out the target where it is not known so downstream code can rely on
    # NaN as the missing-target sentinel (mirrors fdm_alt_target_*, etc.).
    heading_target_deg = np.where(target_known, heading_target_deg, np.nan)

    return flight.with_columns(
        pl.Series("fdm_track_clean_deg", track_clean),
        pl.Series("fdm_declination_deg", declination),
        pl.Series("fdm_drift_deg", drift),
        pl.Series("fdm_wind_std_ms", wind_std),
        pl.Series("fdm_heading_deg", heading_deg),
        pl.Series("fdm_heading_target_deg", heading_target_deg),
        pl.Series("fdm_heading_known", heading_known),
        pl.Series("fdm_heading_target_known", target_known),
        pl.Series("in_turn", in_turn),
        pl.Series("track_ortho", track_ortho),
        pl.Series("track_sel_known", track_sel_known),
    )
