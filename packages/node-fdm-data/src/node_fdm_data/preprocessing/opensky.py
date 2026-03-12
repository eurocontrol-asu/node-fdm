"""OpenSky 2025 preprocessing pipeline.

Computes derived columns (altitude difference, flight-path angle,
longitudinal wind, cumulative distance) and fills missing control
inputs.  Also provides segment filtering and distance-jump cropping
for training data extraction.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_data.meteo import haversine
from node_fdm_data.physics.constants import FTMIN, KT

__all__ = [
    "LOW_THR",
    "UPPER_THR",
    "crop_on_distance_jump",
    "cumulative_distance",
    "flight_processing",
    "segment_filtering",
    "training_preprocessing",
]

LOW_THR: int = 200
"""Minimum acceptable distance-diff per timestep (metres)."""

UPPER_THR: int = 3000
"""Maximum acceptable distance-diff per timestep (metres)."""

MIN_SPEED_KT: float = 90.0
"""Minimum groundspeed (kt) to retain data during distance crop."""


def flight_processing(df: pl.LazyFrame) -> pl.LazyFrame:
    """Prepare OpenSky flight data for model training.

    Renames raw *traffic* columns to normalised schema names, computes
    physics-derived columns (``alt_diff_ft``, ``gamma_air``,
    ``long_wind``), and fills null control inputs with ``0.0``.

    The rename step is idempotent: columns that already carry the
    target name are left untouched.

    Args:
        df: LazyFrame from the preprocess step (traffic column names)
            or already-renamed data.

    Returns:
        LazyFrame with normalised names, derived columns, and nulls filled.
    """
    # --- Sort by timestamp (OpenSky data may arrive unsorted) ---
    schema = df.collect_schema()
    if "timestamp" in schema:
        df = df.sort("timestamp")

    # --- Rename traffic → schema (skip if already renamed) ---
    col_rename: dict[str, str] = {
        "altitude": "altitude_ft",
        "selected_mcp": "alt_sel_ft",
        "vertical_rate": "vz_sel_ftmin",
        "Mach": "mach_sel",
        "IAS": "cas_sel_kt",
        "TAS": "tas_kt",
        "groundspeed": "gs_kt",
    }
    schema = df.collect_schema()
    rename = {k: v for k, v in col_rename.items() if k in schema and v not in schema}
    if rename:
        df = df.rename(rename)

    # Refresh schema after rename
    schema = df.collect_schema()

    # --- Derived columns ---
    exprs: list[pl.Expr] = []

    # alt_diff_ft
    if "alt_sel_ft" in schema and "altitude_ft" in schema:
        exprs.append(
            (pl.col("alt_sel_ft") - pl.col("altitude_ft")).alias("alt_diff_ft"),
        )

    # gamma_air = arcsin(vz[ft/min] * FTMIN / (TAS[kt] * KT))
    tas_col = "tas_kt" if "tas_kt" in schema else "TAS"
    vz_col = "vz_sel_ftmin" if "vz_sel_ftmin" in schema else "vertical_rate"
    if tas_col in schema and vz_col in schema:
        vz_ms = pl.col(vz_col) * FTMIN  # ft/min → m/s
        tas_ms = pl.col(tas_col) * KT  # kt → m/s
        ratio = (vz_ms / tas_ms.clip(lower_bound=1e-6)).clip(-1.0, 1.0)
        exprs.append(ratio.arcsin().alias("gamma_air"))

    # long_wind = TAS - GS (knots)
    gs_col = "gs_kt" if "gs_kt" in schema else "groundspeed"
    if tas_col in schema and gs_col in schema:
        exprs.append(
            (pl.col(tas_col) - pl.col(gs_col)).alias("long_wind"),
        )

    # Fill nulls in control inputs
    fill_cols = {
        "vz_sel_ftmin": 0.0,
        "mach_sel": 0.0,
        "cas_sel_kt": 0.0,
    }
    for col, val in fill_cols.items():
        if col in schema:
            exprs.append(pl.col(col).fill_null(val))

    if exprs:
        df = df.with_columns(exprs)

    return df


def training_preprocessing(df: pl.DataFrame) -> pl.DataFrame:
    """Prepare processed flight data for the training loader.

    Bridges the gap between ``process`` command output (raw column names)
    and the training schema expected by :func:`get_train_val_data`.

    Steps:

    1. Run ``flight_processing`` on the eager DataFrame.
    2. Rename columns to match the schema:
       ``gamma_air`` → ``gamma_rad``,
       ``distance_along_track_m`` → ``distance_m``,
       ``long_wind`` → ``long_wind_kt``,
       ``adep_dist`` → ``adep_dist_nm``,
       ``ades_dist`` → ``ades_dist_nm``,
       ``temperature`` → ``temperature_K``.
    3. Compute finite-difference derivatives:
       ``vz_ftmin`` from ``altitude_ft``,
       ``d_gamma_rad`` from ``gamma_rad``,
       ``d_tas_kt`` from ``tas_kt``.

    Args:
        df: Eager DataFrame from processed parquet files.

    Returns:
        DataFrame ready for the training loader.
    """
    # Step 1: flight_processing (renames traffic cols, adds gamma_air, etc.)
    df = flight_processing(df.lazy()).collect()

    # Step 2: rename to schema names
    rename_map: dict[str, str] = {
        "gamma_air": "gamma_rad",
        "distance_along_track_m": "distance_m",
        "long_wind": "long_wind_kt",
        "adep_dist": "adep_dist_nm",
        "ades_dist": "ades_dist_nm",
        "temperature": "temperature_K",
    }
    rename = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if rename:
        df = df.rename(rename)

    # Step 3: compute derivatives via finite differences
    deriv_exprs: list[pl.Expr] = []

    # vz_ftmin: rate of climb in ft/min from altitude_ft
    # Δalt / Δt where Δt ≈ 4s for OpenSky, so ft/min = Δalt * 60/Δt
    # Since we don't know Δt precisely, use raw diff
    # (the Neural ODE will learn the scaling)
    if "altitude_ft" in df.columns and "vz_ftmin" not in df.columns:
        deriv_exprs.append(pl.col("altitude_ft").diff().fill_null(0.0).alias("vz_ftmin"))

    if "gamma_rad" in df.columns and "d_gamma_rad" not in df.columns:
        deriv_exprs.append(pl.col("gamma_rad").diff().fill_null(0.0).alias("d_gamma_rad"))

    if "tas_kt" in df.columns and "d_tas_kt" not in df.columns:
        deriv_exprs.append(pl.col("tas_kt").diff().fill_null(0.0).alias("d_tas_kt"))

    if deriv_exprs:
        df = df.with_columns(deriv_exprs)

    return df


def cumulative_distance(df: pl.DataFrame) -> pl.DataFrame:
    """Add cumulative along-track distance column.

    Computes haversine distance between consecutive points and
    accumulates into ``distance_along_track_m``.

    Args:
        df: Eager DataFrame with ``latitude`` and ``longitude`` columns.

    Returns:
        DataFrame with ``distance_along_track_m`` appended.
    """
    lat_col = "latitude"
    lon_col = "longitude"

    lat = df[lat_col].to_numpy()
    lon = df[lon_col].to_numpy()

    if len(lat) < 2:  # noqa: PLR2004
        return df.with_columns(pl.lit(0.0).alias("distance_along_track_m"))

    d = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cum_d = np.concatenate(([0.0], np.cumsum(d)))

    return df.with_columns(pl.Series("distance_along_track_m", cum_d))


def crop_on_distance_jump(
    df: pl.DataFrame,
    *,
    threshold: float = 200.0,
    min_speed: float = MIN_SPEED_KT,
    upper_threshold: float = 3000.0,
) -> pl.DataFrame:
    """Crop a flight DataFrame to remove distance discontinuities.

    Filters on minimum groundspeed, then trims the trajectory to the
    region between the first and last large distance jumps.

    Args:
        df: Eager DataFrame with ``distance_along_track_m`` and a
            speed column (``gs_kt`` or ``groundspeed``).
        threshold: Minimum jump (m) considered a discontinuity.
        min_speed: Minimum speed (kt) to retain data.
        upper_threshold: Ignore extremely large jumps above this.

    Returns:
        Cropped DataFrame, reset to contiguous integer index.
    """
    gs_col = "gs_kt" if "gs_kt" in df.columns else "groundspeed"

    # Filter low speed
    df = df.filter(pl.col(gs_col) > min_speed)

    if len(df) < 2:  # noqa: PLR2004
        return df

    # Compute distance diffs
    dist_diff = df["distance_along_track_m"].diff()

    # Find large jumps
    jumps = dist_diff.gt(threshold) & dist_diff.lt(upper_threshold)
    if jumps.any():
        indices = df.with_row_index().filter(jumps)["index"].to_list()
        first_jump = indices[0]
        last_jump = indices[-1]
        df = df.slice(first_jump, last_jump - first_jump + 1)

    return df


def segment_filtering(df: pl.DataFrame, start: int, seq_len: int) -> bool:
    """Check whether a segment meets distance-variation thresholds.

    A segment is *valid* when every consecutive distance difference
    falls within ``[LOW_THR, UPPER_THR]``.  Segments shorter than
    *seq_len* are rejected.

    Args:
        df: **Eager** DataFrame containing a ``distance_m`` column.
        start: Starting row index of the segment to evaluate.
        seq_len: Required length of the segment.

    Returns:
        ``True`` if the segment is valid, ``False`` otherwise.
    """
    dist = df["distance_m"]
    dist_diff = dist.diff()

    end = start + seq_len
    if end > len(dist_diff):
        return False

    seg = dist_diff.slice(start + 1, seq_len - 1)

    if seg.len() == 0:
        return False

    below = seg.lt(LOW_THR).sum()
    above = seg.gt(UPPER_THR).sum()

    return bool(below == 0 and above == 0)
