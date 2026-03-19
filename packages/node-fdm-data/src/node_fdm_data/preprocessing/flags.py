"""Validity flags for pipeline v3 — étape 2.

Each quality criterion becomes a boolean column ``fdm_flag_*``.
No rows are deleted; the consumer filters at read time via
``fdm_flag_valid``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_data.meteo import haversine

__all__ = [
    "compute_flags",
]


def _annotate_distance_flags(
    group: pl.DataFrame,
    low_thr: float,
    upper_thr: float,
) -> pl.DataFrame:
    """Add distance-based flag columns to a single flight group.

    Computes haversine distance-diff between consecutive points and
    derives ``fdm_flag_distance_ok``, ``fdm_flag_crop_start``, and
    ``fdm_flag_crop_end`` for the group.

    Args:
        group: Single-flight DataFrame (sorted by timestamp).
        low_thr: Minimum acceptable distance-diff (metres).
        upper_thr: Maximum acceptable distance-diff (metres).

    Returns:
        DataFrame with distance flag columns and ``_row_idx`` appended.
    """
    n = len(group)
    row_idx = np.arange(n, dtype=np.int64)

    if n < 2:  # noqa: PLR2004
        return group.with_columns(
            pl.lit(False).alias("fdm_flag_distance_ok"),
            pl.lit(0).cast(pl.Int64).alias("fdm_flag_crop_start"),
            pl.lit(0).cast(pl.Int64).alias("fdm_flag_crop_end"),
            pl.Series("_row_idx", row_idx),
        )

    lat = group["raw_lat_deg"].to_numpy()
    lon = group["raw_lon_deg"].to_numpy()

    d = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dist_diff = np.concatenate(([0.0], d))

    # fdm_flag_distance_ok: first row = True (no diff), rest = diff in [low, upper]
    distance_ok = np.empty(n, dtype=bool)
    distance_ok[0] = True
    distance_ok[1:] = (dist_diff[1:] >= low_thr) & (dist_diff[1:] <= upper_thr)

    # Crop indices: first/last row with "normal" distance jump
    jumps = (dist_diff > low_thr) & (dist_diff < upper_thr)
    jump_indices = np.where(jumps)[0]

    if len(jump_indices) > 0:
        crop_start = int(jump_indices[0])
        crop_end = int(jump_indices[-1])
    else:
        # No normal jumps → whole flight valid for crop
        crop_start = 0
        crop_end = n - 1

    return group.with_columns(
        pl.Series("fdm_flag_distance_ok", distance_ok),
        pl.lit(crop_start).cast(pl.Int64).alias("fdm_flag_crop_start"),
        pl.lit(crop_end).cast(pl.Int64).alias("fdm_flag_crop_end"),
        pl.Series("_row_idx", row_idx),
    )


def compute_flags(
    df: pl.DataFrame,
    *,
    min_points: int = 40,
    min_speed_kt: float = 90.0,
    distance_low_thr: float = 200.0,
    distance_upper_thr: float = 3000.0,
) -> pl.DataFrame:
    """Add all ``fdm_flag_*`` columns to a flight DataFrame.

    Implements pipeline v3 étape 2: each quality criterion becomes a
    boolean flag column.  No rows are removed.

    Args:
        df: DataFrame with ``raw_*`` and ``meta_flight_id`` columns
            (output of étape 1 — identify).
        min_points: Minimum points per segment for ``fdm_flag_min_points``.
        min_speed_kt: Minimum groundspeed (kt) for ``fdm_flag_min_speed``.
        distance_low_thr: Minimum distance-diff (m) for ``fdm_flag_distance_ok``.
        distance_upper_thr: Maximum distance-diff (m) for ``fdm_flag_distance_ok``.

    Returns:
        DataFrame with ``fdm_flag_*`` columns added.  Row count is unchanged.
    """
    df = df.sort("meta_flight_id", "raw_timestamp")

    # --- fdm_flag_min_points ---
    df = df.with_columns(
        (pl.len().over("meta_flight_id") >= min_points).alias("fdm_flag_min_points"),
    )

    # --- fdm_flag_min_speed ---
    df = df.with_columns(
        (pl.col("raw_gs_kt") > min_speed_kt).alias("fdm_flag_min_speed"),
    )

    # --- Distance-based flags (per-flight haversine) ---
    df = df.group_by("meta_flight_id", maintain_order=True).map_groups(
        lambda g: _annotate_distance_flags(g, distance_low_thr, distance_upper_thr),
    )

    # --- fdm_flag_valid = AND of all flags ---
    df = df.with_columns(
        (
            pl.col("fdm_flag_min_points")
            & pl.col("fdm_flag_min_speed")
            & pl.col("fdm_flag_distance_ok")
            & (pl.col("_row_idx") >= pl.col("fdm_flag_crop_start"))
            & (pl.col("_row_idx") <= pl.col("fdm_flag_crop_end"))
        ).alias("fdm_flag_valid"),
    )

    return df.drop("_row_idx")
