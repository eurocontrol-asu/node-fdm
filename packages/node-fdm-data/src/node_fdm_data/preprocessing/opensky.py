"""OpenSky 2025 preprocessing pipeline.

Computes derived columns (altitude difference) and fills missing control
inputs.  Also provides segment filtering for training data extraction.
"""

from __future__ import annotations

import polars as pl

__all__ = [
    "LOW_THR",
    "UPPER_THR",
    "flight_processing",
    "segment_filtering",
]

LOW_THR: int = 200
"""Minimum acceptable distance-diff per timestep (metres)."""

UPPER_THR: int = 3000
"""Maximum acceptable distance-diff per timestep (metres)."""


def flight_processing(df: pl.LazyFrame) -> pl.LazyFrame:
    """Prepare OpenSky flight data for model training.

    * Computes ``alt_diff_ft = alt_sel_ft - altitude_ft``
    * Fills null values in control columns with ``0.0``

    Args:
        df: LazyFrame containing at least ``altitude_ft``, ``alt_sel_ft``,
            ``vz_sel_ftmin``, ``mach_sel``, ``cas_sel_kt``.

    Returns:
        LazyFrame with derived columns added and nulls filled.
    """
    return df.with_columns(
        (pl.col("alt_sel_ft") - pl.col("altitude_ft")).alias("alt_diff_ft"),
        pl.col("vz_sel_ftmin").fill_null(0.0),
        pl.col("mach_sel").fill_null(0.0),
        pl.col("cas_sel_kt").fill_null(0.0),
    )


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
