"""Train / val / test splitting by ICAO24 aircraft identifier.

Deterministically assigns each row to a split based on a hash of
``raw_icao24``.  All segments of the same aircraft always land in the
same split, preventing data leakage.

Example::

    result = split_by_icao(df, ratios=(0.7, 0.15, 0.15))
"""

from __future__ import annotations

import hashlib

import polars as pl

__all__ = [
    "split_by_icao",
]


def _hash_bucket(icao24: str, seed: int) -> float:
    """Return a deterministic float in [0, 1) for an icao24."""
    h = hashlib.sha256(f"{seed}:{icao24}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _assign_split(
    bucket: float,
    ratios: tuple[float, float, float],
) -> str:
    """Map a [0, 1) bucket to train/val/test."""
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "val"
    return "test"


def split_by_icao(
    df: pl.DataFrame,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> pl.DataFrame:
    """Assign each row a ``meta_split`` column based on ``raw_icao24`` hash.

    The split is deterministic: the same ``raw_icao24`` always maps to
    the same split regardless of the rest of the data.  This prevents
    data leakage across train/val/test.

    Args:
        df: DataFrame with at least a ``raw_icao24`` column.
        ratios: ``(train, val, test)`` proportions — must sum to ~1.0.
        seed: Hash salt for reproducible but adjustable splits.

    Returns:
        The input DataFrame with an added ``meta_split`` column.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(None).cast(pl.Utf8).alias("meta_split"))

    unique_icao24s = df["raw_icao24"].unique().to_list()
    mapping = {
        icao24: _assign_split(_hash_bucket(icao24, seed), ratios) for icao24 in unique_icao24s
    }

    splits = [mapping[ic] for ic in df["raw_icao24"].to_list()]
    return df.with_columns(pl.Series("meta_split", splits))
