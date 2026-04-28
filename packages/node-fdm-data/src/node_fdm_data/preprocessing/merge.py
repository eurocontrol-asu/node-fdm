"""BDS + ERA5 airspeed merge — coalesce noisy-but-accurate BDS with complete ERA5.

BDS (Mode-S) provides ground-truth airspeed parameters but is noisy and
has significant gaps.  ERA5 wind-derived parameters are continuous but
less precise.  This module merges both sources using ``pl.coalesce``
(cleaned BDS preferred, raw BDS next, ERA5 as gap-fill) to produce a
complete signal ready for EKF smoothing.
"""

from __future__ import annotations

import polars as pl

__all__ = ["merge_bds_era5"]

# (raw_bds_column, era_column, output_column)
# Preference order at runtime: bds_*_clean → bds_* → era_*.
_MERGE_PAIRS: list[tuple[str, str, str]] = [
    ("bds_tas_kt", "era_tas_kt", "ekf_input_tas_kt"),
    ("bds_ias_kt", "era_cas_kt", "ekf_input_cas_kt"),
    ("bds_mach", "era_mach", "ekf_input_mach"),
]


def merge_bds_era5(df: pl.DataFrame) -> pl.DataFrame:
    """Merge BDS and ERA5 airspeed columns via coalesce.

    Preference order per parameter:

    1. ``bds_*_clean`` (when present) — output of the ``clean-speeds`` stage.
    2. ``bds_*`` — raw Mode-S decode.
    3. ``era_*`` — ERA5 fallback.

    Columns produced: ``ekf_input_tas_kt``, ``ekf_input_cas_kt``,
    ``ekf_input_mach``.

    Args:
        df: DataFrame containing ``bds_*`` and ``era_*`` columns.

    Returns:
        DataFrame with ``ekf_input_*`` columns added.  Existing
        ``ekf_input_*`` columns are dropped before recomputation.
    """
    existing = [c for c in df.columns if c.startswith("ekf_input_")]
    if existing:
        df = df.drop(existing)

    coalesce_exprs: list[pl.Expr] = []
    for bds_col, era_col, out_col in _MERGE_PAIRS:
        clean_col = f"{bds_col}_clean"
        sources: list[pl.Expr] = []
        if clean_col in df.columns:
            sources.append(pl.col(clean_col))
        if bds_col in df.columns:
            sources.append(pl.col(bds_col))
        if era_col in df.columns:
            sources.append(pl.col(era_col))
        if not sources:
            coalesce_exprs.append(pl.lit(None).cast(pl.Float64).alias(out_col))
        elif len(sources) == 1:
            coalesce_exprs.append(sources[0].alias(out_col))
        else:
            coalesce_exprs.append(pl.coalesce(*sources).alias(out_col))

    return df.with_columns(coalesce_exprs)
