"""BDS + ERA5 airspeed merge — coalesce noisy-but-accurate BDS with complete ERA5.

BDS (Mode-S) provides ground-truth airspeed parameters but is noisy and
has significant gaps.  ERA5 wind-derived parameters are continuous but
less precise.  This module merges both sources using ``pl.coalesce``
(BDS preferred, ERA5 as gap-fill) to produce a complete signal ready
for EKF smoothing.
"""

from __future__ import annotations

import polars as pl

__all__ = ["merge_bds_era5"]

# (bds_column, era5_column, output_column)
_MERGE_PAIRS: list[tuple[str, str, str]] = [
    ("bds_tas_kt", "era_tas_kt", "ekf_input_tas_kt"),
    ("bds_ias_kt", "era_cas_kt", "ekf_input_cas_kt"),
    ("bds_mach", "era_mach", "ekf_input_mach"),
]


def merge_bds_era5(df: pl.DataFrame) -> pl.DataFrame:
    """Merge BDS and ERA5 airspeed columns via coalesce.

    For each parameter pair, the BDS value is preferred; ERA5 is used
    only when BDS is null.  If both are null the output is null.

    Columns produced: ``ekf_input_tas_kt``, ``ekf_input_cas_kt``,
    ``ekf_input_mach``.

    Args:
        df: DataFrame containing ``bds_*`` and ``era_*`` columns.

    Returns:
        DataFrame with ``ekf_input_*`` columns added.  Existing
        ``ekf_input_*`` columns are dropped before recomputation.
    """
    # Drop existing merge columns for idempotency
    existing = [c for c in df.columns if c.startswith("ekf_input_")]
    if existing:
        df = df.drop(existing)

    coalesce_exprs: list[pl.Expr] = []
    for bds_col, era_col, out_col in _MERGE_PAIRS:
        bds_present = bds_col in df.columns
        era_present = era_col in df.columns

        if bds_present and era_present:
            coalesce_exprs.append(pl.coalesce(pl.col(bds_col), pl.col(era_col)).alias(out_col))
        elif bds_present:
            coalesce_exprs.append(pl.col(bds_col).alias(out_col))
        elif era_present:
            coalesce_exprs.append(pl.col(era_col).alias(out_col))
        else:
            coalesce_exprs.append(pl.lit(None).cast(pl.Float64).alias(out_col))

    return df.with_columns(coalesce_exprs)
