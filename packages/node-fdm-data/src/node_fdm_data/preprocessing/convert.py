"""SI conversion and temporal derivatives (pipeline v3, étapes 6-7).

Converts aeronautical-unit columns to SI and computes finite-difference
derivatives grouped by flight.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from node_fdm_data.conversions import (
    ft_to_m,
    ftmin_to_ms,
    kt_to_ms,
    nm_to_m,
)

__all__ = [
    "DERIVATIVE_BOUNDS",
    "SI_CONVERSIONS",
    "SI_DERIVATIVES",
    "compute_derivatives",
    "convert_si",
]

# ---------------------------------------------------------------------------
# SI conversion table: (source_col, conversion_fn, target_col)
# Matches pipeline v3 column names from étapes 0-5.
# ---------------------------------------------------------------------------
SI_CONVERSIONS: list[tuple[str, Callable[[str], pl.Expr], str]] = [
    ("raw_alt_ft", ft_to_m, "raw_alt_m"),
    ("bds_mcp_sel_alt_ft", ft_to_m, "bds_mcp_sel_alt_m"),
    ("era_tas_kt", kt_to_ms, "era_tas_ms"),
    ("bds_ias_kt", kt_to_ms, "bds_ias_ms"),
    ("raw_gs_kt", kt_to_ms, "raw_gs_ms"),
    ("fdm_long_wind_kt", kt_to_ms, "fdm_long_wind_ms"),
    ("raw_vz_ftmin", ftmin_to_ms, "raw_vz_ms"),
    ("fdm_adep_dist_nm", nm_to_m, "fdm_adep_dist_m"),
    ("fdm_ades_dist_nm", nm_to_m, "fdm_ades_dist_m"),
    ("fdm_alt_target_ft", ft_to_m, "fdm_alt_target_m"),
    ("fdm_cas_sel_kt", kt_to_ms, "fdm_cas_sel_ms"),
    ("fdm_tas_target_kt", kt_to_ms, "fdm_tas_target_ms"),
    ("fdm_vz_sel_ftmin", ftmin_to_ms, "fdm_vz_sel_ms"),
]

# Derivative table: (source_si_col, target_deriv_col)
SI_DERIVATIVES: list[tuple[str, str]] = [
    ("raw_alt_m", "fdm_d_vz_ms"),
    ("fdm_gamma_rad", "fdm_d_gamma_rads"),
    ("era_tas_ms", "fdm_d_tas_ms"),
]

DERIVATIVE_BOUNDS: dict[str, tuple[float, float]] = {
    "fdm_d_vz_ms": (-75.0, 75.0),
    "fdm_d_gamma_rads": (-0.025, 0.025),
    "fdm_d_tas_ms": (-12.5, 12.5),
}


def convert_si(df: pl.DataFrame) -> pl.DataFrame:
    """Add SI-unit columns to the DataFrame (étape 6).

    For each entry in :data:`SI_CONVERSIONS` whose source column exists,
    a new target column is added.  Source columns are preserved.

    Args:
        df: DataFrame with aeronautical-unit columns from étapes 0-5.

    Returns:
        DataFrame with SI columns appended.
    """
    cols = set(df.columns)
    exprs = [fn(src).alias(tgt) for src, fn, tgt in SI_CONVERSIONS if src in cols]
    if exprs:
        df = df.with_columns(exprs)
    return df


def compute_derivatives(
    df: pl.DataFrame,
    *,
    flight_id_col: str = "meta_flight_id",
    dt: float = 4.0,
) -> pl.DataFrame:
    """Add temporal derivative columns grouped by flight (étape 7).

    For each entry in :data:`SI_DERIVATIVES` whose source column exists,
    computes ``(diff() / dt).backward_fill().fill_null(0.0)`` within each
    *flight_id_col* group, then clips to :data:`DERIVATIVE_BOUNDS`.

    Args:
        df: DataFrame with SI columns from :func:`convert_si`.
        flight_id_col: Column used to partition flights.
        dt: Time step in seconds between consecutive rows.

    Returns:
        DataFrame with ``fdm_d_*`` derivative columns appended.
    """
    present = set(df.columns)
    entries = [(src, tgt) for src, tgt in SI_DERIVATIVES if src in present]
    if not entries:
        return df

    deriv_exprs = []
    for src, tgt in entries:
        expr = (pl.col(src).diff() / dt).backward_fill().fill_null(0.0)
        if tgt in DERIVATIVE_BOUNDS:
            lo, hi = DERIVATIVE_BOUNDS[tgt]
            expr = expr.clip(lo, hi)
        deriv_exprs.append(expr.alias(tgt))

    df = df.with_columns(
        *[expr.over(flight_id_col) for expr in deriv_exprs],
    )
    return df
