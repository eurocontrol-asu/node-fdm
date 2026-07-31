"""OpenSky 2025 preprocessing pipeline.

Computes derived columns (altitude difference, flight-path angle,
longitudinal wind, cumulative distance) and fills missing control
inputs.
"""

from __future__ import annotations

import polars as pl
from node_fdm_data.meteo import haversine_expr
from node_fdm_data.physics.constants import FTMIN, KT

__all__ = [
    "cumulative_distance",
    "flight_processing",
]


_TRAFFIC_RENAME: dict[str, str] = {
    "altitude": "raw_alt_ft",
    "selected_mcp": "bds_mcp_alt_sel_ft",
    "vertical_rate": "raw_vz_ftmin",
    "Mach": "era_mach",
    "IAS": "bds_ias_kt",
    "TAS": "era_tas_kt",
    "groundspeed": "raw_gs_kt",
}

_FILL_NULL_COLS: dict[str, float] = {
    "raw_vz_ftmin": 0.0,
    "era_mach": 0.0,
    "bds_ias_kt": 0.0,
}


def _apply_traffic_rename(df: pl.LazyFrame) -> pl.LazyFrame:
    schema = df.collect_schema()
    rename = {k: v for k, v in _TRAFFIC_RENAME.items() if k in schema and v not in schema}
    return df.rename(rename) if rename else df


def _gamma_expr(tas_col: str, vz_col: str) -> pl.Expr:
    vz_ms = pl.col(vz_col) * FTMIN
    tas_ms = pl.col(tas_col) * KT
    ratio = (vz_ms / tas_ms.clip(lower_bound=1e-6)).clip(-1.0, 1.0)
    return ratio.arcsin().alias("fdm_gamma_rad")


def _resolve_speed_cols(schema: pl.Schema) -> tuple[str, str, str]:
    tas_col = "era_tas_kt" if "era_tas_kt" in schema else "TAS"
    vz_col = "raw_vz_ftmin" if "raw_vz_ftmin" in schema else "vertical_rate"
    gs_col = "raw_gs_kt" if "raw_gs_kt" in schema else "groundspeed"
    return tas_col, vz_col, gs_col


def _alt_diff_expr(schema: pl.Schema) -> list[pl.Expr]:
    if "bds_mcp_alt_sel_ft" in schema and "raw_alt_ft" in schema:
        return [(pl.col("bds_mcp_alt_sel_ft") - pl.col("raw_alt_ft")).alias("fdm_alt_diff_ft")]
    return []


def _speed_exprs(schema: pl.Schema, tas_col: str, vz_col: str, gs_col: str) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    if tas_col in schema and vz_col in schema:
        exprs.append(_gamma_expr(tas_col, vz_col))
    if tas_col in schema and gs_col in schema:
        exprs.append((pl.col(tas_col) - pl.col(gs_col)).alias("fdm_long_wind_kt"))
    return exprs


def _fill_null_exprs(schema: pl.Schema) -> list[pl.Expr]:
    return [pl.col(col).fill_null(val) for col, val in _FILL_NULL_COLS.items() if col in schema]


def _derived_exprs(schema: pl.Schema) -> list[pl.Expr]:
    exprs: list[pl.Expr] = []
    exprs.extend(_alt_diff_expr(schema))
    cols = _resolve_speed_cols(schema)
    exprs.extend(_speed_exprs(schema, *cols))
    exprs.extend(_fill_null_exprs(schema))
    return exprs


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
        LazyFrame with normalised names and derived columns.
    """
    if "timestamp" in df.collect_schema():
        df = df.sort("timestamp")

    df = _apply_traffic_rename(df)
    exprs = _derived_exprs(df.collect_schema())
    return df.with_columns(exprs) if exprs else df


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

    # Defensive guard: drop null-coord rows to prevent NaN in haversine (AXM-511)
    df = df.filter(pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null())

    if len(df) < 2:
        return df.with_columns(pl.lit(0.0).alias("distance_along_track_m"))

    df = df.with_columns(
        pl.col(lat_col).shift(1).alias("_prev_lat"),
        pl.col(lon_col).shift(1).alias("_prev_lon"),
    )
    df = df.with_columns(
        haversine_expr("_prev_lat", "_prev_lon", lat_col, lon_col)
        .fill_null(0.0)
        .cum_sum()
        .alias("distance_along_track_m"),
    )
    return df.drop("_prev_lat", "_prev_lon")
