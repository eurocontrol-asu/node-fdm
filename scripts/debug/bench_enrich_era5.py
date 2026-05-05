"""Bench enrich_era5: split timing between Polars↔pandas conversions
and the actual fastmeteo interpolation work.

Loads a representative chunk from data/flights.delta and times each
sub-step of enrich_era5:
  1) df.rename + tz-strip (pure Polars)
  2) df.to_pandas()
  3) arco_grid.interpolate(pd_df)  -- the real work (xarray + ERA5 cache I/O)
  4) pl.from_pandas(pd_df)
  5) downstream Polars compute_tas / mach / cas
"""

from __future__ import annotations

import time
import tracemalloc

import polars as pl

from node_fdm_data.meteo import compute_cas_expr, compute_mach_expr, compute_tas

_FASTMETEO_INPUT_RENAME = {
    "raw_lat_deg": "latitude",
    "raw_lon_deg": "longitude",
    "raw_alt_ft": "altitude",
    "raw_timestamp": "timestamp",
}
_FASTMETEO_INPUT_RESTORE = {v: k for k, v in _FASTMETEO_INPUT_RENAME.items()}
_ERA5_RENAME = {
    "temperature": "era_temp_K",
    "u_component_of_wind": "era_u_wind_ms",
    "v_component_of_wind": "era_v_wind_ms",
}


def make_arco_grid():
    from fastmeteo.source.arco_era5 import ArcoEra5

    return ArcoEra5(
        local_store="data/era5_cache",
        features=["temperature", "u_component_of_wind", "v_component_of_wind"],
    )


def step_polars_prep(df: pl.DataFrame) -> pl.DataFrame:
    df_fm = df.rename(_FASTMETEO_INPUT_RENAME)
    ts_dtype = df_fm.schema["timestamp"]
    if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is not None:
        df_fm = df_fm.with_columns(pl.col("timestamp").dt.replace_time_zone(None))
    return df_fm


def step_downstream(df_fm: pl.DataFrame) -> pl.DataFrame:
    df_fm = df_fm.rename(_FASTMETEO_INPUT_RESTORE).rename(_ERA5_RENAME)
    df_fm = df_fm.with_columns(
        compute_tas(
            gs_col="raw_gs_kt",
            track_col="raw_track_deg",
            u_wind_col="era_u_wind_ms",
            v_wind_col="era_v_wind_ms",
        ).alias("era_tas_kt"),
    )
    return df_fm.with_columns(
        compute_mach_expr().alias("era_mach"),
        compute_cas_expr().alias("era_cas_kt"),
    )


def time_block(label, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  {label:35s} {dt*1000:8.1f} ms   peak={peak/1e6:7.1f} MB")
    return out, dt


def main():
    df = pl.read_delta("data/flights.delta")
    print(f"Loaded {len(df):,} rows")

    needed = ["raw_lat_deg", "raw_lon_deg", "raw_alt_ft", "raw_timestamp",
              "raw_gs_kt", "raw_track_deg"]
    df = df.select(needed).drop_nulls(["raw_lat_deg", "raw_lon_deg",
                                        "raw_alt_ft", "raw_timestamp"])

    # Drop existing era_* if present, to mirror real call.
    for size in (10_000, 100_000, len(df)):
        sub = df.head(size)
        print(f"\n=== {size:,} rows ===")

        arco_grid = make_arco_grid()

        out, t1 = time_block("1) rename + tz-strip (pl)",
                              lambda: step_polars_prep(sub))
        out2, t2 = time_block("2) to_pandas()",
                               lambda: out.to_pandas())
        out3, t3 = time_block("3) arco_grid.interpolate (real)",
                               lambda: arco_grid.interpolate(out2))
        out4, t4 = time_block("4) pl.from_pandas()",
                               lambda: pl.from_pandas(out3))
        _, t5 = time_block("5) downstream compute (pl)",
                            lambda: step_downstream(out4))

        total = t1 + t2 + t3 + t4 + t5
        print(f"  {'TOTAL':35s} {total*1000:8.1f} ms")
        conv = t2 + t4
        print(f"  conversions (2+4)  / total: {conv*1000:7.1f} ms "
              f"({100*conv/total:.1f}%)")
        print(f"  fastmeteo work (3) / total: {t3*1000:7.1f} ms "
              f"({100*t3/total:.1f}%)")


if __name__ == "__main__":
    main()
