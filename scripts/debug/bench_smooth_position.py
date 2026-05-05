"""Bench smooth_position_subsegments: split timing between
Polars↔pandas conversions, the actual traffic.Flight.filter work,
and the Python list assignment scaffold.

Also benchmarks 3 variants of the extract_pairs hotspot:
  V0  current code         : iloc[i][col] Python loop
  V1  to_numpy + zip loop  : single .to_numpy(), then python assign loop
  V2  full vector assign   : np arrays + fancy-index assign

Mirrors the real pipeline call on real flights from data/flights.delta.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import polars as pl

from node_fdm_data.preprocessing.resample import _MIN_RUN_LEN

try:
    from traffic.core import Flight
except ImportError as e:
    raise SystemExit(f"traffic library required: {e}") from None


def time_block(label, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  {label:38s} {dt*1000:9.1f} ms   peak={peak/1e6:7.2f} MB")
    return out, dt


def variant_extract(smoothed_data, indices, lats, lons, variant):
    """Run one of three extraction strategies. Mutates lats/lons in place
    (or returns new arrays). Returns the mutated containers."""
    if variant == "V0_iloc_loop":
        smooth_pd = smoothed_data[["latitude", "longitude"]]
        pairs = [
            (float(smooth_pd.iloc[i]["latitude"]), float(smooth_pd.iloc[i]["longitude"]))
            for i in range(len(indices))
        ]
        for idx, (lat, lon) in zip(indices, pairs, strict=True):
            lats[idx] = lat
            lons[idx] = lon
        return lats, lons
    elif variant == "V1_to_numpy_zip":
        arr = smoothed_data[["latitude", "longitude"]].to_numpy()
        for idx, (lat, lon) in zip(indices, arr, strict=True):
            lats[idx] = float(lat)
            lons[idx] = float(lon)
        return lats, lons
    elif variant == "V2_fancy_index":
        # lats/lons are np arrays here
        arr = smoothed_data[["latitude", "longitude"]].to_numpy()
        idx_arr = np.asarray(indices, dtype=np.int64)
        lats[idx_arr] = arr[:, 0]
        lons[idx_arr] = arr[:, 1]
        return lats, lons
    raise ValueError(variant)


def bench_extract_only(flights, variant):
    """Run smoothing on each flight; only the extract+assign step uses *variant*.
    Filter() result is reused (cached) across variants for fair comparison."""
    # Pre-compute filter outputs once (cache) so we only time extract+assign.
    cache: list[tuple[pl.DataFrame, list]] = []
    for fdf in flights:
        has_pos = fdf.select(
            pl.col("raw_lat_deg").is_not_null() & pl.col("raw_lon_deg").is_not_null(),
        ).to_series()
        if not has_pos.any():
            cache.append((fdf, []))
            continue
        run_ids = (has_pos != has_pos.shift()).cum_sum()
        per_run = []
        for run_id in run_ids.filter(has_pos).unique().drop_nulls().sort().to_list():
            mask = (run_ids == run_id) & has_pos
            indices = mask.arg_true().to_list()
            if len(indices) < _MIN_RUN_LEN:
                continue
            select_cols = [
                pl.col("raw_timestamp").alias("timestamp"),
                pl.col("raw_lat_deg").alias("latitude"),
                pl.col("raw_lon_deg").alias("longitude"),
            ]
            if "raw_alt_ft" in fdf.columns:
                select_cols.append(pl.col("raw_alt_ft").alias("altitude"))
            segment = fdf.filter(mask).select(select_cols)
            smoothed = Flight(segment.to_pandas()).filter("aggressive")
            if smoothed is None or len(smoothed) != len(indices):
                continue
            per_run.append((indices, smoothed.data))
        cache.append((fdf, per_run))

    # Now time only the extract+assign for the chosen variant.
    t0 = time.perf_counter()
    for fdf, per_run in cache:
        if variant == "V2_fancy_index":
            lats = fdf["raw_lat_deg"].to_numpy().copy()
            lons = fdf["raw_lon_deg"].to_numpy().copy()
        else:
            lats = fdf["raw_lat_deg"].to_list()
            lons = fdf["raw_lon_deg"].to_list()
        for indices, sdata in per_run:
            variant_extract(sdata, indices, lats, lons, variant)
        # Mimic the with_columns rebuild
        _ = fdf.with_columns(
            pl.Series("raw_lat_deg", lats, dtype=pl.Float64),
            pl.Series("raw_lon_deg", lons, dtype=pl.Float64),
        )
    return time.perf_counter() - t0, cache


def smooth_one_flight_instrumented(df: pl.DataFrame, accumulators):
    """Run the same logic as smooth_position_subsegments, but record
    cumulative timings into accumulators dict."""
    has_pos = df.select(
        pl.col("raw_lat_deg").is_not_null() & pl.col("raw_lon_deg").is_not_null(),
    ).to_series()
    if not has_pos.any():
        return df

    run_ids = (has_pos != has_pos.shift()).cum_sum()

    t0 = time.perf_counter()
    lats = df["raw_lat_deg"].to_list()
    lons = df["raw_lon_deg"].to_list()
    accumulators["to_list"] += time.perf_counter() - t0

    for run_id in run_ids.filter(has_pos).unique().drop_nulls().sort().to_list():
        mask = (run_ids == run_id) & has_pos
        indices = mask.arg_true().to_list()
        if len(indices) < _MIN_RUN_LEN:
            continue

        select_cols = [
            pl.col("raw_timestamp").alias("timestamp"),
            pl.col("raw_lat_deg").alias("latitude"),
            pl.col("raw_lon_deg").alias("longitude"),
        ]
        if "raw_alt_ft" in df.columns:
            select_cols.append(pl.col("raw_alt_ft").alias("altitude"))

        segment = df.filter(mask).select(select_cols)

        t0 = time.perf_counter()
        seg_pd = segment.to_pandas()
        accumulators["to_pandas"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        smoothed = Flight(seg_pd).filter("aggressive")
        accumulators["filter"] += time.perf_counter() - t0

        if smoothed is None or len(smoothed) != len(indices):
            continue

        t0 = time.perf_counter()
        smooth_pd = smoothed.data[["latitude", "longitude"]]
        pairs = [
            (float(smooth_pd.iloc[i]["latitude"]), float(smooth_pd.iloc[i]["longitude"]))
            for i in range(len(indices))
        ]
        accumulators["extract_pairs"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for idx, (lat, lon) in zip(indices, pairs, strict=True):
            lats[idx] = lat
            lons[idx] = lon
        accumulators["assign_loop"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    out = df.with_columns(
        pl.Series("raw_lat_deg", lats, dtype=pl.Float64),
        pl.Series("raw_lon_deg", lons, dtype=pl.Float64),
    )
    accumulators["rebuild"] += time.perf_counter() - t0
    return out


def main():
    df = pl.read_delta("data/flights.delta")
    print(f"Loaded {len(df):,} rows, {df['meta_flight_id'].n_unique()} flights")

    flights = df.partition_by("meta_flight_id", maintain_order=True)
    n_flights = min(50, len(flights))
    flights = flights[:n_flights]
    n_rows = sum(len(f) for f in flights)
    print(f"Benching on {n_flights} flights = {n_rows:,} rows\n")

    acc = dict.fromkeys(
        ["to_list", "to_pandas", "filter", "extract_pairs", "assign_loop", "rebuild"], 0.0
    )
    tracemalloc.start()
    t0 = time.perf_counter()
    for fdf in flights:
        smooth_one_flight_instrumented(fdf, acc)
    total_t = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=== PER-PHASE TOTALS (sum across flights) ===")
    for k, v in acc.items():
        print(f"  {k:20s} {v*1000:9.1f} ms  ({100*v/total_t:5.1f}%)")
    print(f"  {'TOTAL':20s} {total_t*1000:9.1f} ms  peak={peak/1e6:.1f} MB")
    print(f"\n  per-flight avg : {total_t*1000/n_flights:.1f} ms")
    print(f"  per-row avg    : {total_t*1e6/n_rows:.2f} µs")

    # ---- Variant comparison: extract+assign+rebuild only ----
    print("\n=== EXTRACT+ASSIGN VARIANTS (filter cached) ===")
    results = {}
    for variant in ("V0_iloc_loop", "V1_to_numpy_zip", "V2_fancy_index"):
        dt, _cache = bench_extract_only(flights, variant)
        results[variant] = dt
        print(f"  {variant:20s} {dt*1000:9.1f} ms")
    base = results["V0_iloc_loop"]
    print(
        f"\n  V1 speedup vs V0  : {base/results['V1_to_numpy_zip']:.1f}x"
        f"   V2 speedup vs V0  : {base/results['V2_fancy_index']:.1f}x"
    )


if __name__ == "__main__":
    main()
