"""Bench the 3 sequential `for flight in flights:` loops in the FDM pipeline.

Step 1 — measure the absolute V0 cost of each call site on real data,
plus the partition_by + concat overhead. This identifies which call site
deserves parallelization effort (if any).

Step 2 (only on the dominant site) — compare V0 (sequential) vs
V1 (ThreadPoolExecutor) vs V2 (ProcessPoolExecutor), with a worker sweep.

Site 1: preprocess_flights (resample.py) — but preprocess is upstream,
so we'd need to re-run it on raw data. Skip site 1 in this bench: the
current Delta has already been preprocessed, so we don't have raw input
on hand. Document that gap.

Site 2: clean_bds_speeds per flight (clean-speeds step)
Site 3: build_selected_params per flight (segments step)
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import polars as pl

from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds
from node_fdm_data.segments import build_selected_params

# Realistic config matching what `fdm clean-speeds` uses (defaults from the
# CleanSpeedsConfig in the pipeline package). For benching, these don't need
# to be exact — what matters is that the function does its real work.
_CS_KW = {
    "bds_window": 50,
    "era_window": 50,
    "k": 3.0,
    "n_passes": 3,
    "interp_max_gap": 6,
    "frozen_min_run_len_mach": 20,
    "frozen_min_run_len_ias": 20,
    "frozen_min_run_len_tas": 6,
    "point_jump_max_mach": 0.05,
    "point_jump_max_kt": 30.0,
    "zigzag_jump_min_mach": 0.01,
    "zigzag_jump_min_kt": 5.0,
    "zigzag_half_window": 5,
    "zigzag_density_min_bds": 0.4,
    "zigzag_density_min_era": 0.6,
    "on_ground_vz_threshold": 64.0,
    "on_ground_alt_threshold": 100.0,
}

_SEL_KW: dict = {}  # build_selected_params accepts a config dict; defaults OK


def call_clean(flight_df: pl.DataFrame) -> pl.DataFrame:
    return clean_bds_speeds(flight_df, **_CS_KW)


def call_segments(flight_df: pl.DataFrame) -> pl.DataFrame:
    return build_selected_params(flight_df, _SEL_KW)


def time_block(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:40s} {dt*1000:9.1f} ms")
    return out, dt


def bench_v0_sequential(flights, fn) -> tuple[list[pl.DataFrame], float]:
    t0 = time.perf_counter()
    out = [fn(f) for f in flights]
    return out, time.perf_counter() - t0


def bench_v1_thread(flights, fn, n_workers) -> tuple[list[pl.DataFrame], float]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        out = list(ex.map(fn, flights))
    return out, time.perf_counter() - t0


# Process pool needs picklable callables and may have spawn-time overhead.
def _proc_init_polars_1():
    """Worker initializer: cap Polars internal threads to 1 to avoid
    oversubscription when running inside a ProcessPoolExecutor."""
    import os
    os.environ["POLARS_MAX_THREADS"] = "1"


def _proc_clean(flight_df):
    from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds as _cs
    return _cs(flight_df, **_CS_KW)


def _proc_segments(flight_df):
    from node_fdm_data.segments import build_selected_params as _bsp
    return _bsp(flight_df, _SEL_KW)


def bench_v2_process(flights, proc_fn, n_workers, *, polars_threads=None):
    t0 = time.perf_counter()
    init = _proc_init_polars_1 if polars_threads == 1 else None
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init) as ex:
        out = list(ex.map(proc_fn, flights))
    return out, time.perf_counter() - t0


def correctness_check(out_v0, out_vx, label):
    if len(out_v0) != len(out_vx):
        print(f"    {label}: LEN MISMATCH {len(out_v0)} vs {len(out_vx)}")
        return
    diffs = []
    for a, b in zip(out_v0, out_vx, strict=True):
        if a.shape != b.shape:
            diffs.append(f"shape {a.shape} vs {b.shape}")
            continue
        if a.columns != b.columns:
            diffs.append("columns differ")
            continue
        # Sample numeric diff on a representative cleaned column when present
        for col in ("bds_mach_clean", "bds_ias_kt_clean", "fdm_mach_sel",
                    "fdm_cas_sel_kt", "fdm_alt_sel_ft"):
            if col in a.columns and col in b.columns:
                av = a[col].to_numpy()
                bv = b[col].to_numpy()
                import numpy as np
                m = np.isfinite(av) & np.isfinite(bv)
                if m.any():
                    md = float(np.abs(av[m] - bv[m]).max())
                    if md > 1e-9:
                        diffs.append(f"{col} maxdiff={md:.3e}")
    print(f"    {label}: {'OK (bit-equiv)' if not diffs else 'DIVERGE: ' + '; '.join(diffs[:3])}")


def main():
    df = pl.read_delta("data/flights.delta")
    n_total_rows = len(df)
    n_flights_total = df["meta_flight_id"].n_unique()
    print(f"Loaded {n_total_rows:,} rows, {n_flights_total} flights")
    print(f"CPU count: {os.cpu_count()}\n")

    # =============================================================
    # PHASE A — measure V0 (sequential) cost on each call site.
    # =============================================================
    n_sample = 100  # subsample of flights to keep bench fast yet meaningful
    flights_all = df.partition_by("meta_flight_id", maintain_order=True)
    flights = flights_all[:n_sample]
    n_rows = sum(len(f) for f in flights)
    print(f"=== PHASE A — V0 cost on {n_sample} flights ({n_rows:,} rows) ===\n")

    # Partition + concat overhead alone (already paid by partition above)
    t0 = time.perf_counter()
    _ = pl.concat(flights, how="diagonal_relaxed")
    dt_concat = time.perf_counter() - t0
    print(f"  partition_by (already done above)       — n/a")
    print(f"  pl.concat (no work)                     {dt_concat*1000:9.1f} ms")
    print()

    # Site 2: clean_bds_speeds
    print("Site 2 — clean_bds_speeds:")
    out_v0_clean, t_v0_clean = bench_v0_sequential(flights, call_clean)
    print(f"  V0 sequential                            {t_v0_clean*1000:9.1f} ms")
    print(f"  per-flight avg                           {t_v0_clean*1000/n_sample:9.2f} ms")
    print()

    # Site 3: segments
    print("Site 3 — build_selected_params:")
    # build_selected_params requires the cleaned columns. Use V0 clean output as input.
    out_v0_seg, t_v0_seg = bench_v0_sequential(out_v0_clean, call_segments)
    print(f"  V0 sequential                            {t_v0_seg*1000:9.1f} ms")
    print(f"  per-flight avg                           {t_v0_seg*1000/n_sample:9.2f} ms")
    print()

    # Pick dominant site
    dominant = "clean" if t_v0_clean > t_v0_seg else "segments"
    print(f">> Dominant site: {dominant} (clean={t_v0_clean*1000:.0f}ms, "
          f"seg={t_v0_seg*1000:.0f}ms)\n")

    # =============================================================
    # PHASE B — variants on dominant site, with worker sweep.
    # =============================================================
    if dominant == "clean":
        v0_out, v0_t = out_v0_clean, t_v0_clean
        thread_call = call_clean
        proc_call = _proc_clean
        site_flights = flights
    else:
        v0_out, v0_t = out_v0_seg, t_v0_seg
        thread_call = call_segments
        proc_call = _proc_segments
        site_flights = out_v0_clean  # input must have *_clean columns

    print(f"=== PHASE B — variants on '{dominant}' site ===\n")

    n_cores = os.cpu_count() or 8
    sweep = sorted({2, 4, 8, n_cores, max(2, n_cores // 2)})
    sweep = [w for w in sweep if w <= n_cores]

    print(f"V0 sequential                              {v0_t*1000:9.1f} ms  (1.0x)")
    print()
    print("V1 ThreadPoolExecutor:")
    for w in sweep:
        out, t = bench_v1_thread(site_flights, thread_call, w)
        sp = v0_t / t if t > 0 else 0
        print(f"  workers={w:<3d}                            {t*1000:9.1f} ms  ({sp:.1f}x)")
        if w == sweep[len(sweep) // 2]:
            correctness_check(v0_out, out, f"V1 (workers={w})")
    print()

    print("V2 ProcessPoolExecutor (default Polars threads):")
    for w in sweep:
        out, t = bench_v2_process(site_flights, proc_call, w)
        sp = v0_t / t if t > 0 else 0
        print(f"  workers={w:<3d}                            {t*1000:9.1f} ms  ({sp:.1f}x)")
        if w == sweep[len(sweep) // 2]:
            correctness_check(v0_out, out, f"V2 (workers={w})")
    print()

    print("V3 ProcessPoolExecutor (POLARS_MAX_THREADS=1 per worker):")
    for w in sweep:
        out, t = bench_v2_process(site_flights, proc_call, w, polars_threads=1)
        sp = v0_t / t if t > 0 else 0
        print(f"  workers={w:<3d}                            {t*1000:9.1f} ms  ({sp:.1f}x)")
        if w == sweep[len(sweep) // 2]:
            correctness_check(v0_out, out, f"V3 (workers={w})")
    print()


if __name__ == "__main__":
    main()
