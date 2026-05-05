"""Bench Hampel filter: Python loop (V0) vs Polars rolling (V1) vs scipy.

V0: current numpy Python-loop implementation.
V1: pl.col(x).rolling_median(2W+1, center=True) for med, then
    rolling_median of |x-med| for MAD, then mask.
V2: scipy.ndimage.median_filter (NaN-aware via fallback) — extra control point.

Tests on real bds_mach / bds_ias_kt / bds_tas_kt arrays from
data/flights.delta. Verifies bit-equivalence (or documents the divergence).
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from node_fdm_data.preprocessing.clean_speeds import _hampel_filter as v0_hampel

_HAMPEL_SCALE = 1.4826
_HAMPEL_MIN_WINDOW = 3


def v1_hampel_polars(x: np.ndarray, *, window: int, k: float) -> np.ndarray:
    """Polars rolling-median based Hampel.

    Note: Polars rolling_median treats NaN as a numeric value, not as
    missing. Convert NaN -> null first so the rolling median ignores them
    (matching V0's `w[~np.isnan(w)]` filter).
    """
    win = 2 * window + 1
    # NaN -> null so Polars rolling ignores them
    s = pl.Series("x", x, dtype=pl.Float64).fill_nan(None)
    med = s.rolling_median(window_size=win, min_samples=_HAMPEL_MIN_WINDOW, center=True)
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling_median(window_size=win, min_samples=_HAMPEL_MIN_WINDOW, center=True)

    x_arr = x  # original numpy with NaN intact (for output preservation)
    # Polars Series.to_numpy() converts null -> NaN automatically
    med_arr = med.to_numpy()
    mad_arr = mad.to_numpy()

    out = x_arr.copy()
    threshold = k * _HAMPEL_SCALE * mad_arr
    # Standard case: |x - med| > threshold AND mad > 0
    mask_std = (np.abs(x_arr - med_arr) > threshold) & (mad_arr > 0)
    # mad == 0 edge case: any value != med is an outlier
    mask_zero_mad = (mad_arr == 0) & (x_arr != med_arr)
    out[mask_std | mask_zero_mad] = np.nan
    return out


def load_real_signals(n_flights: int = 50):
    df = pl.read_delta("data/flights.delta")
    flights = df.partition_by("meta_flight_id", maintain_order=True)[:n_flights]
    signals = []
    for fdf in flights:
        for col in ("bds_mach", "bds_ias_kt", "bds_tas_kt"):
            if col in fdf.columns:
                arr = fdf[col].cast(pl.Float64).to_numpy()
                if np.sum(~np.isnan(arr)) >= _HAMPEL_MIN_WINDOW:
                    signals.append((col, arr))
    return signals


def main():
    signals = load_real_signals(50)
    print(f"Loaded {len(signals)} signal arrays from real flights")
    n_total = sum(len(arr) for _, arr in signals)
    print(f"Total points: {n_total:,}\n")

    window = 50
    k = 3.0
    n_passes = 3  # match production config

    # ---- V0 sequential (current code) ----
    t0 = time.perf_counter()
    v0_outputs = []
    for _col, arr in signals:
        out = arr.copy()
        for _ in range(n_passes):
            out = v0_hampel(out, window=window, k=k)
        v0_outputs.append(out)
    t_v0 = time.perf_counter() - t0
    print(f"V0 numpy loop (3 passes)             {t_v0*1000:9.1f} ms")

    # ---- V1 Polars rolling ----
    t0 = time.perf_counter()
    v1_outputs = []
    for _col, arr in signals:
        out = arr.copy()
        for _ in range(n_passes):
            out = v1_hampel_polars(out, window=window, k=k)
        v1_outputs.append(out)
    t_v1 = time.perf_counter() - t0
    print(f"V1 Polars rolling (3 passes)         {t_v1*1000:9.1f} ms  ({t_v0/t_v1:.1f}x)")
    print()

    # ---- Correctness ----
    diffs_count = []
    diffs_max = []
    for (col, _), v0, v1 in zip(signals, v0_outputs, v1_outputs, strict=True):
        # Compare NaN masks
        nan_v0 = np.isnan(v0)
        nan_v1 = np.isnan(v1)
        mask_disagree = nan_v0 != nan_v1
        diffs_count.append(int(mask_disagree.sum()))
        # Compare numeric values where both finite
        both_finite = (~nan_v0) & (~nan_v1)
        if both_finite.any():
            d = float(np.abs(v0[both_finite] - v1[both_finite]).max())
            diffs_max.append(d)

    total_disagree = sum(diffs_count)
    print(f"Correctness V0 vs V1:")
    print(f"  NaN-mask disagreements (over {n_total:,} points): {total_disagree:,} "
          f"({100*total_disagree/n_total:.3f}%)")
    if diffs_max:
        print(f"  max |v0 - v1| where both finite: {max(diffs_max):.3e}")
    # Per-signal breakdown of worst offenders
    sig_worst = sorted(zip(diffs_count, [s[0] for s in signals]),
                       reverse=True)[:5]
    print(f"  worst signals (disagreements): {sig_worst}")


if __name__ == "__main__":
    main()
