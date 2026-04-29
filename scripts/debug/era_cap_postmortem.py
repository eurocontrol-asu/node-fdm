"""What did the ERA-deviation cap actually catch that Hampel did not?

Replay both pipelines on the full delta:

A. current = frozen + Hampel + interp                  (no cap)
B. with-cap = frozen + Hampel + cap(0.025/10) + interp  (old behaviour)

For each BDS column (mach, ias), find points where:
  - A keeps the value (clean is finite)
  - B flags it as NaN
i.e. points the cap killed and Hampel did not.

For those points, characterise:
  - what fraction were already noise the user wanted gone?
  - what fraction were legitimate climb/descent (cap false positives)?
  - was the BDS raw stable (cap killed a steady reading) or jumping (cap did Hampel's job)?

Heuristics for "noise vs legit":
  - if raw value is stable in a +/- 5s window AND era is also stable AND
    they disagree by >cap → suspect: either real bias (BDS wrong) or transient.
  - if raw is jumping in window → Hampel-style outlier.
  - look at altitude derivative |dh/dt|: large => transitoire (cap likely
    false positive); small => cruise (cap likely real catch).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_data.preprocessing.clean_speeds import _flag_frozen_runs, _hampel_filter, _interpolate_short_gaps


def _clean_no_cap(values, *, window=7, k=3.0, n_passes=3, interp_max_gap=10, frozen=20):
    cleaned = values.astype(np.float64, copy=True)
    cleaned = _flag_frozen_runs(cleaned, min_run_len=frozen)
    for _ in range(n_passes):
        cleaned = _hampel_filter(cleaned, window=window, k=k)
    cleaned = _interpolate_short_gaps(cleaned, max_gap=interp_max_gap)
    return cleaned


def _clean_with_cap(values, era, *, cap, window=7, k=3.0, n_passes=3, interp_max_gap=10, frozen=20):
    cleaned = values.astype(np.float64, copy=True)
    cleaned = _flag_frozen_runs(cleaned, min_run_len=frozen)
    for _ in range(n_passes):
        cleaned = _hampel_filter(cleaned, window=window, k=k)
    valid = ~np.isnan(cleaned) & ~np.isnan(era)
    dev = valid & (np.abs(cleaned - era) > cap)
    cleaned[dev] = np.nan
    cleaned = _interpolate_short_gaps(cleaned, max_gap=interp_max_gap)
    return cleaned


def _local_std(x, half=2):
    """Std over centered window (excluding NaN)."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = x[lo:hi]
        w = w[~np.isnan(w)]
        if w.size >= 3:
            out[i] = float(np.std(w))
    return out


df = pl.read_delta("data/flights.delta")

# Limit to flights with reasonable size to keep it fast but representative.
counts = df.group_by("meta_flight_id").len().sort("len", descending=True)
flights = counts["meta_flight_id"].to_list()[:50]  # top 50 flights
df = df.filter(pl.col("meta_flight_id").is_in(flights))
print(f"flights analysed: {len(flights)}, rows: {df.height}")

results = []
for col, era_col, cap, frozen in [
    ("bds_mach", "era_mach", 0.025, 20),
    ("bds_ias_kt", "era_cas_kt", 10.0, 20),
]:
    cap_killed_total = 0
    n_in_climb = 0
    n_in_cruise = 0
    n_raw_stable = 0
    n_raw_jumpy = 0
    deltas_climb = []
    deltas_cruise = []

    for fid in flights:
        f = df.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
        raw = f[col].cast(pl.Float64).to_numpy()
        era = f[era_col].cast(pl.Float64).to_numpy()
        alt = f["raw_alt_ft"].cast(pl.Float64).to_numpy() if "raw_alt_ft" in f.columns else np.full(len(raw), np.nan)

        a = _clean_no_cap(raw, frozen=frozen)
        b = _clean_with_cap(raw, era, cap=cap, frozen=frozen)

        # Cap-only kills: A finite, B nan
        cap_only = ~np.isnan(a) & np.isnan(b)
        cap_killed_total += int(cap_only.sum())
        if cap_only.sum() == 0:
            continue

        # Altitude rate (ft/s); ts spacing ~4s in this dataset
        dh = np.gradient(alt)
        # absolute climb rate threshold
        climb_mask = np.abs(dh) > 5  # ft/sample ≈ 75 ft/min/sample at 4s

        # Local stability of raw signal in the killed regions
        loc_std = _local_std(raw, half=3)

        for i in np.where(cap_only)[0]:
            if climb_mask[i]:
                n_in_climb += 1
                deltas_climb.append(abs(raw[i] - era[i]) if not np.isnan(raw[i]) and not np.isnan(era[i]) else np.nan)
            else:
                n_in_cruise += 1
                deltas_cruise.append(abs(raw[i] - era[i]) if not np.isnan(raw[i]) and not np.isnan(era[i]) else np.nan)
            if not np.isnan(loc_std[i]):
                if loc_std[i] < (cap / 5):  # very stable locally
                    n_raw_stable += 1
                elif loc_std[i] > cap:  # jumping
                    n_raw_jumpy += 1

    print()
    print(f"=== {col}  (cap={cap}) ===")
    print(f"  total points killed only by cap : {cap_killed_total}")
    print(f"    in climb/descent (|dh/dt|>5)  : {n_in_climb}  ({n_in_climb/max(1,cap_killed_total):.0%})")
    print(f"    in cruise                     : {n_in_cruise}  ({n_in_cruise/max(1,cap_killed_total):.0%})")
    print(f"    raw locally stable (~bds bias): {n_raw_stable}")
    print(f"    raw locally jumpy             : {n_raw_jumpy}")
    if deltas_climb:
        arr = np.array([d for d in deltas_climb if not np.isnan(d)])
        if arr.size:
            print(f"    |raw - era| in climb : median={np.median(arr):.4f}  p95={np.quantile(arr, 0.95):.4f}  max={arr.max():.4f}")
    if deltas_cruise:
        arr = np.array([d for d in deltas_cruise if not np.isnan(d)])
        if arr.size:
            print(f"    |raw - era| in cruise: median={np.median(arr):.4f}  p95={np.quantile(arr, 0.95):.4f}  max={arr.max():.4f}")
