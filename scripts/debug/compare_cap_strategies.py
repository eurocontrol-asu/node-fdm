"""Plot 4 cleaning strategies side-by-side on 2 pathological flights.

Strategies (Mach + IAS only — TAS unchanged):

  V0 — current production: frozen + Hampel + ERA fill (NO cap)
  V1 — wide cap everywhere: frozen + Hampel + cap_wide (0.15 Mach / 80 kt)
  V2 — phase-aware: frozen + Hampel + cap (0.05 Mach / 30 kt) ONLY in cruise
       (|dh/dt|<3 ft/sample ≈ 45 ft/min @ 4s sampling)
  V3 — V2 + post-fill cleanup: same as V2 but also re-Hampel after ERA fill

For each flight:
  - 2 rows (Mach, IAS), 1 column
  - lines: bds raw (gray dots), era (orange), V0 (blue), V1 (green), V2 (red), V3 (purple)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from node_fdm_data.preprocessing.clean_speeds import (
    _fill_with_era,
    _flag_frozen_runs,
    _hampel_filter,
    _interpolate_short_gaps,
)

OUT = Path("data/figures/cap_strategies")
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 7
K = 3.0
N_PASSES = 3
INTERP = 10
FROZEN = 20

DH_CRUISE = 3.0  # ft/sample threshold

# Caps
CAP_WIDE_MACH = 0.15
CAP_WIDE_IAS = 80.0
CAP_NARROW_MACH = 0.05
CAP_NARROW_IAS = 30.0


def _hampel_pass(x):
    out = x.copy()
    for _ in range(N_PASSES):
        out = _hampel_filter(out, window=WINDOW, k=K)
    return out


def _v0(values, era):
    """Current prod: frozen + Hampel + ERA fill."""
    out = _flag_frozen_runs(values.astype(np.float64, copy=True), min_run_len=FROZEN)
    out = _hampel_pass(out)
    out = _interpolate_short_gaps(out, max_gap=INTERP)
    out = _fill_with_era(out, era)
    out = _hampel_filter(out, window=max(3, WINDOW // 2 + 1), k=K)
    out = _interpolate_short_gaps(out, max_gap=INTERP)
    return out


def _v1(values, era, cap):
    """Wide cap everywhere."""
    out = _flag_frozen_runs(values.astype(np.float64, copy=True), min_run_len=FROZEN)
    out = _hampel_pass(out)
    valid = ~np.isnan(out) & ~np.isnan(era)
    out[valid & (np.abs(out - era) > cap)] = np.nan
    out = _interpolate_short_gaps(out, max_gap=INTERP)
    out = _fill_with_era(out, era)
    out = _hampel_filter(out, window=max(3, WINDOW // 2 + 1), k=K)
    out = _interpolate_short_gaps(out, max_gap=INTERP)
    return out


def _v2(values, era, alt, cap):
    """Phase-aware: cap only in cruise (|dh/dt| < threshold)."""
    out = _flag_frozen_runs(values.astype(np.float64, copy=True), min_run_len=FROZEN)
    out = _hampel_pass(out)
    dh = np.gradient(alt)
    cruise = np.abs(dh) < DH_CRUISE
    valid = ~np.isnan(out) & ~np.isnan(era) & cruise
    out[valid & (np.abs(out - era) > cap)] = np.nan
    out = _interpolate_short_gaps(out, max_gap=INTERP)
    out = _fill_with_era(out, era)
    out = _hampel_filter(out, window=max(3, WINDOW // 2 + 1), k=K)
    out = _interpolate_short_gaps(out, max_gap=INTERP)
    return out


def _v3(values, era, alt, cap):
    """V2 + extra post-fill Hampel pass to catch ERA fill artifacts."""
    out = _v2(values, era, alt, cap)
    # Extra cleanup: re-flag frozen + Hampel after the ERA fill, since ERA
    # itself can introduce constant plateaus or step jumps.
    out = _flag_frozen_runs(out, min_run_len=FROZEN)
    out = _hampel_pass(out)
    out = _interpolate_short_gaps(out, max_gap=INTERP)
    return out


df = pl.read_delta("data/flights.delta")

FLIGHTS = ["4a0443_BTI7FR_s0", "738284_ISR826_s0"]

for fid in FLIGHTS:
    f = df.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
    t = f["raw_timestamp"].to_numpy()
    alt = f["raw_alt_ft"].cast(pl.Float64).to_numpy() if "raw_alt_ft" in f.columns else np.full(f.height, np.nan)

    raw_mach = f["bds_mach"].cast(pl.Float64).to_numpy()
    era_mach = f["era_mach"].cast(pl.Float64).to_numpy()
    raw_ias = f["bds_ias_kt"].cast(pl.Float64).to_numpy()
    era_cas = f["era_cas_kt"].cast(pl.Float64).to_numpy()

    v0_m = _v0(raw_mach, era_mach)
    v1_m = _v1(raw_mach, era_mach, cap=CAP_WIDE_MACH)
    v2_m = _v2(raw_mach, era_mach, alt, cap=CAP_NARROW_MACH)
    v3_m = _v3(raw_mach, era_mach, alt, cap=CAP_NARROW_MACH)

    v0_i = _v0(raw_ias, era_cas)
    v1_i = _v1(raw_ias, era_cas, cap=CAP_WIDE_IAS)
    v2_i = _v2(raw_ias, era_cas, alt, cap=CAP_NARROW_IAS)
    v3_i = _v3(raw_ias, era_cas, alt, cap=CAP_NARROW_IAS)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, raw_mach, ".", ms=2, alpha=0.3, color="tab:gray", label="bds_mach raw")
    ax.plot(t, era_mach, "-", lw=0.8, alpha=0.6, color="tab:orange", label="era_mach")
    ax.plot(t, v0_m, "-", lw=1.2, color="tab:blue", label="V0 prod (no cap)")
    ax.plot(t, v1_m, "-", lw=1.0, alpha=0.8, color="tab:green", label=f"V1 wide cap ({CAP_WIDE_MACH})")
    ax.plot(t, v2_m, "-", lw=1.0, alpha=0.8, color="tab:red", label=f"V2 cruise-only cap ({CAP_NARROW_MACH})")
    ax.plot(t, v3_m, "--", lw=1.0, alpha=0.8, color="tab:purple", label="V3 V2+post-fill clean")
    ax.set_ylabel("Mach")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, raw_ias, ".", ms=2, alpha=0.3, color="tab:gray", label="bds_ias_kt raw")
    ax.plot(t, era_cas, "-", lw=0.8, alpha=0.6, color="tab:orange", label="era_cas_kt")
    ax.plot(t, v0_i, "-", lw=1.2, color="tab:blue", label="V0 prod (no cap)")
    ax.plot(t, v1_i, "-", lw=1.0, alpha=0.8, color="tab:green", label=f"V1 wide cap ({CAP_WIDE_IAS})")
    ax.plot(t, v2_i, "-", lw=1.0, alpha=0.8, color="tab:red", label=f"V2 cruise-only cap ({CAP_NARROW_IAS})")
    ax.plot(t, v3_i, "--", lw=1.0, alpha=0.8, color="tab:purple", label="V3 V2+post-fill clean")
    ax.set_ylabel("IAS / CAS [kt]")
    ax.set_xlabel("timestamp")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Cleaning strategy comparison — {fid}", fontsize=12)
    fig.tight_layout()
    out = OUT / f"{fid}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")
