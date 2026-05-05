"""Diagnose where V0-MAD and V1-MAD (Polars rolling-rolling) diverge.

Plots, for one real flight signal:
  - the raw signal x
  - V0-MAD (numpy, single median per window)
  - V1-MAD (Polars rolling of abs_dev with j-local medians)
  - the resulting outlier masks
"""

from __future__ import annotations

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HAMPEL_SCALE = 1.4826
_HAMPEL_MIN_WINDOW = 3


def v0_mad(x: np.ndarray, window: int) -> np.ndarray:
    n = len(x)
    med = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(x[i]):
            continue
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        w = x[lo:hi]
        w = w[~np.isnan(w)]
        if len(w) < _HAMPEL_MIN_WINDOW:
            continue
        m = float(np.median(w))
        med[i] = m
        mad[i] = float(np.median(np.abs(w - m)))
    return med, mad


def v1_mad(x: np.ndarray, window: int):
    win = 2 * window + 1
    s = pl.Series("x", x, dtype=pl.Float64).fill_nan(None)
    med = s.rolling_median(window_size=win, min_samples=_HAMPEL_MIN_WINDOW, center=True)
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling_median(window_size=win, min_samples=_HAMPEL_MIN_WINDOW, center=True)
    return med.to_numpy(), mad.to_numpy()


def main():
    df = pl.read_delta("data/flights.delta")
    flights = df.partition_by("meta_flight_id", maintain_order=True)
    # Find a flight with bds_mach signal
    for fdf in flights:
        arr = fdf["bds_mach"].cast(pl.Float64).to_numpy()
        if np.sum(~np.isnan(arr)) > 200:
            flight_id = fdf["meta_flight_id"][0]
            alt = fdf["raw_alt_ft"].cast(pl.Float64).to_numpy()
            break

    print(f"Using flight {flight_id}, n={len(arr)}, valid={int(np.sum(~np.isnan(arr)))}")
    window = 50
    k = 3.0

    med0, mad0 = v0_mad(arr, window)
    med1, mad1 = v1_mad(arr, window)

    threshold0 = k * _HAMPEL_SCALE * mad0
    threshold1 = k * _HAMPEL_SCALE * mad1

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(arr, ".", ms=2, label="bds_mach raw", color="C0")
    axes[0].plot(med0, label="V0 rolling median (numpy)", color="C1", lw=1)
    axes[0].plot(med1, label="V1 rolling median (Polars)", color="C2", lw=1, ls="--")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Mach")
    axes[0].set_title(f"Flight {flight_id} — V0 vs V1 Hampel internals (window=50)")

    axes[1].plot(alt, color="gray", lw=0.8)
    axes[1].set_ylabel("alt (ft)")
    axes[1].set_title("Flight altitude profile (context)")

    axes[2].plot(mad0, label="V0 MAD", color="C1")
    axes[2].plot(mad1, label="V1 MAD", color="C2", ls="--")
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("MAD")
    axes[2].set_yscale("log")
    axes[2].set_title("MAD comparison (log scale)")

    axes[3].plot(threshold0, label="V0 threshold = k·1.4826·MAD", color="C1")
    axes[3].plot(threshold1, label="V1 threshold", color="C2", ls="--")
    # Mark V1-extra outliers (V0 keeps but V1 drops)
    abs_diff = np.abs(arr - med0)
    v0_outlier = (abs_diff > threshold0) & (mad0 > 0)
    abs_diff1 = np.abs(arr - med1)
    v1_outlier = (abs_diff1 > threshold1) & (mad1 > 0)
    extra = v1_outlier & ~v0_outlier & ~np.isnan(arr)
    extra_idx = np.where(extra)[0]
    if len(extra_idx):
        axes[3].plot(extra_idx, threshold0[extra_idx], "rx", ms=6,
                     label=f"V1-extra outliers ({len(extra_idx)})")
    axes[3].legend(loc="upper right")
    axes[3].set_ylabel("threshold (Mach)")
    axes[3].set_xlabel("sample index")
    axes[3].set_yscale("log")

    out_path = "scripts/debug/hampel_mad_divergence.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print(f"Saved {out_path}")

    # Quantitative summary
    print(f"\nV0 outliers: {int(v0_outlier.sum())}")
    print(f"V1 outliers: {int(v1_outlier.sum())}")
    print(f"V1-extra (V0 keeps, V1 drops): {len(extra_idx)}")
    if len(extra_idx):
        # In which altitude phase?
        alt_at_extra = alt[extra_idx]
        print(f"  altitude at V1-extra outliers: "
              f"min={np.nanmin(alt_at_extra):.0f}ft, "
              f"median={np.nanmedian(alt_at_extra):.0f}ft, "
              f"max={np.nanmax(alt_at_extra):.0f}ft")
        # MAD ratio
        ratio = mad1[extra_idx] / mad0[extra_idx]
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio):
            print(f"  V1-MAD / V0-MAD at extras: median={np.median(ratio):.3f}, "
                  f"min={np.min(ratio):.3f}")


if __name__ == "__main__":
    main()
