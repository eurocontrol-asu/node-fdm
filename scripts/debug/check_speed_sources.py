"""Compare speed sources: BDS (Mode-S) vs ERA5-derived.

Plots 2x2:
- Row 1: Mach (bds_mach vs era_mach), IAS/CAS (bds_ias_kt vs era_cas_kt)
- Row 2: TAS (bds_tas_kt vs era_tas_kt), Altitude profile

Usage:
    uv run python scripts/check_speed_sources.py [--flight FLIGHT_ID] [--n N] [--seed SEED]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from deltalake import DeltaTable


def _fill_bds_with_era(bds: np.ndarray, era: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in BDS with ERA values."""
    out = bds.copy()
    mask = np.isnan(out) & ~np.isnan(era)
    out[mask] = era[mask]
    return out


def _flag_point_jumps(x: np.ndarray, max_jump: float) -> np.ndarray:
    """Flag isolated V-shape outliers as NaN.

    A point x[i] is flagged if BOTH neighbours differ by more than max_jump:
        |x[i] - x[i-1]| > max_jump AND |x[i+1] - x[i]| > max_jump
    AND the two jumps go in opposite directions (V-shape, not gradient).
    This catches spike-then-recover artifacts without flagging legitimate
    rate-of-change events (climb/descent transitions).
    """
    out = x.copy()
    n = len(out)
    for i in range(1, n - 1):
        if np.isnan(x[i]) or np.isnan(x[i - 1]) or np.isnan(x[i + 1]):
            continue
        d_prev = x[i] - x[i - 1]
        d_next = x[i + 1] - x[i]
        if abs(d_prev) > max_jump and abs(d_next) > max_jump and d_prev * d_next < 0:
            out[i] = np.nan
    return out


def _hampel_filter(x: np.ndarray, window: int = 7, k: float = 3.0) -> np.ndarray:
    """Hampel filter — flag outliers as NaN.

    For each point, compute median and MAD over a centered window.
    Flag point as outlier if |x - median| > k * 1.4826 * MAD.

    Args:
        x: 1D array (NaN-aware).
        window: half-window size (full window = 2*window + 1).
        k: number of MADs for outlier threshold.

    Returns:
        Copy of x with outliers replaced by NaN.
    """
    n = len(x)
    out = x.copy()
    scale = 1.4826  # consistency factor for Gaussian
    for i in range(n):
        if np.isnan(x[i]):
            continue
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        w = x[lo:hi]
        w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if mad == 0:
            continue
        if abs(x[i] - med) > k * scale * mad:
            out[i] = np.nan
    return out


def _interpolate_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length <= max_gap; leave longer runs as NaN."""
    out = x.copy()
    n = len(out)
    isnan = np.isnan(out)
    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap and i > 0 and j < n and not isnan[i - 1] and not isnan[j]:
                lo, hi = out[i - 1], out[j]
                for k in range(i, j):
                    out[k] = lo + (hi - lo) * (k - i + 1) / (gap_len + 1)
            i = j
        else:
            i += 1
    return out


def _clean_bds(
    bds: np.ndarray,
    era: np.ndarray,
    window: int = 5,
    k: float = 2.5,
    era_dev_max: float | None = None,
    n_passes: int = 3,
    interp_max_gap: int = 10,
    point_jump_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean BDS: multi-pass Hampel + ERA cap + point-jump residual + interp gaps.

    Strategy:
    1. Multi-pass Hampel flags outliers as NaN (handles clusters where median is biased).
    2. ERA-deviation cap catches systematic biases that median misses.
    3. Point-jump V-shape filter catches isolated spike-then-recover outliers
       that survived Hampel (local gradient inflated MAD) AND the cap (deviation
       stayed below absolute threshold).
    4. Linearly interpolate short NaN runs (smooth, avoids brutal BDS→ERA jumps).
    5. (Caller) ERA fills only long gaps (true Mode-S blackouts).

    Returns:
        (cleaned_bds, outlier_mask)
    """
    cleaned = bds.copy()
    for _ in range(n_passes):
        cleaned = _hampel_filter(cleaned, window=window, k=k)
    if era_dev_max is not None:
        valid = ~np.isnan(cleaned) & ~np.isnan(era)
        dev_outlier = valid & (np.abs(cleaned - era) > era_dev_max)
        cleaned[dev_outlier] = np.nan
    if point_jump_max is not None:
        cleaned = _flag_point_jumps(cleaned, max_jump=point_jump_max)
    outlier_mask = ~np.isnan(bds) & np.isnan(cleaned)
    cleaned = _interpolate_short_gaps(cleaned, max_gap=interp_max_gap)
    return cleaned, outlier_mask


def plot_speed_sources(df: pl.DataFrame, out_path: Path | None = None) -> None:
    n = len(df)
    t = np.arange(n) * 4 / 60  # minutes

    fid = df["meta_flight_id"][0]

    bds_mach = df["bds_mach"].to_numpy() if "bds_mach" in df.columns else np.full(n, np.nan)
    era_mach = df["era_mach"].to_numpy() if "era_mach" in df.columns else np.full(n, np.nan)
    bds_mach_clean, mach_outliers = _clean_bds(bds_mach, era_mach, window=7, k=3.0, era_dev_max=0.025, point_jump_max=0.02)
    cleaned_filled_mach = _fill_bds_with_era(bds_mach_clean, era_mach)
    cleaned_filled_mach = _hampel_filter(cleaned_filled_mach, window=5, k=3.0)
    cleaned_filled_mach = _interpolate_short_gaps(cleaned_filled_mach, max_gap=10)

    bds_ias = df["bds_ias_kt"].to_numpy() if "bds_ias_kt" in df.columns else np.full(n, np.nan)
    era_cas = df["era_cas_kt"].to_numpy() if "era_cas_kt" in df.columns else np.full(n, np.nan)
    bds_ias_clean, ias_outliers = _clean_bds(bds_ias, era_cas, window=7, k=3.0, era_dev_max=10.0, point_jump_max=8.0)
    cleaned_filled_ias = _fill_bds_with_era(bds_ias_clean, era_cas)
    cleaned_filled_ias = _hampel_filter(cleaned_filled_ias, window=5, k=3.0)
    cleaned_filled_ias = _interpolate_short_gaps(cleaned_filled_ias, max_gap=10)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(f"Speed Sources — {fid}", fontsize=13, fontweight="bold")

    # --- Mach ---
    ax = axes[0]
    ax.plot(t, bds_mach, color="#e74c3c", linewidth=0.6, alpha=0.6, label="BDS Mach (raw)")
    ax.plot(t, era_mach, color="#3498db", linewidth=0.8, alpha=0.7, label="ERA Mach")
    ax.plot(t, cleaned_filled_mach, color="#27ae60", linewidth=1.2, label="BDS cleaned + filled w/ ERA")
    if mach_outliers.any():
        ax.scatter(t[mach_outliers], bds_mach[mach_outliers], color="#c0392b", s=12,
                   marker="x", label=f"Outliers ({mach_outliers.sum()})", zorder=5)
    n_out = mach_outliers.sum()
    pct = 100 * n_out / max(1, (~np.isnan(bds_mach)).sum())
    ax.set_title(f"Mach  (outliers flagged: {n_out} / {pct:.1f}%)", fontsize=10)
    ax.set_ylabel("Mach")
    ax.set_xlabel("Time (min)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # --- IAS / CAS ---
    ax = axes[1]
    ax.plot(t, bds_ias, color="#e74c3c", linewidth=0.6, alpha=0.6, label="BDS IAS (raw)")
    ax.plot(t, era_cas, color="#3498db", linewidth=0.8, alpha=0.7, label="ERA CAS")
    ax.plot(t, cleaned_filled_ias, color="#27ae60", linewidth=1.2, label="BDS cleaned + filled w/ ERA")
    if ias_outliers.any():
        ax.scatter(t[ias_outliers], bds_ias[ias_outliers], color="#c0392b", s=12,
                   marker="x", label=f"Outliers ({ias_outliers.sum()})", zorder=5)
    n_out = ias_outliers.sum()
    pct = 100 * n_out / max(1, (~np.isnan(bds_ias)).sum())
    ax.set_title(f"IAS / CAS  (outliers flagged: {n_out} / {pct:.1f}%)", fontsize=10)
    ax.set_ylabel("Speed (kt)")
    ax.set_xlabel("Time (min)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight", type=str, default=None, help="Flight ID")
    parser.add_argument("--n", type=int, default=3, help="Number of random flights")
    parser.add_argument("--seed", type=int, default=99, help="Random seed")
    args = parser.parse_args()

    delta_path = Path("data/flights.delta")
    dt = DeltaTable(str(delta_path))
    df_all = pl.from_arrow(dt.to_pyarrow_table())

    if args.flight:
        flight_ids = [args.flight]
    else:
        flights = df_all.group_by("meta_flight_id").len()
        long_flights = flights.filter(pl.col("len") > 500)
        rng = np.random.default_rng(args.seed)
        sampled = rng.choice(
            long_flights["meta_flight_id"].to_numpy(),
            size=min(args.n, len(long_flights)),
            replace=False,
        )
        flight_ids = list(sampled)

    for fid in flight_ids:
        print(f"\n{'=' * 60}")
        print(f"Flight: {fid}")
        print(f"{'=' * 60}")

        df_flight = df_all.filter(pl.col("meta_flight_id") == fid)
        print(f"  Rows: {len(df_flight)}")

        out = Path(f"data/figures/speeds_{fid}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_speed_sources(df_flight, out)


if __name__ == "__main__":
    main()
