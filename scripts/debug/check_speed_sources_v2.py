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


def _on_ground_mask(
    vz_ftmin: np.ndarray,
    alt_ft: np.ndarray,
    *,
    vz_threshold: float = 200.0,
    alt_threshold: float = 1500.0,
) -> np.ndarray:
    """True for samples considered "on ground".

    A sample is on ground if BOTH:
      - alt < alt_threshold ft  (below any reasonable cruise altitude)
      - |vz| < vz_threshold ft/min  (no climb/descent activity)

    NaN inputs are treated as "no evidence of being airborne" (vz NaN
    → 0, alt NaN → 0).
    """
    abs_vz = np.where(np.isnan(vz_ftmin), 0.0, np.abs(vz_ftmin))
    alt = np.where(np.isnan(alt_ft), 0.0, alt_ft)
    return (alt < alt_threshold) & (abs_vz < vz_threshold)


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


def _flag_zigzag_region(
    x: np.ndarray, *, half_window: int, jump_min: float, density_min: float
) -> np.ndarray:
    """NaN-out entire regions with high density of large deltas.

    A physically reasonable speed signal has very few large 1-sample
    jumps (climb/descent transitions are smooth at 4s sampling). When
    the local density of |delta| > jump_min exceeds density_min over a
    centered window of half_window points, the whole window is corrupt
    (alternating plateaus, dropouts, BDS frame desync) and the center
    point is flagged.
    """
    out = x.copy()
    n = len(out)
    deltas = np.abs(np.diff(x))  # length n-1, NaN-propagating
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n - 1, i + half_window)
        win = deltas[lo:hi]
        win = win[~np.isnan(win)]
        if win.size < 4:
            continue
        if (win > jump_min).mean() >= density_min:
            out[i] = np.nan
    return out


def _clean_bds(
    bds: np.ndarray,
    window: int = 7,
    k: float = 3.0,
    n_passes: int = 3,
    point_jump_max: float | None = None,
    zigzag_jump_min: float | None = None,
    zigzag_half_window: int = 15,
    zigzag_density_min: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean BDS: Hampel + V-shape + zigzag-region detector.

    The zigzag-region detector kills entire windows where >40% of deltas
    are sign-flipping jumps — catches fundamentally corrupted zones that
    point-wise filters only punch holes in.
    """
    cleaned = bds.copy()
    for _ in range(n_passes):
        cleaned = _hampel_filter(cleaned, window=window, k=k)
    if point_jump_max is not None:
        for _ in range(n_passes):
            cleaned = _flag_point_jumps(cleaned, max_jump=point_jump_max)
    if zigzag_jump_min is not None:
        # Run on the RAW signal (not on already-filtered cleaned) so that
        # NaNs introduced by Hampel don't break delta computation.
        zigzag_mask = np.isnan(
            _flag_zigzag_region(
                bds,
                half_window=zigzag_half_window,
                jump_min=zigzag_jump_min,
                density_min=zigzag_density_min,
            )
        ) & ~np.isnan(bds)
        cleaned[zigzag_mask] = np.nan
    outlier_mask = ~np.isnan(bds) & np.isnan(cleaned)
    return cleaned, outlier_mask


def _clean_era(
    era: np.ndarray,
    window: int = 7,
    k: float = 3.0,
    n_passes: int = 3,
    zigzag_jump_min: float | None = None,
    zigzag_half_window: int = 15,
    zigzag_density_min: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean ERA: multi-pass Hampel + big-jump density region detector.

    ERA is normally extremely smooth in time, so any region with multiple
    large 1-sample jumps is corrupt (grid artefact, missing tile, etc.)
    and should be NaN-ed out as a block, not point-by-point.
    """
    cleaned = era.copy()
    for _ in range(n_passes):
        cleaned = _hampel_filter(cleaned, window=window, k=k)
    if zigzag_jump_min is not None:
        zigzag_mask = np.isnan(
            _flag_zigzag_region(
                era,
                half_window=zigzag_half_window,
                jump_min=zigzag_jump_min,
                density_min=zigzag_density_min,
            )
        ) & ~np.isnan(era)
        cleaned[zigzag_mask] = np.nan
    outlier_mask = ~np.isnan(era) & np.isnan(cleaned)
    return cleaned, outlier_mask


def plot_speed_sources(df: pl.DataFrame, out_path: Path | None = None) -> None:
    n = len(df)
    t = np.arange(n) * 4 / 60  # minutes

    fid = df["meta_flight_id"][0]

    bds_mach = df["bds_mach"].to_numpy() if "bds_mach" in df.columns else np.full(n, np.nan)
    era_mach = df["era_mach"].to_numpy() if "era_mach" in df.columns else np.full(n, np.nan)
    bds_mach_clean, mach_outliers = _clean_bds(
        bds_mach, window=15, k=3.0,
        point_jump_max=0.05,
        zigzag_jump_min=0.05, zigzag_half_window=15, zigzag_density_min=0.3,
    )
    era_mach_clean, era_mach_outliers = _clean_era(
        era_mach, window=15, k=3.0,
        zigzag_jump_min=0.05, zigzag_half_window=15, zigzag_density_min=0.15,
    )
    # On-ground mask: drop pre-flight & post-landing samples from final output.
    vz = df["raw_vz_ftmin"].cast(pl.Float64).to_numpy() if "raw_vz_ftmin" in df.columns else np.full(n, np.nan)
    alt = df["raw_alt_ft"].cast(pl.Float64).to_numpy() if "raw_alt_ft" in df.columns else np.full(n, np.nan)
    on_ground = _on_ground_mask(vz, alt, vz_threshold=200.0, alt_threshold=1500.0)

    cleaned_filled_mach = _fill_bds_with_era(bds_mach_clean, era_mach_clean)
    cleaned_filled_mach = _hampel_filter(cleaned_filled_mach, window=5, k=3.0)
    cleaned_filled_mach = _interpolate_short_gaps(cleaned_filled_mach, max_gap=10)
    cleaned_filled_mach[on_ground] = np.nan

    bds_ias = df["bds_ias_kt"].to_numpy() if "bds_ias_kt" in df.columns else np.full(n, np.nan)
    era_cas = df["era_cas_kt"].to_numpy() if "era_cas_kt" in df.columns else np.full(n, np.nan)
    bds_ias_clean, ias_outliers = _clean_bds(
        bds_ias, window=15, k=3.0,
        point_jump_max=20.0,
        zigzag_jump_min=20.0, zigzag_half_window=15, zigzag_density_min=0.3,
    )
    era_cas_clean, era_cas_outliers = _clean_era(
        era_cas, window=15, k=3.0,
        zigzag_jump_min=20.0, zigzag_half_window=15, zigzag_density_min=0.15,
    )
    cleaned_filled_ias = _fill_bds_with_era(bds_ias_clean, era_cas_clean)
    cleaned_filled_ias = _hampel_filter(cleaned_filled_ias, window=5, k=3.0)
    cleaned_filled_ias = _interpolate_short_gaps(cleaned_filled_ias, max_gap=10)
    cleaned_filled_ias[on_ground] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(f"Speed Sources — {fid}", fontsize=13, fontweight="bold")

    # --- Mach ---
    ax = axes[0]
    ax.plot(t, bds_mach, color="#e74c3c", linewidth=0.6, alpha=0.6, label="BDS Mach (raw)")
    ax.plot(t, era_mach, color="#3498db", linewidth=0.8, alpha=0.7, label="ERA Mach")
    ax.plot(t, cleaned_filled_mach, color="#27ae60", linewidth=1.2, label="BDS cleaned + filled w/ ERA")
    if mach_outliers.any():
        ax.scatter(t[mach_outliers], bds_mach[mach_outliers], color="#c0392b", s=12,
                   marker="x", label=f"BDS outliers ({mach_outliers.sum()})", zorder=5)
    if era_mach_outliers.any():
        ax.scatter(t[era_mach_outliers], era_mach[era_mach_outliers], color="#8e44ad", s=12,
                   marker="+", label=f"ERA outliers ({era_mach_outliers.sum()})", zorder=5)
    n_out = mach_outliers.sum()
    pct = 100 * n_out / max(1, (~np.isnan(bds_mach)).sum())
    ax.set_title(f"Mach  (BDS out: {n_out} / {pct:.1f}%, ERA out: {era_mach_outliers.sum()})", fontsize=10)
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
                   marker="x", label=f"BDS outliers ({ias_outliers.sum()})", zorder=5)
    if era_cas_outliers.any():
        ax.scatter(t[era_cas_outliers], era_cas[era_cas_outliers], color="#8e44ad", s=12,
                   marker="+", label=f"ERA outliers ({era_cas_outliers.sum()})", zorder=5)
    n_out = ias_outliers.sum()
    pct = 100 * n_out / max(1, (~np.isnan(bds_ias)).sum())
    ax.set_title(f"IAS / CAS  (BDS out: {n_out} / {pct:.1f}%, ERA out: {era_cas_outliers.sum()})", fontsize=10)
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

        out = Path(f"data/figures/speeds_v2_{fid}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_speed_sources(df_flight, out)


if __name__ == "__main__":
    main()
