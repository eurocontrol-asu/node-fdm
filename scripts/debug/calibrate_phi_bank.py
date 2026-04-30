"""Calibrate phi_bank cap from empirical bank-angle distribution.

Computes the implicit bank angle phi = atan(V * d_track/dt / g) per
sample for every flight in the Delta dataset and recommends a cap value
for ``StructuredLayer.fdm_phi_bank_rad`` based on the p99.9 percentile.

Usage:
    uv run python scripts/debug/calibrate_phi_bank.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from deltalake import DeltaTable
from scipy.signal import medfilt, savgol_filter

REPO_ROOT = Path("/Users/gabriel/Documents/Code/python/node-fdm-v2")
DELTA_PATH = REPO_ROOT / "data" / "flights.delta"
OUT_DIR = REPO_ROOT / "data" / "figures" / "lateral_step1" / "phi_bank"

G = 9.80665
KT_TO_MS = 0.5144444
DT = 4.0
SAVGOL_WINDOW = 9
SAVGOL_POLY = 3
MEDFILT_KERNEL = 5  # median filter width (samples) to kill GPS spikes pre-unwrap
EDGE_TRIM = 5
MIN_TAS_MS = 50.0
PHI_OUTLIER_RAD = 1.4  # ~80 deg, clearly GPS noise


def _fill_track_rad(track_deg: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs in track[deg] and convert to radians.

    Mirrors detect_turning_starts: np.unwrap and savgol_filter cannot
    tolerate NaNs, so we forward-fill (then backfill the leading run)
    before downstream processing. Within-flight only -- caller ensures
    slice boundaries.
    """
    n = track_deg.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    x = track_deg.astype(np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.all():
        return np.full(n, np.nan)
    if bad.any():
        good = np.flatnonzero(~bad)
        last = x[good[0]]
        for i in range(n):
            if bad[i]:
                x[i] = last
            else:
                last = x[i]
    return np.radians(x)


def _track_rate_rad_per_s(track_deg: np.ndarray) -> np.ndarray:
    """Smoothed d_track/dt in rad/s.

    Pipeline: NaN-fill -> medfilt(k=5) on the deg signal -> deg2rad ->
    np.unwrap (mod 2pi) -> Savgol on the continuous unwrapped signal ->
    finite difference. The pre-unwrap median kills 1-2 sample GPS
    spikes (which Savgol blurs into multi-sample bumps) without
    deforming real manoeuvres (10-30 s = 3-8 samples at 0.25 Hz).
    """
    n = track_deg.size
    if n < SAVGOL_WINDOW:
        return np.full(n, np.nan)
    track_rad_filled = _fill_track_rad(track_deg)
    if not np.isfinite(track_rad_filled).any():
        return np.full(n, np.nan)
    # Median filter on the continuous-after-fill radian signal. We
    # apply it BEFORE unwrap because spikes in raw track[deg] often
    # appear as huge sample-to-sample jumps that np.unwrap would
    # mistake for true 2pi crossings.
    track_rad_median = medfilt(track_rad_filled, kernel_size=MEDFILT_KERNEL)
    unwrapped = np.unwrap(track_rad_median)
    smoothed = savgol_filter(unwrapped, SAVGOL_WINDOW, SAVGOL_POLY)
    d = np.diff(smoothed, prepend=smoothed[0])
    return d / DT


def _count_wrap_crossings(track_deg: np.ndarray) -> tuple[int, int]:
    """Diagnostic: (n_crossings, n_valid_diffs) on raw track[deg].

    A wrap crossing is |raw diff| > 300 deg, i.e. a sample-to-sample
    apparent jump that reflects the 0/360 discontinuity rather than a
    real turn.
    """
    n = track_deg.size
    if n < 2:
        return 0, 0
    x = track_deg.astype(np.float64)
    finite_pair = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    raw_diff = np.abs(x[1:] - x[:-1])
    crossings = int(np.sum(finite_pair & (raw_diff > 300.0)))
    return crossings, int(finite_pair.sum())


def _flight_phi(flight: pl.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Compute |phi_implicite| array for one flight + drop-counters."""
    counters = {
        "raw": 0,
        "low_tas": 0,
        "outlier": 0,
        "used": 0,
        "wrap_crossings": 0,
        "diff_pairs": 0,
    }
    n = flight.height
    if n < SAVGOL_WINDOW + 2 * EDGE_TRIM:
        return np.empty(0), counters

    track = flight["raw_track_deg"].to_numpy().astype(np.float64)
    bds_tas = flight["bds_tas_kt_clean"].to_numpy().astype(np.float64)
    era_tas = flight["era_tas_kt"].to_numpy().astype(np.float64)
    # Prefer cleaned BDS, fallback to ERA5.
    tas_kt = np.where(np.isfinite(bds_tas), bds_tas, era_tas)
    tas_ms = tas_kt * KT_TO_MS

    crossings, diff_pairs = _count_wrap_crossings(track)
    counters["wrap_crossings"] = crossings
    counters["diff_pairs"] = diff_pairs

    rate = _track_rate_rad_per_s(track)
    phi = np.arctan(tas_ms * rate / G)
    abs_phi = np.abs(phi)

    # Edge trim
    abs_phi[:EDGE_TRIM] = np.nan
    abs_phi[-EDGE_TRIM:] = np.nan

    base_valid = np.isfinite(abs_phi) & np.isfinite(tas_ms)
    counters["raw"] = int(base_valid.sum())

    low_tas = base_valid & (tas_ms < MIN_TAS_MS)
    counters["low_tas"] = int(low_tas.sum())
    valid = base_valid & ~low_tas

    outlier = valid & (abs_phi > PHI_OUTLIER_RAD)
    counters["outlier"] = int(outlier.sum())
    valid &= ~outlier

    counters["used"] = int(valid.sum())
    return abs_phi[valid], counters


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    qs = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]
    keys = ["p50", "p75", "p90", "p95", "p99", "p99.5", "p99.9", "p99.99"]
    vals = np.percentile(arr, qs)
    return dict(zip(keys, [float(v) for v in vals], strict=True))


def _plot_histogram(abs_phi_deg: np.ndarray, pct_deg: dict[str, float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 90, 181)
    ax.hist(abs_phi_deg, bins=bins, color="tab:blue", alpha=0.7)
    ax.set_yscale("log")
    colors = {"p50": "tab:green", "p90": "tab:olive", "p99": "tab:orange",
              "p99.9": "tab:red", "p99.99": "tab:purple"}
    for k, c in colors.items():
        v = pct_deg[k]
        ax.axvline(v, color=c, ls="--", lw=1.2, label=f"{k}={v:.2f} deg")
    ax.set_xlabel("|phi_implicite| [deg]")
    ax.set_ylabel("count (log)")
    ax.set_title("Empirical distribution of implicit bank angle")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_cdf(abs_phi_deg: np.ndarray, p999_deg: float, out: Path) -> None:
    sorted_phi = np.sort(abs_phi_deg)
    cdf = np.arange(1, sorted_phi.size + 1) / sorted_phi.size
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_phi, cdf, color="tab:blue", lw=1.0)
    ax.axvline(p999_deg, color="tab:red", ls="--",
               label=f"p99.9 = {p999_deg:.2f} deg")
    ax.axhline(0.999, color="grey", ls=":", lw=0.8)
    ax.set_xlim(0, 60)
    ax.set_xlabel("|phi_implicite| [deg]")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Empirical CDF of |phi_implicite| (clipped to 60 deg)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_per_flight_p99(per_flight_p99_deg: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, max(60.0, float(per_flight_p99_deg.max()) * 1.05), 80)
    ax.hist(per_flight_p99_deg, bins=bins, color="tab:cyan", alpha=0.75)
    med = float(np.median(per_flight_p99_deg))
    p95 = float(np.percentile(per_flight_p99_deg, 95))
    ax.axvline(med, color="tab:green", ls="--", label=f"median={med:.2f} deg")
    ax.axvline(p95, color="tab:orange", ls="--", label=f"p95(flights)={p95:.2f} deg")
    ax.set_xlabel("per-flight p99 of |phi_implicite| [deg]")
    ax.set_ylabel("flight count")
    ax.set_title(f"Per-flight p99 distribution (n={per_flight_p99_deg.size} flights)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _recommend(p999: float, p9999: float) -> tuple[float, list[str]]:
    """Apply recommendation rules; returns (cap_rad, warnings)."""
    warnings: list[str] = []
    if p9999 > 1.5 * p999:
        warnings.append(
            f"HEAVY TAIL: p99.99/p99.9 = {p9999 / p999:.2f} > 1.5"
        )
    if p999 <= 0.7:
        cap = 0.7
        warnings.append("p99.9 <= 0.7 rad: default cap holds, headroom OK.")
    elif p999 <= 1.0:
        cap = round(p999, 2)
        warnings.append(
            "0.7 < p99.9 <= 1.0 rad: gradient stability acceptable up to 1.0 rad."
        )
    else:
        cap = float("nan")
        warnings.append(
            "p99.9 > 1.0 rad: GPS noise likely dominates. "
            "Pre-filter phi_implicite (e.g. median filter) before re-running."
        )
    return cap, warnings


def main() -> None:
    if not DELTA_PATH.exists():
        print(f"Delta table not found: {DELTA_PATH}", file=sys.stderr)
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DELTA_PATH} ...")
    dt_table = DeltaTable(str(DELTA_PATH))
    df = pl.DataFrame(
        dt_table.to_pyarrow_table(
            columns=[
                "meta_flight_id",
                "raw_timestamp",
                "raw_track_deg",
                "era_tas_kt",
                "bds_tas_kt_clean",
            ]
        )
    )
    n_total = df.height
    print(f"  {n_total} rows total")

    # Per-flight processing -- groupby + sort within group keeps smoothing
    # bound to flight boundaries.
    flight_ids = df["meta_flight_id"].unique().to_list()
    print(f"  {len(flight_ids)} flights")

    n_dropped_low_tas = 0
    n_dropped_outlier = 0
    n_used = 0
    n_raw = 0
    n_wrap_crossings = 0
    n_diff_pairs = 0
    abs_phi_chunks: list[np.ndarray] = []
    per_flight_p99: list[float] = []
    per_flight_p999: list[tuple[str, float]] = []

    df_sorted = df.sort(["meta_flight_id", "raw_timestamp"])
    # Iterate via partition_by to avoid 835x filter cost.
    for flight in df_sorted.partition_by("meta_flight_id", maintain_order=True):
        phi_arr, counters = _flight_phi(flight)
        n_raw += counters["raw"]
        n_dropped_low_tas += counters["low_tas"]
        n_dropped_outlier += counters["outlier"]
        n_used += counters["used"]
        n_wrap_crossings += counters["wrap_crossings"]
        n_diff_pairs += counters["diff_pairs"]
        if phi_arr.size > 0:
            abs_phi_chunks.append(phi_arr)
            per_flight_p99.append(float(np.percentile(phi_arr, 99)))
            fid = str(flight["meta_flight_id"][0])
            per_flight_p999.append((fid, float(np.percentile(phi_arr, 99.9))))

    if not abs_phi_chunks:
        print("No samples passed filters -- aborting.", file=sys.stderr)
        raise SystemExit(1)

    abs_phi = np.concatenate(abs_phi_chunks)
    per_flight_p99_arr = np.array(per_flight_p99)
    print(f"  used {n_used} samples (raw {n_raw}, dropped low_tas {n_dropped_low_tas}, "
          f"outlier {n_dropped_outlier})")

    pct_rad = _percentiles(abs_phi)
    pct_deg = {k: float(np.degrees(v)) for k, v in pct_rad.items()}
    p999 = pct_rad["p99.9"]
    p9999 = pct_rad["p99.99"]

    cap, warnings = _recommend(p999, p9999)

    _plot_histogram(np.degrees(abs_phi), pct_deg, OUT_DIR / "histogram.png")
    _plot_cdf(np.degrees(abs_phi), pct_deg["p99.9"], OUT_DIR / "cdf.png")
    _plot_per_flight_p99(np.degrees(per_flight_p99_arr), OUT_DIR / "per_flight_p99.png")

    wrap_pct = (
        100.0 * n_wrap_crossings / n_diff_pairs if n_diff_pairs else 0.0
    )
    stats = {
        "n_samples_total": int(n_total),
        "n_samples_used": int(n_used),
        "n_dropped_low_tas": int(n_dropped_low_tas),
        "n_dropped_outlier": int(n_dropped_outlier),
        "n_flights": int(len(flight_ids)),
        "n_wrap_crossings": int(n_wrap_crossings),
        "n_diff_pairs": int(n_diff_pairs),
        "wrap_crossings_pct": wrap_pct,
        "percentiles_rad": pct_rad,
        "percentiles_deg": pct_deg,
        "recommended_cap_rad": cap,
        "warnings": warnings,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    # Per-flight outlier listing: flights with p99.9 above the 1.0 rad cap.
    outliers = [(fid, q) for fid, q in per_flight_p999 if q > 1.0]
    outliers.sort(key=lambda t: t[1], reverse=True)
    n_outlier_flights = len(outliers)
    if 0 < n_outlier_flights < 10:
        lines = [f"{fid}\t{q:.4f}" for fid, q in outliers]
        (OUT_DIR / "outlier_flights_phi.txt").write_text("\n".join(lines) + "\n")
    elif n_outlier_flights >= 10:
        # Still write but cap to top 50 for inspection.
        lines = [f"{fid}\t{q:.4f}" for fid, q in outliers[:50]]
        (OUT_DIR / "outlier_flights_phi.txt").write_text(
            f"# {n_outlier_flights} flights have p99.9 > 1.0 rad; top 50 listed\n"
            + "\n".join(lines) + "\n"
        )

    cap_str = f"{cap:.3f} rad ({np.degrees(cap):.2f} deg)" if np.isfinite(cap) else "N/A (rule violation)"
    print()
    print("=" * 72)
    print("phi_bank calibration summary")
    print("=" * 72)
    print(
        f"Samples: {n_used}/{n_total} used (low_tas dropped: {n_dropped_low_tas}, "
        f"outlier > {PHI_OUTLIER_RAD:.2f} rad dropped: {n_dropped_outlier}). "
        f"|phi| percentiles [deg]: p50={pct_deg['p50']:.2f}, p90={pct_deg['p90']:.2f}, "
        f"p99={pct_deg['p99']:.2f}, p99.9={pct_deg['p99.9']:.2f}, "
        f"p99.99={pct_deg['p99.99']:.2f}. Recommended cap: {cap_str}."
    )
    for w in warnings:
        print(f"  ! {w}")
    print(
        f"Wrap-crossings (raw |diff| > 300 deg): {n_wrap_crossings}/"
        f"{n_diff_pairs} pairs ({wrap_pct:.3f}%)."
    )
    print(
        f"Per-flight outliers (p99.9 > 1.0 rad): {n_outlier_flights}/"
        f"{len(per_flight_p999)} flights."
    )
    print(f"Figures + stats -> {OUT_DIR}")


if __name__ == "__main__":
    main()
