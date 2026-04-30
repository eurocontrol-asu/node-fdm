"""Bimodality diagnostic for `recon_diff` outside cruise.

Step 1.3 follow-up of `data/mardown/TODO/lateral_layer.md`. The stratified
report shows median(recon_diff) ~1.3 deg in all phases, but p95 explodes
in climb / descent / approach (76 / 120 / 116 deg). Question: are bad
samples isolated jitter (recoverable by a median filter) or continuous
runs (real drift-model bug)?

For each non-cruise phase we compute recon_diff over a flight, then
identify contiguous runs where ``recon_diff > 10 deg`` and aggregate
their lengths dataset-wide. We also separate "good" (< 3 deg) and
"bad" (> 10 deg) samples and compare 4 covariates: TAS-rolling-std,
|cross_wind|, wind variance, |d_alt/dt|.

Outputs (in ``data/figures/lateral_step1/heading_clean/bimodality/``):
  - run_length_histogram.png
  - covariate_comparison.png
  - bimodality_stats.json
  - top_bad_flights.txt

Usage:
    uv run python scripts/debug/recon_diff_bimodality.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from deltalake import DeltaTable

sys.path.insert(0, str(Path(__file__).parent))
from check_lateral import (  # noqa: E402
    KT_TO_MS,
    _decimal_year,
    _declination_series,
    _drift_from_wind,
    _prepare_for_lateral,
    _wrap_signed_deg,
)
from heading_clean_stratified import (  # noqa: E402
    _classify_phase,
    _rolling_d_alt_per_min,
)

from node_fdm_data.lateral import augment_lateral  # noqa: E402

PHASES_NONCRUISE = ["climb", "descent", "approach"]
BAD_THRESHOLD_DEG = 10.0
GOOD_THRESHOLD_DEG = 3.0
RUN_BINS = [(1, 1), (2, 2), (3, 3), (4, 10), (11, 30), (31, 10**9)]
RUN_BIN_LABELS = ["1", "2", "3", "4-10", "11-30", ">30"]
ROLL_WIN = 8  # 32 s rolling window

OUT_DIR = Path(
    "data/figures/lateral_step1/heading_clean/bimodality"
)


def _runs_above(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) for contiguous True runs."""
    if mask.size == 0 or not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _bin_run_length(length: int) -> int:
    for i, (lo, hi) in enumerate(RUN_BINS):
        if lo <= length <= hi:
            return i
    return len(RUN_BINS) - 1


def _compute_per_flight(  # noqa: C901
    flight: pl.DataFrame, decl_stride: int = 8
) -> dict | None:
    """Return per-sample arrays for non-cruise diagnostics, or None on skip."""
    if flight.height < 30:
        return None
    flight = _prepare_for_lateral(flight.sort("raw_timestamp"))
    try:
        flight = augment_lateral(flight)
    except Exception as exc:  # noqa: BLE001
        print(f"  augment_lateral failed: {exc}", file=sys.stderr)
        return None

    lat = flight["latitude"].to_numpy().astype(np.float64)
    lon = flight["longitude"].to_numpy().astype(np.float64)
    track = flight["track"].to_numpy().astype(np.float64)
    heading = flight["heading"].to_numpy().astype(np.float64)
    in_turn = flight["in_turn"].to_numpy().astype(bool)
    alt_ft = flight["raw_alt_ft"].to_numpy().astype(np.float64)
    tas_kt = flight["TAS"].to_numpy().astype(np.float64)
    u_wind = flight["era_u_wind_ms"].to_numpy().astype(np.float64)
    v_wind = flight["era_v_wind_ms"].to_numpy().astype(np.float64)

    drift_wind = _drift_from_wind(heading, tas_kt * KT_TO_MS, u_wind, v_wind)

    ts_series = flight["raw_timestamp"]
    mid_ts = ts_series[ts_series.len() // 2]
    dec_year = _decimal_year(mid_ts)

    n = lat.size
    idx = np.arange(0, n, decl_stride, dtype=np.int64)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    decl_strided = _declination_series(lat[idx], lon[idx], alt_ft[idx], dec_year)
    full_idx = np.arange(n, dtype=np.float64)
    valid_mask_strided = np.isfinite(decl_strided)
    if valid_mask_strided.sum() < 2:
        declination = np.full(n, np.nan, dtype=np.float64)
    else:
        declination = np.interp(
            full_idx,
            idx[valid_mask_strided].astype(np.float64),
            decl_strided[valid_mask_strided],
        )

    bds_true = _wrap_signed_deg(heading + declination)
    track_minus_drift = _wrap_signed_deg(track - drift_wind)
    recon_diff = np.abs(_wrap_signed_deg(bds_true - track_minus_drift))

    d_alt = _rolling_d_alt_per_min(alt_ft)
    phase = _classify_phase(alt_ft, d_alt)

    # ---- covariates --------------------------------------------------
    # 1) TAS rolling std (kt) over 32 s
    tas_s = pl.Series(tas_kt)
    tas_std = tas_s.rolling_std(window_size=ROLL_WIN, min_samples=2).to_numpy()
    # 2) |cross_wind| (m/s) using heading
    psi = np.radians(heading)
    cross_wind = np.abs(u_wind * np.cos(psi) - v_wind * np.sin(psi))
    # 3) wind variance: u_std + v_std (m/s) over 32 s
    u_std = pl.Series(u_wind).rolling_std(window_size=ROLL_WIN, min_samples=2).to_numpy()
    v_std = pl.Series(v_wind).rolling_std(window_size=ROLL_WIN, min_samples=2).to_numpy()
    wind_std = u_std + v_std
    # 4) |d_alt/dt| (ft/min) magnitude
    d_alt_mag = np.abs(d_alt)

    valid = (
        np.isfinite(heading)
        & np.isfinite(track)
        & np.isfinite(tas_kt)
        & np.isfinite(u_wind)
        & np.isfinite(v_wind)
        & np.isfinite(drift_wind)
        & np.isfinite(declination)
        & np.isfinite(recon_diff)
    )

    return {
        "recon_diff": recon_diff,
        "phase": phase,
        "in_turn": in_turn,
        "valid": valid,
        "tas_std": tas_std,
        "cross_wind": cross_wind,
        "wind_std": wind_std,
        "d_alt_mag": d_alt_mag,
    }


def _accumulate_runs(
    recon: np.ndarray, phase_mask: np.ndarray, valid: np.ndarray
) -> tuple[list[int], int, int]:
    """Per-flight per-phase run aggregation.

    Returns (run_lengths, n_bad_total, n_valid_total) where bad = recon>10
    among valid samples in this phase.
    """
    eligible = valid & phase_mask
    if not eligible.any():
        return [], 0, 0
    bad = eligible & (recon > BAD_THRESHOLD_DEG)
    runs = _runs_above(bad)
    lengths = [end - start for start, end in runs]
    return lengths, int(bad.sum()), int(eligible.sum())


def _percentile_dict(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {k: float("nan") for k in ("p50", "p75", "p90", "p95")}
    qs = np.percentile(arr, [50, 75, 90, 95])
    return {"p50": float(qs[0]), "p75": float(qs[1]),
            "p90": float(qs[2]), "p95": float(qs[3])}


def _verdict_from_runs(stats: dict) -> str:
    """Aggregate verdict across non-cruise phases (sample-weighted)."""
    short_bad = 0
    long_bad = 0
    total_bad = 0
    for ph in PHASES_NONCRUISE:
        s = stats[ph]
        short_bad += s["bad_samples_in_runs_le3"]
        long_bad += s["bad_samples_in_runs_gt10"]
        total_bad += s["n_bad_samples"]
    if total_bad == 0:
        return "no_bad_samples"
    short_frac = short_bad / total_bad
    long_frac = long_bad / total_bad
    if short_frac > 0.70:
        return "isolated_jitter"
    if long_frac > 0.50:
        return "continuous_pattern"
    return "mixed"


def main() -> None:  # noqa: C901
    delta_path = Path("data/flights.delta")
    if not delta_path.exists():
        print(f"Delta table not found: {delta_path}", file=sys.stderr)
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {delta_path} ...")
    t0 = time.time()
    dt = DeltaTable(str(delta_path))
    df = pl.DataFrame(dt.to_pyarrow_table())
    print(f"  loaded {df.height:,} rows in {time.time() - t0:.1f}s")

    df_by_flight = df.partition_by(
        "meta_flight_id", as_dict=True, maintain_order=False
    )
    n_flights = len(df_by_flight)
    print(f"Processing {n_flights} flights ...")

    # Per-phase aggregators
    run_lengths: dict[str, list[int]] = {ph: [] for ph in PHASES_NONCRUISE}
    n_bad: dict[str, int] = dict.fromkeys(PHASES_NONCRUISE, 0)
    n_valid: dict[str, int] = dict.fromkeys(PHASES_NONCRUISE, 0)
    # bad samples by run-length bin
    bad_by_bin: dict[str, list[int]] = {
        ph: [0] * len(RUN_BINS) for ph in PHASES_NONCRUISE
    }
    # per-flight: time spent in long (>10) runs in each phase
    flight_long_samples: dict[str, dict[str, int]] = {
        ph: {} for ph in PHASES_NONCRUISE
    }
    # covariate samples (concatenated across phases, non-cruise only)
    cov_good: dict[str, list[np.ndarray]] = {
        k: [] for k in ("tas_std", "cross_wind", "wind_std", "d_alt_mag")
    }
    cov_bad: dict[str, list[np.ndarray]] = {
        k: [] for k in ("tas_std", "cross_wind", "wind_std", "d_alt_mag")
    }

    t0 = time.time()
    for k, (key, flight) in enumerate(df_by_flight.items()):
        flight_id = key[0] if isinstance(key, tuple) else key
        if k % 50 == 0 and k > 0:
            elapsed = time.time() - t0
            eta = elapsed / k * (n_flights - k)
            print(f"  [{k}/{n_flights}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        out = _compute_per_flight(flight)
        if out is None:
            continue
        valid = out["valid"]
        if not valid.any():
            continue
        recon = out["recon_diff"]
        phase = out["phase"]

        for ph in PHASES_NONCRUISE:
            ph_mask = phase == ph
            lengths, n_bad_ph, n_valid_ph = _accumulate_runs(recon, ph_mask, valid)
            run_lengths[ph].extend(lengths)
            n_bad[ph] += n_bad_ph
            n_valid[ph] += n_valid_ph
            # accumulate by bin
            long_samples = 0
            for length in lengths:
                bin_idx = _bin_run_length(length)
                bad_by_bin[ph][bin_idx] += length
                if length > 10:
                    long_samples += length
            if long_samples > 0:
                flight_long_samples[ph][flight_id] = long_samples

        # covariates: pool across non-cruise phases
        non_cruise_mask = (
            valid & ((phase == "climb") | (phase == "descent") | (phase == "approach"))
        )
        if non_cruise_mask.any():
            good_m = non_cruise_mask & (recon < GOOD_THRESHOLD_DEG)
            bad_m = non_cruise_mask & (recon > BAD_THRESHOLD_DEG)
            for cov_name in ("tas_std", "cross_wind", "wind_std", "d_alt_mag"):
                arr = out[cov_name]
                # subsample to keep memory bounded
                g = arr[good_m & np.isfinite(arr)]
                b = arr[bad_m & np.isfinite(arr)]
                if g.size > 5000:
                    rng = np.random.default_rng(42)
                    g = rng.choice(g, size=5000, replace=False)
                cov_good[cov_name].append(g)
                cov_bad[cov_name].append(b)

    print(f"Done processing in {time.time() - t0:.1f}s")

    # ---- Aggregate run stats ----
    runs_per_phase: dict = {}
    for ph in PHASES_NONCRUISE:
        lengths = np.array(run_lengths[ph], dtype=np.int64)
        bins_counts = bad_by_bin[ph]
        # samples in runs <=3 = sum of bins 0,1,2
        bad_le3 = bins_counts[0] + bins_counts[1] + bins_counts[2]
        # samples in runs >10 = bins 4,5
        bad_gt10 = bins_counts[4] + bins_counts[5]
        total_bad = sum(bins_counts)
        runs_per_phase[ph] = {
            "n_runs": int(lengths.size),
            "n_bad_samples": int(total_bad),
            "n_valid_samples": int(n_valid[ph]),
            "bad_fraction": (
                float(total_bad / n_valid[ph]) if n_valid[ph] else 0.0
            ),
            "run_length_percentiles": _percentile_dict(lengths.astype(np.float64)),
            "run_length_max": int(lengths.max()) if lengths.size else 0,
            "samples_by_run_bin": dict(zip(RUN_BIN_LABELS, bins_counts, strict=True)),
            "bad_samples_in_runs_le3": int(bad_le3),
            "bad_samples_in_runs_gt10": int(bad_gt10),
            "frac_in_runs_le3": (
                float(bad_le3 / total_bad) if total_bad else 0.0
            ),
            "frac_in_runs_gt10": (
                float(bad_gt10 / total_bad) if total_bad else 0.0
            ),
        }

    verdict = _verdict_from_runs(runs_per_phase)

    # ---- Covariate separation ----
    cov_separation: dict = {}
    cov_data_for_plot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cov_name in ("tas_std", "cross_wind", "wind_std", "d_alt_mag"):
        good_arr = (
            np.concatenate(cov_good[cov_name])
            if cov_good[cov_name]
            else np.array([])
        )
        bad_arr = (
            np.concatenate(cov_bad[cov_name])
            if cov_bad[cov_name]
            else np.array([])
        )
        good_arr = good_arr[np.isfinite(good_arr)]
        bad_arr = bad_arr[np.isfinite(bad_arr)]
        cov_data_for_plot[cov_name] = (good_arr, bad_arr)
        g_pcts = _percentile_dict(good_arr)
        b_pcts = _percentile_dict(bad_arr)
        ratio_p50 = (
            float(b_pcts["p50"] / g_pcts["p50"])
            if g_pcts["p50"] not in (0.0, float("nan")) and np.isfinite(g_pcts["p50"])
            else float("nan")
        )
        cov_separation[cov_name] = {
            "n_good": int(good_arr.size),
            "n_bad": int(bad_arr.size),
            "good": g_pcts,
            "bad": b_pcts,
            "ratio_p50_bad_over_good": ratio_p50,
        }

    # ---- Top bad flights per phase ----
    top_bad: dict[str, list[tuple[str, int]]] = {}
    for ph in PHASES_NONCRUISE:
        ranked = sorted(
            flight_long_samples[ph].items(), key=lambda kv: kv[1], reverse=True
        )
        top_bad[ph] = ranked[:10]

    # ---- Save JSON ----
    payload = {
        "verdict": verdict,
        "thresholds": {
            "bad_deg": BAD_THRESHOLD_DEG,
            "good_deg": GOOD_THRESHOLD_DEG,
        },
        "runs_per_phase": runs_per_phase,
        "covariate_separation": cov_separation,
        "top_bad_flights": {
            ph: [{"flight_id": fid, "samples_in_runs_gt10": n}
                 for fid, n in top_bad[ph]]
            for ph in PHASES_NONCRUISE
        },
    }
    with (OUT_DIR / "bimodality_stats.json").open("w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- top_bad_flights.txt ----
    lines: list[str] = []
    for ph in PHASES_NONCRUISE:
        lines.append(f"# Phase: {ph} -- top 10 flights by samples in runs > 10")
        lines.append(f"# {'flight_id':<28} {'samples_in_long_runs':>20}")
        for fid, n in top_bad[ph]:
            lines.append(f"  {fid:<28} {n:>20d}")
        lines.append("")
    with (OUT_DIR / "top_bad_flights.txt").open("w") as fh:
        fh.write("\n".join(lines))

    # ---- Plot 1: run length histogram ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, ph in zip(axes, PHASES_NONCRUISE, strict=True):
        counts = bad_by_bin[ph]
        total = sum(counts) or 1
        fracs = [c / total for c in counts]
        bars = ax.bar(RUN_BIN_LABELS, fracs, color="tab:red", alpha=0.7)
        ax.set_title(
            f"{ph}\nn_runs={runs_per_phase[ph]['n_runs']:,}  "
            f"bad_frac={runs_per_phase[ph]['bad_fraction']*100:.1f}%",
            fontsize=10,
        )
        ax.set_xlabel("Run length [samples (4 s)]")
        ax.set_ylabel("Fraction of bad samples")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, frac, c in zip(bars, fracs, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{c:,}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylim(0, max(0.05, max(fracs) * 1.25 if fracs else 1))
    fig.suptitle(
        f"recon_diff > {BAD_THRESHOLD_DEG} deg -- run-length distribution "
        f"by phase  [verdict: {verdict}]",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "run_length_histogram.png", dpi=120)
    plt.close(fig)

    # ---- Plot 2: covariate boxplots good vs bad ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    cov_titles = {
        "tas_std": "TAS rolling std (32 s) [kt]",
        "cross_wind": "|cross_wind| [m/s]",
        "wind_std": "wind u+v rolling std (32 s) [m/s]",
        "d_alt_mag": "|d_alt/dt| [ft/min]",
    }
    for ax, cov_name in zip(axes, cov_titles, strict=True):
        good_arr, bad_arr = cov_data_for_plot[cov_name]
        if good_arr.size == 0 and bad_arr.size == 0:
            ax.set_title(f"{cov_titles[cov_name]}\n(no data)")
            continue
        bp = ax.boxplot(
            [good_arr, bad_arr],
            tick_labels=["good (<3 deg)", "bad (>10 deg)"],
            showfliers=False,
            patch_artist=True,
        )
        for patch, c in zip(bp["boxes"], ["tab:green", "tab:red"], strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        s = cov_separation[cov_name]
        ax.set_title(
            f"{cov_titles[cov_name]}\n"
            f"p50_good={s['good']['p50']:.2f}  p50_bad={s['bad']['p50']:.2f}  "
            f"ratio={s['ratio_p50_bad_over_good']:.2f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(
        "Covariate comparison: good (<3 deg) vs bad (>10 deg) -- "
        "non-cruise samples",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "covariate_comparison.png", dpi=120)
    plt.close(fig)

    # ---- Console summary ----
    print()
    print("=" * 72)
    print("BIMODALITY DIAGNOSTIC -- summary")
    print("=" * 72)
    print(f"Verdict: {verdict}")
    print()
    print("Per-phase run stats:")
    for ph in PHASES_NONCRUISE:
        s = runs_per_phase[ph]
        print(
            f"  {ph:<10} n_runs={s['n_runs']:>6,}  bad={s['n_bad_samples']:>7,} "
            f"({s['bad_fraction']*100:5.2f}%)  "
            f"len_p50={s['run_length_percentiles']['p50']:.0f}  "
            f"len_p95={s['run_length_percentiles']['p95']:.0f}  "
            f"max={s['run_length_max']}  "
            f"frac_le3={s['frac_in_runs_le3']*100:.1f}%  "
            f"frac_gt10={s['frac_in_runs_gt10']*100:.1f}%"
        )
    print()
    print("Covariate separation (p50_bad / p50_good):")
    for cov_name, s in cov_separation.items():
        print(
            f"  {cov_name:<14} good_p50={s['good']['p50']:>8.2f}  "
            f"bad_p50={s['bad']['p50']:>8.2f}  "
            f"ratio={s['ratio_p50_bad_over_good']:.2f}"
        )
    print()
    print(f"Wrote: {OUT_DIR}/")
    print("  - run_length_histogram.png")
    print("  - covariate_comparison.png")
    print("  - bimodality_stats.json")
    print("  - top_bad_flights.txt")


if __name__ == "__main__":
    main()
