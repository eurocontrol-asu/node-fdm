"""Stratified validation of `heading_clean` fallback (track - drift_ERA5).

Step 1.3 of `data/mardown/TODO/lateral_layer.md`. Computes the
reconciliation diff |bds_true - track_minus_drift| per flight phase
(climb / cruise / descent / approach) and per turn-state (straight /
in_turn) on the full Delta dataset. Outputs:

  data/figures/lateral_step1/heading_clean/
      recon_diff_by_phase.png       boxplot per phase x turn-state
      recon_diff_cdf_by_phase.png   CDF overlay (straight only)
      per_flight_median.png         per-flight median (cruise-straight)
      stats.json                    full table + outlier list
      outlier_flights.txt           flight ids w/ cruise-straight med > 5 deg

Usage:
    uv run python scripts/debug/heading_clean_stratified.py
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

# Reuse helpers from the reference script (same dir).
sys.path.insert(0, str(Path(__file__).parent))
from check_lateral import (  # noqa: E402
    KT_TO_MS,
    _decimal_year,
    _declination_series,
    _drift_from_wind,
    _prepare_for_lateral,
    _wrap_signed_deg,
)

from node_fdm_data.lateral import augment_lateral  # noqa: E402

PHASES = ["climb", "cruise", "descent", "approach"]
TURN_STATES = ["straight", "in_turn"]
OUTLIER_THRESHOLD_DEG = 5.0
P95_VALIDATED_DEG = 3.0
P95_DEGRADES_DEG = 5.0

OUT_DIR = Path("data/figures/lateral_step1/heading_clean")


def _classify_phase(alt_ft: np.ndarray, d_alt_ft_per_min: np.ndarray) -> np.ndarray:
    """Tag each sample with one phase label (str). Order matters: approach
    overrides descent below 10 000 ft.
    """
    n = alt_ft.size
    out = np.full(n, "other", dtype="<U10")
    finite = np.isfinite(alt_ft) & np.isfinite(d_alt_ft_per_min)
    climb = finite & (alt_ft < 25000.0) & (d_alt_ft_per_min > 200.0)
    descent = finite & (d_alt_ft_per_min < -200.0)
    cruise = finite & (alt_ft >= 25000.0) & (np.abs(d_alt_ft_per_min) <= 200.0)
    approach = finite & (alt_ft < 10000.0) & (d_alt_ft_per_min < -200.0)
    # Apply in order: climb, descent, cruise, approach (approach last so it wins).
    out[climb] = "climb"
    out[descent] = "descent"
    out[cruise] = "cruise"
    out[approach] = "approach"
    return out


def _rolling_d_alt_per_min(alt_ft: np.ndarray, window: int = 8) -> np.ndarray:
    """Rolling-mean smoothed d_alt/dt in ft/min (sampling = 4 s)."""
    if alt_ft.size < 2:
        return np.full_like(alt_ft, np.nan, dtype=np.float64)
    s = pl.Series(alt_ft.astype(np.float64))
    smooth = s.rolling_mean(window_size=window, min_samples=1).to_numpy()
    d = np.diff(smooth, prepend=smooth[0])
    # 4 s sampling -> per minute scale = 60/4 = 15
    return d * 15.0


def _process_flight(
    flight: pl.DataFrame, decl_stride: int = 8
) -> dict[str, np.ndarray] | None:
    """Compute per-sample arrays needed for the stratification, or None on skip."""
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

    # Magnetic declination on a strided grid (every decl_stride samples = 32 s)
    # then linearly interpolated. Declination drifts < 0.05 deg over ~100 km
    # so 32 s of cruise (~6 km) is well under that.
    ts_series = flight["raw_timestamp"]
    mid_ts = ts_series[ts_series.len() // 2]
    dec_year = _decimal_year(mid_ts)

    n = lat.size
    idx = np.arange(0, n, decl_stride, dtype=np.int64)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    decl_strided = _declination_series(lat[idx], lon[idx], alt_ft[idx], dec_year)
    # Interpolate (np.interp tolerates NaN by carrying through;
    # we'll mark NaN afterwards if either bracketing sample was NaN).
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
        "declination": declination,
        "lat": lat,
        "valid": valid,
    }


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return dict.fromkeys(("p50", "p75", "p90", "p95", "p99"), float("nan"))
    qs = np.percentile(arr, [50, 75, 90, 95, 99])
    return {
        "p50": float(qs[0]),
        "p75": float(qs[1]),
        "p90": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
    }


def _verdict(cell_stats: dict) -> str:
    p95 = {ph: cell_stats[(ph, "straight")]["p95"] for ph in PHASES}
    if not np.isfinite(p95["cruise"]):
        return "insufficient cruise data -- cannot evaluate"
    if p95["cruise"] > P95_DEGRADES_DEG:
        return "fallback unreliable (cruise p95 > 5 deg)"
    bad = []
    for ph in PHASES:
        v = p95[ph]
        if not np.isfinite(v):
            continue
        if v > P95_DEGRADES_DEG:
            bad.append(ph)
    if not bad and all(
        np.isfinite(p95[ph]) and p95[ph] <= P95_VALIDATED_DEG for ph in PHASES
    ):
        return "fallback validated across phases"
    if bad:
        return f"fallback degrades in {','.join(bad)}"
    return (
        "fallback acceptable in cruise but other phases between 3 and 5 deg "
        "(borderline; see per-phase numbers)"
    )


def main() -> None:
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

    flight_ids = df["meta_flight_id"].unique().to_list()
    print(f"  {len(flight_ids)} unique flights")

    # Storage: per (phase, in_turn) -> list of np arrays of recon_diff
    # and decl values. Per-flight: median in cruise-straight.
    bucket_recon: dict[tuple[str, str], list[np.ndarray]] = {
        (ph, ts): [] for ph in PHASES for ts in TURN_STATES
    }
    bucket_decl: dict[tuple[str, str], list[np.ndarray]] = {
        (ph, ts): [] for ph in PHASES for ts in TURN_STATES
    }
    bucket_lat: dict[tuple[str, str], list[np.ndarray]] = {
        (ph, ts): [] for ph in PHASES for ts in TURN_STATES
    }
    bucket_flights: dict[tuple[str, str], set[str]] = {
        (ph, ts): set() for ph in PHASES for ts in TURN_STATES
    }
    other_n = 0
    per_flight_cruise_straight_med: dict[str, float] = {}

    df_by_flight = df.partition_by("meta_flight_id", as_dict=True, maintain_order=False)
    n_flights = len(df_by_flight)
    print(f"Processing {n_flights} flights ...")

    t0 = time.time()
    for k, (key, flight) in enumerate(df_by_flight.items()):
        flight_id = key[0] if isinstance(key, tuple) else key
        if k % 50 == 0 and k > 0:
            elapsed = time.time() - t0
            eta = elapsed / k * (n_flights - k)
            print(f"  [{k}/{n_flights}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        out = _process_flight(flight)
        if out is None:
            continue
        valid = out["valid"]
        if not valid.any():
            continue
        recon = out["recon_diff"]
        phase = out["phase"]
        in_turn = out["in_turn"]
        decl = out["declination"]
        lat_arr = out["lat"]

        for ph in PHASES:
            for ts in TURN_STATES:
                turn_mask = in_turn if ts == "in_turn" else ~in_turn
                m = valid & (phase == ph) & turn_mask
                if m.any():
                    bucket_recon[(ph, ts)].append(recon[m])
                    bucket_decl[(ph, ts)].append(decl[m])
                    bucket_lat[(ph, ts)].append(lat_arr[m])
                    bucket_flights[(ph, ts)].add(flight_id)
        other_n += int((valid & (phase == "other")).sum())

        m_cs = valid & (phase == "cruise") & (~in_turn)
        if m_cs.sum() >= 30:
            per_flight_cruise_straight_med[flight_id] = float(
                np.median(recon[m_cs])
            )

    print(f"Done processing in {time.time() - t0:.1f}s")

    # ---- Aggregate stats ----
    cell_stats: dict = {}
    flat_recon: dict[tuple[str, str], np.ndarray] = {}
    flat_decl: dict[tuple[str, str], np.ndarray] = {}
    flat_lat: dict[tuple[str, str], np.ndarray] = {}
    for cell, arrs in bucket_recon.items():
        flat = np.concatenate(arrs) if arrs else np.array([])
        flat_recon[cell] = flat
        decls = (
            np.concatenate(bucket_decl[cell]) if bucket_decl[cell] else np.array([])
        )
        flat_decl[cell] = decls
        lats_ = (
            np.concatenate(bucket_lat[cell]) if bucket_lat[cell] else np.array([])
        )
        flat_lat[cell] = lats_
        cell_stats[cell] = {
            "n_samples": int(flat.size),
            "n_flights_contributing": len(bucket_flights[cell]),
            "mean_decl": float(np.median(decls)) if decls.size else float("nan"),
            **_percentiles(flat),
        }

    # Total per phase (sanity)
    per_phase_total = {
        ph: cell_stats[(ph, "straight")]["n_samples"]
        + cell_stats[(ph, "in_turn")]["n_samples"]
        for ph in PHASES
    }
    per_phase_total["other"] = other_n

    # Outlier flights
    outliers = sorted(
        fid
        for fid, med in per_flight_cruise_straight_med.items()
        if med > OUTLIER_THRESHOLD_DEG
    )

    verdict = _verdict(cell_stats)

    # ---- decl correlation in cruise-straight ----
    decl_corr_msg = "n/a"
    cs_lat = flat_lat[("cruise", "straight")]
    cs_rd = flat_recon[("cruise", "straight")]
    if cs_lat.size > 200:
        # Bin by |lat| in 10 deg bins, report median recon_diff per bin.
        abs_lat = np.abs(cs_lat)
        bins = [0, 30, 45, 60, 90]
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            m = (abs_lat >= lo) & (abs_lat < hi)
            if m.sum() > 100:
                rows.append((f"{lo}-{hi}", int(m.sum()), float(np.median(cs_rd[m]))))
        decl_corr_msg = "; ".join(
            f"|lat|in[{lo}): n={n}, med={med:.2f} deg" for (lo, n, med) in rows
        )

    # ---- Save stats.json ----
    stats_payload = {
        "verdict": verdict,
        "per_cell": {
            f"{ph}|{ts}": cell_stats[(ph, ts)]
            for ph in PHASES
            for ts in TURN_STATES
        },
        "per_phase_total_samples": per_phase_total,
        "outlier_flights_count": len(outliers),
        "outlier_flights_threshold_deg": OUTLIER_THRESHOLD_DEG,
        "outlier_flights": outliers,
        "decl_vs_recon_diff_lat_bins_cruise_straight": decl_corr_msg,
        "n_flights_with_cruise_straight": len(per_flight_cruise_straight_med),
    }
    with (OUT_DIR / "stats.json").open("w") as fh:
        json.dump(stats_payload, fh, indent=2)
    with (OUT_DIR / "outlier_flights.txt").open("w") as fh:
        fh.write("\n".join(outliers) + ("\n" if outliers else ""))

    # ---- Plot 1: boxplot ----
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = []
    data = []
    labels = []
    pos = 1
    colors = []
    for ph in PHASES:
        for ts in TURN_STATES:
            arr = flat_recon[(ph, ts)]
            if arr.size > 0:
                # Cap visualization at 90 deg to keep boxes readable.
                data.append(np.clip(arr, 0.0, 90.0))
                labels.append(f"{ph}\n{ts}")
                positions.append(pos)
                colors.append("tab:blue" if ts == "straight" else "tab:orange")
            pos += 1
        pos += 0.5
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.7,
        showfliers=False,
        patch_artist=True,
    )
    for patch, c in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|bds_true - track_minus_drift| [deg]")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Reconciliation diff per phase x turn-state (full Delta)")
    ax.axhline(P95_VALIDATED_DEG, color="green", ls="--", lw=0.8, label="3 deg")
    ax.axhline(P95_DEGRADES_DEG, color="red", ls="--", lw=0.8, label="5 deg")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recon_diff_by_phase.png", dpi=120)
    plt.close(fig)

    # ---- Plot 2: CDF overlay (straight only) ----
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {
        "climb": "tab:blue",
        "cruise": "tab:green",
        "descent": "tab:orange",
        "approach": "tab:red",
    }
    for ph in PHASES:
        arr = flat_recon[(ph, "straight")]
        if arr.size == 0:
            continue
        x_sorted = np.sort(arr)
        y = np.arange(1, x_sorted.size + 1) / x_sorted.size
        ax.plot(x_sorted, y, lw=1.5, color=palette[ph], label=f"{ph} (n={arr.size:,})")
    ax.set_xlim(0, 30)
    ax.set_xlabel("|bds_true - track_minus_drift| [deg]")
    ax.set_ylabel("CDF")
    ax.set_title("Reconciliation CDF -- straight segments only")
    ax.axvline(P95_VALIDATED_DEG, color="green", ls="--", lw=0.8, label="3 deg")
    ax.axvline(P95_DEGRADES_DEG, color="red", ls="--", lw=0.8, label="5 deg")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recon_diff_cdf_by_phase.png", dpi=120)
    plt.close(fig)

    # ---- Plot 3: per-flight median (cruise-straight) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    if per_flight_cruise_straight_med:
        vals = np.array(list(per_flight_cruise_straight_med.values()))
        ax.hist(np.clip(vals, 0, 30), bins=60, color="tab:blue", alpha=0.7)
        ax.axvline(
            OUTLIER_THRESHOLD_DEG,
            color="red",
            ls="--",
            label=f"{OUTLIER_THRESHOLD_DEG} deg threshold",
        )
        ax.axvline(
            float(np.median(vals)),
            color="black",
            ls=":",
            label=f"dataset median = {np.median(vals):.2f} deg",
        )
        ax.set_xlabel("Per-flight median |recon_diff| in cruise-straight [deg]")
        ax.set_ylabel("flights")
        ax.set_title(
            f"Per-flight median (n={len(vals)} flights, "
            f"{len(outliers)} > {OUTLIER_THRESHOLD_DEG} deg)"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_flight_median.png", dpi=120)
    plt.close(fig)

    # ---- Console summary ----
    print()
    print("=" * 72)
    print("HEADING_CLEAN STRATIFIED VALIDATION -- summary")
    print("=" * 72)
    print(f"Verdict: {verdict}")
    print()
    print("Per-cell stats (recon_diff in deg):")
    print(
        f"{'phase':<10} {'state':<10} {'n':>10} {'flights':>8} "
        f"{'p50':>7} {'p90':>7} {'p95':>7} {'p99':>7}  decl_med"
    )
    for ph in PHASES:
        for ts in TURN_STATES:
            s = cell_stats[(ph, ts)]
            print(
                f"{ph:<10} {ts:<10} {s['n_samples']:>10,} "
                f"{s['n_flights_contributing']:>8} "
                f"{s['p50']:>7.2f} {s['p90']:>7.2f} "
                f"{s['p95']:>7.2f} {s['p99']:>7.2f}  "
                f"{s['mean_decl']:.2f}"
            )
    print()
    print("Per-phase total samples (sanity):")
    for ph in PHASES:
        print(f"  {ph:<10}: {per_phase_total[ph]:>12,}")
    print(f"  {'other':<10}: {per_phase_total['other']:>12,}")
    print()
    print(f"Outlier flights (cruise-straight median > {OUTLIER_THRESHOLD_DEG} deg): "
          f"{len(outliers)}")
    print(f"Decl/lat correlation (cruise-straight): {decl_corr_msg}")
    print()
    print(f"Wrote: {OUT_DIR}/")
    print("  - recon_diff_by_phase.png")
    print("  - recon_diff_cdf_by_phase.png")
    print("  - per_flight_median.png")
    print("  - stats.json")
    print("  - outlier_flights.txt")


if __name__ == "__main__":
    main()
