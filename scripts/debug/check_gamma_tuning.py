"""Compare gamma target: current (backfilled) vs NaN-preserving with new priority.

Left:  current config (sm=5, tol=0.115°, backfill, alt_hold priority max)
Right: new approach (sm=15, tol=0.2°, abs>0.5°, NaN where no info,
       priority: alt_hold < vz→γ < gamma_sel, diff=0 where NaN)

Usage:
    uv run python scripts/check_gamma_tuning.py [FLIGHT_ID]
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from deltalake import DeltaTable

from node_fdm_data.physics.speed import vz_to_gamma
from node_fdm_data.segments import detect_constant_segments

DELTA_PATH = "data/flights.delta"
DEFAULT_FLIGHT = "40666a_EZY58HC_s0"


def load_flight(flight_id: str) -> pl.DataFrame:
    dt = DeltaTable(DELTA_PATH)
    df = pl.from_arrow(dt.to_pyarrow_dataset().to_table())
    f = df.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
    if len(f) == 0:
        print(f"Flight {flight_id!r} not found", file=sys.stderr)
        raise SystemExit(1)
    return f


def to_np(s: pl.Series) -> np.ndarray:
    return s.to_numpy(allow_copy=True).astype(np.float64)


def build_gamma_target_new(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Build gamma target with new approach: NaN-preserving, new priority.

    Priority (last overwrites):
      P1 (lowest):  alt_sel detected → γ = 0
      P2:           vz_sel detected  → γ = arcsin(vz_sel / TAS)
      P3 (highest): gamma_sel detected → γ = gamma_sel (direct)

    No backfill — NaN where no segment detected.
    """
    n = len(df)
    gamma_target = np.full(n, np.nan)

    # Detect gamma segments with tuned config
    gamma_raw = df["fdm_gamma_rad"].to_numpy() if "fdm_gamma_rad" in df.columns else None
    gamma_segs_arr = np.full(n, np.nan)
    if gamma_raw is not None:
        segs = detect_constant_segments(
            gamma_raw, tol=0.0035, min_len=15, use_alt=False,
            smooth_window=15, smooth_method="savgol", min_abs_value=0.0087,
        )
        for s in segs:
            gamma_segs_arr[s["start_idx"] : s["end_idx"] + 1] = s["var_mean"]

    # Detect vz segments (same as current config)
    vz_raw = df["raw_vz_ftmin"].to_numpy() if "raw_vz_ftmin" in df.columns else None
    vz_segs_arr = np.full(n, np.nan)
    if vz_raw is not None:
        vz_segs = detect_constant_segments(
            vz_raw, tol=25, min_len=25, use_alt=False,
            min_abs_value=75, smooth_window=15, smooth_method="savgol",
        )
        for s in vz_segs:
            vz_segs_arr[s["start_idx"] : s["end_idx"] + 1] = s["var_mean"]

    # Detect alt segments (same as current config)
    alt_col = "raw_alt_ft" if "raw_alt_ft" in df.columns else "altitude"
    alt_raw = df[alt_col].to_numpy() if alt_col in df.columns else None
    alt_segs_arr = np.full(n, np.nan)
    if alt_raw is not None:
        alt_segs = detect_constant_segments(
            alt_raw, tol=25, min_len=5, use_alt=False,
            min_abs_value=25, smooth_window=5, smooth_method="savgol",
        )
        for s in alt_segs:
            alt_segs_arr[s["start_idx"] : s["end_idx"] + 1] = s["var_mean"]

    # Layer 1 (lowest priority): alt_hold → γ = 0
    mask_alt = ~np.isnan(alt_segs_arr)
    gamma_target[mask_alt] = 0.0

    # Layer 2: gamma_sel direct
    mask_gamma = ~np.isnan(gamma_segs_arr)
    gamma_target[mask_gamma] = gamma_segs_arr[mask_gamma]

    # Layer 3 (highest priority): vz_sel → γ = arcsin(vz / TAS)
    mask_vz = ~np.isnan(vz_segs_arr)
    if mask_vz.any():
        tas_col = "era_tas_kt" if "era_tas_kt" in df.columns else None
        if tas_col:
            tas_kt = df[tas_col].to_numpy()
            tas_ms = tas_kt[mask_vz] * 0.514444
            vz_ms = vz_segs_arr[mask_vz] * 0.00508  # ft/min → m/s
            gamma_target[mask_vz] = vz_to_gamma(vz_ms, tas_ms)

    # NO backfill — NaN stays NaN

    # Gamma diff: 0 where NaN target, target - actual otherwise
    gamma_actual = gamma_raw if gamma_raw is not None else np.zeros(n)
    gamma_diff = np.where(np.isnan(gamma_target), 0.0, gamma_target - gamma_actual)

    return {
        "gamma_target": gamma_target,
        "gamma_diff": gamma_diff,
        "gamma_sel": gamma_segs_arr,
        "vz_gamma": np.where(mask_vz, gamma_target, np.nan) if mask_vz.any() else np.full(n, np.nan),
        "alt_hold": np.where(mask_alt, 0.0, np.nan),
    }


def main() -> None:
    flight_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FLIGHT
    raw = load_flight(flight_id)
    n = len(raw)
    t = np.arange(n) * 4.0 / 60.0

    r2d = np.degrees

    alt_ft = to_np(raw["raw_alt_ft"])
    vz_ftmin = to_np(raw["raw_vz_ftmin"])
    gamma = to_np(raw["fdm_gamma_rad"]) if "fdm_gamma_rad" in raw.columns else None

    # Current: from preprocessed data
    gamma_target_cur = to_np(raw["fdm_gamma_target_rad"]) if "fdm_gamma_target_rad" in raw.columns else None
    gamma_sel_cur = to_np(raw["fdm_gamma_sel_rad"]) if "fdm_gamma_sel_rad" in raw.columns else None
    gamma_from_alt_cur = to_np(raw["fdm_gamma_from_alt_rad"]) if "fdm_gamma_from_alt_rad" in raw.columns else None
    alt_sel = to_np(raw["fdm_alt_sel_ft"]) if "fdm_alt_sel_ft" in raw.columns else None
    vz_sel = to_np(raw["fdm_vz_sel_ftmin"]) if "fdm_vz_sel_ftmin" in raw.columns else None
    alt_target = to_np(raw["fdm_alt_target_ft"]) if "fdm_alt_target_ft" in raw.columns else None

    gamma_diff_cur = None
    if gamma_target_cur is not None and gamma is not None:
        gamma_diff_cur = gamma_target_cur - gamma

    # New approach
    new = build_gamma_target_new(raw)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 2, figsize=(22, 18), sharex=True)
    fig.suptitle(f"Gamma Target — Current vs NaN-preserving — {flight_id}",
                 fontsize=14, fontweight="bold")

    labels = [
        "Current (bfill, alt>γ>vz)",
        "New (NaN, alt>γ>vz, abs>0.5°)",
    ]

    for col_idx in range(2):
        # 1) Altitude
        ax = axes[0, col_idx]
        ax.plot(t, alt_ft / 100, color="black", linewidth=0.8, label="Altitude (FL)")
        if alt_target is not None:
            ax.plot(t, alt_target / 100, color="blue", linewidth=2.0, alpha=0.7, label="Alt target")
        if alt_sel is not None:
            mask = ~np.isnan(alt_sel)
            ax.scatter(t[mask], alt_sel[mask] / 100, color="green", s=3, alpha=0.5, label="Alt sel")
        ax.set_ylabel("FL")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"{labels[col_idx]}\nAltitude", fontsize=10)
        ax.grid(True, alpha=0.3)

        # 2) Vz
        ax = axes[1, col_idx]
        ax.plot(t, vz_ftmin, color="black", linewidth=0.5, alpha=0.6, label="Vz")
        if vz_sel is not None:
            mask = ~np.isnan(vz_sel)
            ax.scatter(t[mask], vz_sel[mask], color="red", s=4, alpha=0.7, label="Vz sel")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Vz (ft/min)")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Vertical Speed", fontsize=10)
        ax.grid(True, alpha=0.3)

        # 3) Gamma sources
        ax = axes[2, col_idx]
        if gamma is not None:
            ax.plot(t, r2d(gamma), color="black", linewidth=0.5, alpha=0.5, label="γ actual")

        if col_idx == 0:
            # Current sources
            if gamma_from_alt_cur is not None:
                m = ~np.isnan(gamma_from_alt_cur)
                ax.scatter(t[m], r2d(gamma_from_alt_cur[m]), color="orange", s=5, alpha=0.7,
                           label="γ from Vz (P3)", zorder=3)
            if gamma_sel_cur is not None:
                m = ~np.isnan(gamma_sel_cur)
                ax.scatter(t[m], r2d(gamma_sel_cur[m]), color="purple", s=5, alpha=0.7,
                           label="γ_sel (P2)", zorder=4)
            if alt_sel is not None:
                m = ~np.isnan(alt_sel)
                ax.scatter(t[m], np.zeros(m.sum()), color="green", s=5, alpha=0.5,
                           label="γ=0 hold (P1 max)", zorder=5)
        else:
            # New sources (reversed priority labels)
            alt_hold = new["alt_hold"]
            m = ~np.isnan(alt_hold)
            ax.scatter(t[m], np.zeros(m.sum()), color="green", s=5, alpha=0.5,
                       label="γ=0 hold (P1 min)", zorder=3)
            gs = new["gamma_sel"]
            m = ~np.isnan(gs)
            ax.scatter(t[m], r2d(gs[m]), color="purple", s=5, alpha=0.7,
                       label="γ_sel (P2)", zorder=4)
            vz_g = new["vz_gamma"]
            m = ~np.isnan(vz_g)
            if m.any():
                ax.scatter(t[m], r2d(vz_g[m]), color="orange", s=5, alpha=0.7,
                           label="γ from Vz (P3 max)", zorder=5)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("γ (°)")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Gamma sources", fontsize=10)
        ax.grid(True, alpha=0.3)

        # 4) Gamma target vs actual
        ax = axes[3, col_idx]
        if gamma is not None:
            ax.plot(t, r2d(gamma), color="black", linewidth=0.5, alpha=0.5, label="γ actual")

        if col_idx == 0:
            if gamma_target_cur is not None:
                ax.plot(t, r2d(gamma_target_cur), color="blue", linewidth=2.0, alpha=0.8,
                        label="γ target (bfill)")
        else:
            gt = new["gamma_target"]
            # Plot NaN gaps as grey background
            is_nan = np.isnan(gt)
            if is_nan.any():
                ax.fill_between(t, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -5,
                                ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 8,
                                where=is_nan, color="lightgray", alpha=0.3, label="NaN (no info)")
            valid = ~is_nan
            # Plot target as step-like segments
            gt_plot = gt.copy()
            gt_plot[is_nan] = np.nan  # matplotlib skips NaN
            ax.plot(t, r2d(gt_plot), color="blue", linewidth=2.5, alpha=0.9,
                    label="γ target (NaN gaps)")

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("γ (°)")
        ax.set_ylim(-6, 8)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Gamma target unifié", fontsize=10)
        ax.grid(True, alpha=0.3)

        # 5) Gamma diff
        ax = axes[4, col_idx]
        if col_idx == 0:
            if gamma_diff_cur is not None:
                ax.plot(t, r2d(gamma_diff_cur), color="red", linewidth=0.8, alpha=0.8, label="Δγ")
                ax.fill_between(t, 0, r2d(gamma_diff_cur), alpha=0.15, color="red")
        else:
            gd = new["gamma_diff"]
            ax.plot(t, r2d(gd), color="red", linewidth=0.8, alpha=0.8, label="Δγ (0 where NaN)")
            ax.fill_between(t, 0, r2d(gd), alpha=0.15, color="red")

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Δγ (°)")
        ax.set_xlabel("Time (min)")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Gamma diff (erreur ODE)", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"data/figures/gamma_tuning_{flight_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
