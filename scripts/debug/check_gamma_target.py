"""Diagnostic plot: gamma target reconciliation from 3 sources.

Shows 5 subplots:
1. Altitude + alt_target + alt_sel (plateau detection)
2. Vz (actual) + vz_sel segments
3. Gamma (actual) + gamma_sel + gamma_from_alt (vz→γ conversion)
4. Gamma target (unified) + gamma actual — the final reconciled signal
5. Gamma diff (error signal fed to ODE)

Usage:
    uv run python scripts/check_gamma_target.py [FLIGHT_ID]
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from deltalake import DeltaTable

# ── Config ──────────────────────────────────────────────────────────────────
DELTA_PATH = "data/flights.delta"
DEFAULT_FLIGHT = "40666a_EZY58HC_s0"
FT_TO_M = 0.3048
FTMIN_TO_MS = 0.00508


def load_flight(flight_id: str) -> pl.DataFrame:
    dt = DeltaTable(DELTA_PATH)
    df = pl.from_arrow(dt.to_pyarrow_dataset().to_table())
    f = df.filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
    if len(f) == 0:
        print(f"Flight {flight_id!r} not found", file=sys.stderr)
        raise SystemExit(1)
    return f


def to_np(series: pl.Series) -> np.ndarray:
    return series.to_numpy(allow_copy=True).astype(np.float64)


def main() -> None:
    flight_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FLIGHT
    f = load_flight(flight_id)
    n = len(f)
    t = np.arange(n) * 4.0  # 4s sampling → seconds
    t_min = t / 60.0  # → minutes for x-axis

    # ── Extract columns ─────────────────────────────────────────────────
    alt_ft = to_np(f["raw_alt_ft"])
    alt_target_ft = to_np(f["fdm_alt_target_ft"]) if "fdm_alt_target_ft" in f.columns else None
    alt_sel_ft = to_np(f["fdm_alt_sel_ft"]) if "fdm_alt_sel_ft" in f.columns else None

    vz_ftmin = to_np(f["raw_vz_ftmin"])
    vz_sel_ftmin = to_np(f["fdm_vz_sel_ftmin"]) if "fdm_vz_sel_ftmin" in f.columns else None

    gamma = to_np(f["fdm_gamma_rad"])
    gamma_sel = to_np(f["fdm_gamma_sel_rad"]) if "fdm_gamma_sel_rad" in f.columns else None
    gamma_from_alt = (
        to_np(f["fdm_gamma_from_alt_rad"]) if "fdm_gamma_from_alt_rad" in f.columns else None
    )
    gamma_target = to_np(f["fdm_gamma_target_rad"]) if "fdm_gamma_target_rad" in f.columns else None
    gamma_diff = to_np(f["fdm_gamma_diff_rad"]) if "fdm_gamma_diff_rad" in f.columns else None

    # ── Convert to degrees for readability ──────────────────────────────
    r2d = np.degrees

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(18, 16), sharex=True)
    fig.suptitle(f"Gamma Target Reconciliation — {flight_id}", fontsize=14, fontweight="bold")

    # 1) Altitude
    ax = axes[0]
    ax.plot(t_min, alt_ft / 100, color="black", linewidth=0.8, label="Altitude (FL)")
    if alt_target_ft is not None:
        ax.plot(t_min, alt_target_ft / 100, color="blue", linewidth=2.0, alpha=0.7,
                label="Alt target (bfill)")
    if alt_sel_ft is not None:
        mask = ~np.isnan(alt_sel_ft)
        ax.scatter(t_min[mask], alt_sel_ft[mask] / 100, color="green", s=3, alpha=0.5,
                   label="Alt sel (plateaux)")
    ax.set_ylabel("Flight Level")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Altitude + target + plateau detection", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2) Vertical speed
    ax = axes[1]
    ax.plot(t_min, vz_ftmin, color="black", linewidth=0.5, alpha=0.6, label="Vz (ft/min)")
    if vz_sel_ftmin is not None:
        mask = ~np.isnan(vz_sel_ftmin)
        ax.scatter(t_min[mask], vz_sel_ftmin[mask], color="red", s=4, alpha=0.7,
                   label="Vz sel (plateaux)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Vz (ft/min)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Vertical Speed + segment detection", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3) Gamma sources (3 layers)
    ax = axes[2]
    ax.plot(t_min, r2d(gamma), color="black", linewidth=0.5, alpha=0.5, label="γ actual")
    if gamma_from_alt is not None:
        mask = ~np.isnan(gamma_from_alt)
        ax.scatter(t_min[mask], r2d(gamma_from_alt[mask]), color="orange", s=5, alpha=0.7,
                   label="γ from Vz_sel (P3)", zorder=3)
    if gamma_sel is not None:
        mask = ~np.isnan(gamma_sel)
        ax.scatter(t_min[mask], r2d(gamma_sel[mask]), color="purple", s=5, alpha=0.7,
                   label="γ_sel direct (P2)", zorder=4)
    if alt_sel_ft is not None:
        mask = ~np.isnan(alt_sel_ft)
        ax.scatter(t_min[mask], np.zeros(mask.sum()), color="green", s=5, alpha=0.5,
                   label="γ=0 alt hold (P1)", zorder=5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("γ (degrees)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Gamma sources: Vz→γ (P3) < γ_sel (P2) < alt_hold=0 (P1)", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4) Gamma target (unified) vs actual
    ax = axes[3]
    ax.plot(t_min, r2d(gamma), color="black", linewidth=0.5, alpha=0.5, label="γ actual")
    if gamma_target is not None:
        ax.plot(t_min, r2d(gamma_target), color="blue", linewidth=2.0, alpha=0.8,
                label="γ target (unified, bfill)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("γ (degrees)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Gamma target unifié vs actual", fontsize=10)
    ax.grid(True, alpha=0.3)

    # 5) Gamma diff (error signal)
    ax = axes[4]
    if gamma_diff is not None:
        ax.plot(t_min, r2d(gamma_diff), color="red", linewidth=0.8, alpha=0.8,
                label="γ_diff (target - actual)")
        ax.fill_between(t_min, 0, r2d(gamma_diff), alpha=0.15, color="red")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Δγ (degrees)")
    ax.set_xlabel("Time (minutes)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Gamma diff — signal d'erreur pour l'ODE", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"data/figures/gamma_target_{flight_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
