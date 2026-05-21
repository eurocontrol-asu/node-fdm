"""Strategy E / Experiment 10 — P&S Eq (1) fuel-flow mass inversion vs QAR.

Phase 1.8a follow-up after the Eq 100 path plateaued at corr 0.36
(Exp 07-09). Tests whether fuel flow — a *causally independent*
observable channel from altitude — provides a stronger mass signal.

Eq 1 (P&S Part 1) rearranged for mass :

    dm_f/dS = m·g / ((eta_o L/D) · LCV)
    ==>  m  = (eta_o L/D) · LCV · m_dot_f / (V · g)

Two inversion methods :

    METHOD 3 — Pure fuel-flow, constant (eta_o L/D)
        Use a single typical A320 cruise value (eta_o L/D)_typ ~ 16
        from the optimum theory at mass_ratio = 0.85. Treats fuel
        flow as the dominant observable, ignores Mach/FL dependence.

    METHOD 1 — Joint Eq 100 + Eq 1
        Step a : invert FL_obs via Eq 100 to get mass_ratio_PS.
        Step b : compute (eta_o L/D)_o at mass_ratio_PS via
                 optimum_in_isa(psi, mr).eta_o_lod_o.
        Step c : m_PS_eq1_joint = (eta_o L/D)_o · LCV · m_dot_f / (V·g).
        Uses both altitude (Eq 100) and fuel flow as independent
        channels. Expected stronger than either alone.

Cruise filter : alt >= 9000 m, |dh/dt| < 1 m/s, |dV/dt| < 0.1 m/s^2,
                and FF non-zero. Below cruise the engines are at idle
                or climb power — Eq 1 doesn't apply.

R5 compliant : QAR is read for validation only ; no training data is
modified or used.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(
    0, "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages/ps-core/src"
)

from ps_core._types import AircraftPsi
from ps_core.optimum import optimum_in_isa
from test_a_mass_encoder_qar import (
    _FT_TO_M,
    _KT_TO_MS,
    _dedup_qar_files,
)

A320_PSI = AircraftPsi(
    psi_1=0.156,
    psi_2=8.05,
    psi_4=0.753,
    psi_5=6.29e7,
    psi_6=0.656,
    tau=0.162,
)
A320_MTOW_KG = 77_000.0

_LCV_JET_A = 4.3e7  # J/kg (Jet A kerosene lower calorific value)
_G = 9.80665  # m/s^2
_KG_PER_H_TO_KG_PER_S = 1.0 / 3600.0

_CRUISE_ALT_MIN_M = 9_000.0
_CRUISE_DH_DT_MAX_MS = 1.0
_CRUISE_DV_DT_MAX_MS2 = 0.1
_MIN_FUEL_FLOW_KG_PER_H = 100.0  # exclude idle / null

# Eq 100 inversion table (same as Exp 07).
_MASS_RATIO_MIN = 0.50
_MASS_RATIO_MAX = 1.00
_MASS_RATIO_GRID_N = 1001

_SEQ_LEN_S = 60
_SHIFT_S = 60


def _build_optimum_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mass_ratio_grid, fl_o_grid, eta_o_lod_o_grid)."""
    grid = np.linspace(_MASS_RATIO_MIN, _MASS_RATIO_MAX, _MASS_RATIO_GRID_N)
    fl_o = np.empty(_MASS_RATIO_GRID_N)
    eta_lod = np.empty(_MASS_RATIO_GRID_N)
    for i, m in enumerate(grid):
        pt = optimum_in_isa(A320_PSI, float(m))
        fl_o[i] = pt.fl_o
        eta_lod[i] = pt.eta_o_lod_o
    return grid, fl_o, eta_lod


def _cruise_stable_mask(alt_m: np.ndarray, tas_ms: np.ndarray) -> np.ndarray:
    n = len(alt_m)
    if n < 2:
        return np.zeros(n, dtype=bool)
    dh_dt = np.gradient(alt_m)
    dv_dt = np.gradient(tas_ms)
    return (
        np.isfinite(alt_m)
        & np.isfinite(tas_ms)
        & (alt_m >= _CRUISE_ALT_MIN_M)
        & (np.abs(dh_dt) <= _CRUISE_DH_DT_MAX_MS)
        & (np.abs(dv_dt) <= _CRUISE_DV_DT_MAX_MS2)
    )


def _segments_from_flight(
    df: pl.DataFrame, *, seq_len: int, shift: int
) -> list[dict]:
    n = df.shape[0]
    if n < seq_len:
        return []
    if "FUEL__FF_LEFT" not in df.columns or "FUEL__FF_RIGHT" not in df.columns:
        return []
    alt_ft = df["ALT__STD"].to_numpy().astype(np.float64)
    alt_m = alt_ft * _FT_TO_M
    gw = df["SYS__GW"].to_numpy().astype(np.float64)
    tas_ms = (
        df["SPD__TAS"].to_numpy().astype(np.float64) * _KT_TO_MS
        if "SPD__TAS" in df.columns
        else np.full(n, np.nan)
    )
    ff_l = df["FUEL__FF_LEFT"].to_numpy().astype(np.float64)
    ff_r = df["FUEL__FF_RIGHT"].to_numpy().astype(np.float64)
    ff_total_kgh = ff_l + ff_r  # kg/h total fuel flow
    cruise_mask = _cruise_stable_mask(alt_m, tas_ms)

    out: list[dict] = []
    for start in range(0, n - seq_len + 1, shift):
        m_truth = float(gw[start]) if start < n else float("nan")
        if not (np.isfinite(m_truth) and 40_000 < m_truth < 80_000):
            continue
        a_ft = float(alt_ft[start])
        a_m = float(alt_m[start])
        v_ms = float(tas_ms[start])
        ff_kgh = float(ff_total_kgh[start])
        if not (np.isfinite(a_ft) and np.isfinite(v_ms) and np.isfinite(ff_kgh)):
            continue
        if a_ft < 1_000.0 or v_ms < 100.0 or ff_kgh < _MIN_FUEL_FLOW_KG_PER_H:
            continue
        out.append(
            {
                "alt_m": a_m,
                "fl_obs": a_ft / 100.0,
                "tas_ms": v_ms,
                "ff_kg_s": ff_kgh * _KG_PER_H_TO_KG_PER_S,
                "m_truth": m_truth,
                "is_cruise": bool(cruise_mask[start]),
            }
        )
    return out


def _invert_eq1_pure_fuel_flow(
    ff_kg_s: np.ndarray, tas_ms: np.ndarray, eta_o_lod_typ: float
) -> np.ndarray:
    """Method 3 : m = (eta_o L/D)_typ * LCV * m_dot_f / (V * g). Pure FF."""
    return eta_o_lod_typ * _LCV_JET_A * ff_kg_s / (tas_ms * _G)


def _invert_eq1_joint_eq100(
    fl_obs: np.ndarray,
    ff_kg_s: np.ndarray,
    tas_ms: np.ndarray,
    grid_mr: np.ndarray,
    grid_fl: np.ndarray,
    grid_eta_lod: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Method 1 : joint Eq 100 (FL -> mass_ratio) + Eq 1 (FF -> m).

    Returns (m_estimate, converged_mask).
    """
    fl_asc = grid_fl[::-1]
    eta_desc = grid_eta_lod[::-1]
    fl_min, fl_max = float(grid_fl[-1]), float(grid_fl[0])
    converged = np.isfinite(fl_obs) & (fl_obs >= fl_min) & (fl_obs <= fl_max)
    eta_at_mr = np.interp(fl_obs, fl_asc, eta_desc)
    m_est = eta_at_mr * _LCV_JET_A * ff_kg_s / (tas_ms * _G)
    # ``grid_mr`` is kept in the signature for symmetry/diagnostic introspection
    _ = grid_mr
    return m_est, converged


def _flight_id(p: Path) -> str:
    fid = p.name.split("_")[1] if "_" in p.name else p.stem
    return f"{fid}__{p.stem}"


def _decompose_between_within(
    y: np.ndarray, preds: np.ndarray, flight_ids: np.ndarray
) -> dict[str, float]:
    unique = sorted(set(flight_ids.tolist()))
    if not unique:
        return {"between_flight_corr": float("nan"), "within_flight_corr": float("nan")}
    y_means = np.zeros_like(y, dtype=np.float64)
    p_means = np.zeros_like(preds, dtype=np.float64)
    n_per: dict[str, int] = {}
    for fid in unique:
        mask = flight_ids == fid
        y_means[mask] = y[mask].mean()
        p_means[mask] = preds[mask].mean()
        n_per[fid] = int(mask.sum())
    between = float("nan")
    if len(unique) >= 2 and np.std(y_means) > 1e-6 and np.std(p_means) > 1e-6:
        between = float(np.corrcoef(y_means, p_means)[0, 1])
    multi_mask = np.array([n_per[fid] >= 2 for fid in flight_ids], dtype=bool)
    within = float("nan")
    if multi_mask.any():
        y_w = (y - y_means)[multi_mask]
        p_w = (preds - p_means)[multi_mask]
        if np.std(y_w) > 1e-6 and np.std(p_w) > 1e-6:
            within = float(np.corrcoef(y_w, p_w)[0, 1])
    return {"between_flight_corr": between, "within_flight_corr": within}


def _metrics(
    y: np.ndarray, preds: np.ndarray, flight_ids: np.ndarray
) -> dict[str, float]:
    err = preds - y
    n = len(y)
    if n < 2 or np.std(y) < 1e-6 or np.std(preds) < 1e-6:
        return {"n": n, "corr": float("nan"), "mae": float("nan"), "bias_pct": float("nan")}
    return {
        "n": n,
        "corr": float(np.corrcoef(y, preds)[0, 1]),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(err.mean()),
        "bias_pct": float(np.mean(err / y) * 100.0),
        **_decompose_between_within(y, preds, flight_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qar-dir", default="/Users/gabriel/Downloads/QAR3")
    parser.add_argument("--seq-len", type=int, default=_SEQ_LEN_S)
    parser.add_argument("--shift", type=int, default=_SHIFT_S)
    parser.add_argument(
        "--report",
        default="data/models/_comparison/strategy_e_ps_eq1_fuelflow.md",
    )
    args = parser.parse_args()

    grid_mr, grid_fl, grid_eta = _build_optimum_table()
    fl_min, fl_max = float(grid_fl[-1]), float(grid_fl[0])
    # Take (eta_o L/D) at mass_ratio = 0.85 (~mid operational A320) as
    # the constant for Method 3.
    eta_o_lod_typ = float(np.interp(0.85, grid_mr, grid_eta))
    print(
        f"Optimum tables: FL_o in [{fl_min:.1f}, {fl_max:.1f}], "
        f"(eta_o L/D) range [{grid_eta.min():.2f}, {grid_eta.max():.2f}]"
    )
    print(f"Method 3 constant (eta_o L/D)_typ at mass_ratio=0.85 = {eta_o_lod_typ:.3f}")

    qar_files = _dedup_qar_files(sorted(Path(args.qar_dir).glob("*.parquet")))
    print(f"Found {len(qar_files)} unique QAR files")

    alt_m_l: list[float] = []
    fl_obs_l: list[float] = []
    tas_l: list[float] = []
    ff_l: list[float] = []
    truth_l: list[float] = []
    is_cruise_l: list[bool] = []
    flight_ids_l: list[str] = []

    n_files_ok = 0
    for f in qar_files:
        try:
            df = pl.read_parquet(f)
        except Exception:  # noqa: S112 # bad parquet files: skip
            continue
        segs = _segments_from_flight(df, seq_len=args.seq_len, shift=args.shift)
        if not segs:
            continue
        n_files_ok += 1
        fid = _flight_id(f)
        for s in segs:
            alt_m_l.append(s["alt_m"])
            fl_obs_l.append(s["fl_obs"])
            tas_l.append(s["tas_ms"])
            ff_l.append(s["ff_kg_s"])
            truth_l.append(s["m_truth"])
            is_cruise_l.append(s["is_cruise"])
            flight_ids_l.append(fid)

    if not truth_l:
        print("No segments collected", file=sys.stderr)
        return 1

    fl_arr = np.array(fl_obs_l)
    tas_arr = np.array(tas_l)
    ff_arr = np.array(ff_l)
    y_arr = np.array(truth_l)
    cruise_arr = np.array(is_cruise_l, dtype=bool)
    flight_arr = np.array(flight_ids_l)

    # Method 3 : pure fuel-flow
    m_eq1_pure = _invert_eq1_pure_fuel_flow(ff_arr, tas_arr, eta_o_lod_typ)
    # Method 1 : joint Eq 100 + Eq 1
    m_eq1_joint, joint_conv = _invert_eq1_joint_eq100(
        fl_arr, ff_arr, tas_arr, grid_mr, grid_fl, grid_eta
    )

    n_total = len(y_arr)
    n_cruise = int(cruise_arr.sum())
    n_joint = int(joint_conv.sum())
    n_cruise_joint = int((cruise_arr & joint_conv).sum())
    print(
        f"Segments: total={n_total}  cruise_stable={n_cruise}  "
        f"in_eq100_range={n_joint}  cruise&joint={n_cruise_joint}  "
        f"across {n_files_ok} flights"
    )

    views = {
        "Method 3 — Pure FF, ALL segments (no cruise filter)": (
            m_eq1_pure, np.ones(n_total, dtype=bool),
        ),
        "Method 3 — Pure FF, CRUISE STABLE only": (m_eq1_pure, cruise_arr),
        "Method 1 — Joint Eq 100+Eq 1, in inversion range": (m_eq1_joint, joint_conv),
        "Method 1 — Joint Eq 100+Eq 1, CRUISE STABLE": (
            m_eq1_joint, cruise_arr & joint_conv,
        ),
    }
    metrics: dict[str, dict[str, float]] = {}
    for label, (preds, mask) in views.items():
        if mask.sum() < 10:
            print(f"  [{label}]  n={int(mask.sum())} (skipped)")
            metrics[label] = {"n": int(mask.sum())}
            continue
        m = _metrics(y_arr[mask], preds[mask], flight_arr[mask])
        metrics[label] = m
        print(
            f"  [{label}]\n"
            f"    n={m['n']:>6}  corr={m['corr']:+.3f}  "
            f"MAE={m['mae']:>6.0f} kg  bias={m['bias_pct']:+5.1f}%  "
            f"between={m['between_flight_corr']:+.3f}  "
            f"within={m['within_flight_corr']:+.3f}"
        )

    # --- Write markdown report ---
    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Strategy E — Diagnostic Exp 10 : P&S Eq 1 fuel-flow inversion vs QAR",
        "",
        "> Phase 1.8a — tests whether fuel flow as an *independent observable* "
        "carries a stronger mass signal than altitude (Eq 100) alone.",
        "",
        "## Inversion setup",
        "",
        f"- LCV (Jet A) : {_LCV_JET_A / 1e6:.1f} MJ/kg",
        f"- (eta_o L/D) constant (Method 3, mass_ratio=0.85) : {eta_o_lod_typ:.3f}",
        f"- Cruise filter : alt >= {_CRUISE_ALT_MIN_M:.0f} m, "
        f"|dh/dt| <= {_CRUISE_DH_DT_MAX_MS} m/s, "
        f"|dV/dt| <= {_CRUISE_DV_DT_MAX_MS2} m/s^2, "
        f"FF >= {_MIN_FUEL_FLOW_KG_PER_H} kg/h",
        f"- Sliding window : seq_len={args.seq_len}s, shift={args.shift}s",
        "",
        "## Sample counts",
        "",
        f"- Total segments : **{n_total}** across {n_files_ok} flights",
        f"- Cruise-stable : **{n_cruise}** ({100.0 * n_cruise / max(1, n_total):.1f} %)",
        f"- In Eq 100 inversion range (FL_obs in [{fl_min:.1f}, {fl_max:.1f}]) : **{n_joint}**",
        f"- Cruise & joint-invertible : **{n_cruise_joint}**",
        "",
        "## Headline metrics",
        "",
        "| View | n | corr | MAE (kg) | bias % | between | within |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, m in metrics.items():
        if m.get("n", 0) < 10:
            lines.append(f"| {label} | {m.get('n', 0)} | n/a | n/a | n/a | n/a | n/a |")
            continue
        within = (
            f"{m['within_flight_corr']:+.3f}"
            if not math.isnan(m["within_flight_corr"])
            else "n/a"
        )
        between = (
            f"{m['between_flight_corr']:+.3f}"
            if not math.isnan(m["between_flight_corr"])
            else "n/a"
        )
        lines.append(
            f"| {label} | {m['n']} | {m['corr']:+.3f} | "
            f"{m['mae']:.0f} | {m['bias_pct']:+.1f} | {between} | {within} |"
        )

    lines.extend([
        "",
        "## Reference rows (from prior experiments, pooled n=49 400 / 12 305)",
        "",
        "| Estimator | corr | within | MAE |",
        "|---|---:|---:|---:|",
        "| Exp 07 : m_PS_eq100 (cruise) | 0.449 | +0.393 | 8 614 |",
        "| Exp 08 : v10 aux-loss best | 0.362 | +0.65 | 5 403 |",
        "| Exp 09 : v11 residual | 0.352 | +0.717 | 4 500 |",
        "| v9 baseline | 0.351 | +0.727 | 4 501 |",
        "| OLS oracle | 0.590 | — | 3 452 |",
        "",
        "## Decision tree",
        "",
        "- corr ≥ 0.55 on cruise → fuel flow IS the independent channel. "
        "Recommend Exp 11 : integrate as aux teacher or residual anchor.",
        "- corr ∈ [0.35, 0.55] → improves over Eq 100 modestly. "
        "Combine in Exp 11 if cost-justified.",
        "- corr < 0.35 → fuel flow doesn't carry more info than altitude alone "
        "(under our simple linearisation). Theory falsified for this path ; "
        "escalate to counterfactual augmentation or skip to Phase 2.",
    ])
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
