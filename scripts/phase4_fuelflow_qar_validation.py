"""Phase 4 — AC7 QAR fuel-flow validation. The central scientific gate.

Adapted from `phase3_thrust_qar_t_observable.py`. For each cruise-stable
QAR sample :

    1. Reconstruct kinematic T-observable
       `T_obs = m·dV/dt + D_PS(C_L) + m·g·sin γ`.
    2. Compute analytical `eta_PS(C_T_inst, M)` from PSEfficiencyLayer
       (where C_T_inst = T_obs / (q · S_ref)).
    3. Compute predicted fuel flow `mdot_f_pred = T_obs · V / (eta_PS · LCV)`.
    4. Compare to QAR ground truth `mdot_f_qar = (FUEL__FF_L + FUEL__FF_R) / 3600`
       (kg/s).

This is the **first moment** of the project where Eq 19 inverse is
tested against the QAR observable — pure forward-physics, no NN.

R5-safe : QAR is used for validation only, never for training.

Stop Condition 3 — Theory falsified when median `mdot_f_pred` off > 50 %
vs `mdot_f_qar`.

AC7 PASS : `corr(mdot_f_pred, mdot_f_qar) ≥ +0.50`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Lazy import of the torch layer (the script can run pure-numpy too if
# the layer instantiation fails, since we have a numpy eta_o_reference).
REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_FDM_SRC = REPO_ROOT / "packages" / "node-fdm" / "src"
if str(NODE_FDM_SRC) not in sys.path:
    sys.path.insert(0, str(NODE_FDM_SRC))

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")

# A320 + CFM56-5B4_P (same constants as PSThrustLayer / PSEfficiencyLayer).
M_DO = 0.753
C_T_DO = 0.0347
BPR = 5.6
ETA_O_DO = 0.309
S_REF = 122.6
C_T_RATIO_K = 0.55
G = 9.80665
LCV_KEROSENE = 43.0e6
H0_CURVATURE = 0.43

# A320 polar (same as PSDragLayer).
PSI_0 = 7.846
SPAN_M = 34.10
S_REF_LIB = 122.4
SWEEP_DEG = 25.0
AR = SPAN_M * SPAN_M / S_REF_LIB
MTF_AC = 0.87
J_1 = 11.0
J_2 = 1.0
E_LS = 0.778
K_INDUCED = 1.0 / (np.pi * AR * E_LS)
SKIN_A = 0.0269
SKIN_B = 0.14
L_REF = np.sqrt(S_REF)
MU_REF = 1.716e-5
T_REF_K = 273.15
S_SUTH = 110.4
R_GAS = 287.05
GAMMA_AIR = 1.4

# Eq 29 : eta_2(BPR) = 0.65 * (1 - 0.035 * BPR)
ETA_2 = 0.65 * (1.0 - 0.035 * BPR)

FT_PER_M = 3.28084
KTS_PER_MS = 1.9438445


def h_2(mach: np.ndarray) -> np.ndarray:
    """Eq 28 P&S Part 3 : (C_T)_etaB / (C_T)_DO."""
    m_safe = np.maximum(mach, 0.1)
    num = 1.0 + C_T_RATIO_K * m_safe
    den = 1.0 + C_T_RATIO_K * M_DO
    return (num / den) * (M_DO / m_safe) ** 2


def eta_o_np(
    c_t: np.ndarray,
    mach: np.ndarray,
    eta_o_do_override: float | None = None,
) -> np.ndarray:
    """Eqs 24-29 high-thrust branch only (cruise mode).

    Phase 4.5 : accepts an `eta_o_do_override` to sweep the fleet-effective
    `η_o_DO` without modifying the module constant (which stays at the
    textbook 0.309 to preserve v14 parity).
    """
    eta_o_do_eff = eta_o_do_override if eta_o_do_override is not None else ETA_O_DO
    m_safe = np.maximum(mach, 0.1)
    h_1 = (m_safe / M_DO) ** ETA_2
    eta_o_b = h_1 * eta_o_do_eff
    c_t_eta_b = h_2(m_safe) * C_T_DO
    ratio = c_t / np.maximum(c_t_eta_b, 1e-6)
    # Omega(M) — Eqs 25-26.
    omega = np.where(
        m_safe >= 0.4, 0.0, np.where(m_safe >= 0.2, 1.30 * (0.4 - m_safe), 1.30 * (0.4 - 0.2))
    )
    arg = ratio - 1.0
    h_0 = (1.0 - H0_CURVATURE * arg * arg) * (1.0 + omega * arg * arg)
    return h_0 * eta_o_b


def c_d_ps(
    c_l: np.ndarray, mach: np.ndarray, temp_k: np.ndarray, q_pa: np.ndarray, tas_ms: np.ndarray
) -> np.ndarray:
    """Analytical C_D Part 3 §4 — same as PSDragLayer."""
    temp_safe = np.maximum(temp_k, 180.0)
    mu = (
        MU_REF
        * (temp_safe / T_REF_K).clip(min=0.1) ** 1.5
        * (T_REF_K + S_SUTH)
        / (temp_safe + S_SUTH)
    )
    tas_safe = np.maximum(tas_ms, 50.0)
    rho = (2.0 * np.maximum(q_pa, 100.0)) / (tas_safe * tas_safe)
    re_ac = np.maximum(rho * tas_safe * L_REF / np.maximum(mu, 1e-7), 1e3)
    c_f = SKIN_A / re_ac**SKIN_B
    c_d0 = PSI_0 * c_f
    cl_safe = np.clip(c_l, 0.0, 2.0)
    cs = np.cos(np.radians(SWEEP_DEG))
    m_cc = MTF_AC - 0.10 * cl_safe / (cs * cs)
    m_cc_safe = np.maximum(m_cc, 0.30)
    x = mach * cs / m_cc_safe
    c_dw = (cs**3) * J_1 * np.maximum(x - J_2, 0.0) ** 2
    c_di = K_INDUCED * cl_safe * cl_safe
    return c_d0 + c_di + c_dw


def isa_temp_pressure(alt_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ISA static T (K) and p (Pa) at altitude (m)."""
    T_SL = 288.15
    P_SL = 101325.0
    L = 0.0065
    TROP_M = 11_000.0
    T_TROP = T_SL - L * TROP_M
    P_TROP = P_SL * (T_TROP / T_SL) ** 5.2561
    below = alt_m < TROP_M
    t_k = np.where(below, T_SL - L * alt_m, T_TROP)
    p_pa = np.where(
        below,
        P_SL * (t_k / T_SL) ** 5.2561,
        P_TROP * np.exp(-G * (alt_m - TROP_M) / (R_GAS * T_TROP)),
    )
    return t_k, p_pa


def process_flight(parquet_path: Path) -> dict[str, np.ndarray] | None:
    """Return per-cruise-sample arrays for the AC7 diagnostic."""
    df = pl.read_parquet(parquet_path)
    needed = {
        "ALT__STD",
        "SPD__TAS",
        "SPD__MACH",
        "SYS__GW",
        "ATT__PITCH",
        "TEMP__SAT",
        "FUEL__FF_LEFT",
        "FUEL__FF_RIGHT",
    }
    if not needed.issubset(df.columns):
        return None

    df = df.with_columns(
        [
            (pl.col("ALT__STD") / FT_PER_M).alias("alt_m"),
            (pl.col("SPD__TAS") / KTS_PER_MS).alias("tas_ms"),
            (pl.col("TEMP__SAT") + 273.15).alias("temp_k"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("alt_m").diff().alias("dalt_m_s"),
            pl.col("tas_ms").diff().alias("dtas_ms2"),
        ]
    )
    df = df.drop_nulls(
        ["alt_m", "tas_ms", "dtas_ms2", "dalt_m_s", "SPD__MACH", "FUEL__FF_LEFT", "FUEL__FF_RIGHT"]
    )
    df_cruise = df.filter(
        (pl.col("alt_m") > 8500.0)
        & (pl.col("alt_m") < 12500.0)
        & (pl.col("dalt_m_s").abs() < 0.5)
        & (pl.col("dtas_ms2").abs() < 0.3)
        & (pl.col("SPD__MACH") > 0.6)
        & (pl.col("SPD__MACH") < 0.85)
        & (pl.col("SYS__GW") > 40000)
        & (pl.col("SYS__GW") < 80000)
        & (pl.col("FUEL__FF_LEFT") > 0)
        & (pl.col("FUEL__FF_RIGHT") > 0)
    )
    if len(df_cruise) < 50:
        return None

    alt_m = df_cruise["alt_m"].to_numpy()
    tas_ms = df_cruise["tas_ms"].to_numpy()
    mach = df_cruise["SPD__MACH"].to_numpy()
    mass_kg = df_cruise["SYS__GW"].to_numpy()
    temp_k = df_cruise["temp_k"].to_numpy()
    dalt = df_cruise["dalt_m_s"].to_numpy()
    dtas = df_cruise["dtas_ms2"].to_numpy()
    ff_l = df_cruise["FUEL__FF_LEFT"].to_numpy().astype(np.float64)
    ff_r = df_cruise["FUEL__FF_RIGHT"].to_numpy().astype(np.float64)

    # Flight-path angle from dh/dt and TAS.
    gamma = np.arcsin(np.clip(dalt / np.maximum(tas_ms, 50.0), -0.3, 0.3))

    # Dynamic pressure (ISA + SAT).
    _, p_pa = isa_temp_pressure(alt_m)
    rho = p_pa / (R_GAS * temp_k)
    q_pa = 0.5 * rho * tas_ms * tas_ms

    # Analytical drag (Phase 2).
    c_l = mass_kg * G / (q_pa * S_REF)
    cd = c_d_ps(c_l, mach, temp_k, q_pa, tas_ms)
    drag_n = q_pa * S_REF * cd

    # T-observable from kinematics.
    t_obs = mass_kg * dtas + drag_n + mass_kg * G * np.sin(gamma)
    # Filter implausibly negative thrust (descent below idle).
    t_obs_valid = t_obs > 1000.0

    # C_T_inst = T_obs / (q * S_ref).
    c_t_inst = t_obs / np.maximum(q_pa * S_REF, 1.0)

    # QAR ground truth fuel flow. FUEL__FF_* is in kg/h on Honeywell A320.
    mdot_f_qar_kg_s = (ff_l + ff_r) / 3600.0

    # Kinematic validity (independent of η_o_DO sweep value).
    valid_kin = t_obs_valid & (mdot_f_qar_kg_s > 0.1) & (mdot_f_qar_kg_s < 5.0)

    return {
        "alt_m": alt_m,
        "tas_ms": tas_ms,
        "mach": mach,
        "mass_kg": mass_kg,
        "q_pa": q_pa,
        "drag_n": drag_n,
        "t_obs_n": t_obs,
        "c_t_inst": c_t_inst,
        "mdot_f_qar_kg_s": mdot_f_qar_kg_s,
        "valid_kin": valid_kin,
    }


def compute_metrics_for_eta_o_do(
    cached: dict[str, np.ndarray],
    eta_o_do: float,
) -> dict[str, float]:
    """Apply a single `η_o_DO` value to cached kinematics, return AC7 metrics.

    The QAR loop is run *once* (expensive ; loads ~500 parquet files and
    runs the cruise filter), then this function is called per sweep value
    on the cached arrays. Cost per call : ~150 ms on 793 k samples.
    """
    c_t_inst = cached["c_t_inst"]
    mach = cached["mach"]
    t_obs = cached["t_obs_n"]
    tas_ms = cached["tas_ms"]
    mdot_qar = cached["mdot_f_qar_kg_s"]
    valid_kin = cached["valid_kin"]

    eta_ps = eta_o_np(c_t_inst, mach, eta_o_do_override=eta_o_do)
    mdot_pred = t_obs * tas_ms / np.maximum(eta_ps * LCV_KEROSENE, 1e-6)
    valid = valid_kin & (eta_ps > 0.05) & (eta_ps < 0.55)

    if int(valid.sum()) < 100:
        return {
            "eta_o_do": eta_o_do,
            "n": 0,
            "corr": float("nan"),
            "bias": float("nan"),
            "median_abs_rel_err": float("nan"),
            "p90_abs_rel_err": float("nan"),
            "p99_abs_rel_err": float("nan"),
            "eta_ps_median": float("nan"),
            "mdot_pred_median": float("nan"),
            "mdot_qar_median": float("nan"),
        }

    mp = mdot_pred[valid]
    mq = mdot_qar[valid]
    corr = float(np.corrcoef(mp, mq)[0, 1])
    abs_rel = np.abs((mp - mq) / mq)
    return {
        "eta_o_do": eta_o_do,
        "n": int(valid.sum()),
        "corr": corr,
        "bias": float(np.median(mp / mq)),
        "median_abs_rel_err": float(np.median(abs_rel)),
        "p90_abs_rel_err": float(np.percentile(abs_rel, 90)),
        "p99_abs_rel_err": float(np.percentile(abs_rel, 99)),
        "eta_ps_median": float(np.median(eta_ps[valid])),
        "mdot_pred_median": float(np.median(mp)),
        "mdot_qar_median": float(np.median(mq)),
    }


def _run_sweep(
    cached: dict[str, np.ndarray],
    sweep_values: list[float],
    out_csv: Path | None,
    n_flights: int,
) -> int:
    """Sweep mode : evaluate every η_o_DO in the list, write CSV + markdown."""
    rows: list[dict[str, float]] = []
    print(f"\n=== Phase 4.5 η_o_DO sweep ({len(sweep_values)} values) ===\n")
    print(f"{'η_o_DO':>8} {'n':>9} {'corr':>9} {'bias':>9} {'median|Δ|%':>11} "
          f"{'p99|Δ|%':>9} {'η_PS_med':>9} {'mdot_pred':>10}")
    for eta in sweep_values:
        m = compute_metrics_for_eta_o_do(cached, eta)
        rows.append(m)
        print(
            f"{m['eta_o_do']:>8.4f} {m['n']:>9d} {m['corr']:>+9.4f} "
            f"{m['bias']:>9.4f} {m['median_abs_rel_err']*100:>11.2f} "
            f"{m['p99_abs_rel_err']*100:>9.2f} {m['eta_ps_median']:>9.4f} "
            f"{m['mdot_pred_median']:>10.4f}"
        )

    # Identify η_o_DO* : min |bias - 1| subject to corr ≥ +0.30 (loose filter).
    feasible = [r for r in rows if np.isfinite(r["corr"]) and r["corr"] >= 0.30]
    if feasible:
        best = min(feasible, key=lambda r: abs(r["bias"] - 1.0))
        print(f"\nBest by min |bias - 1| : η_o_DO* = {best['eta_o_do']:.4f}  "
              f"(bias={best['bias']:.4f}, corr={best['corr']:+.4f}, "
              f"median|Δ|={best['median_abs_rel_err']*100:.2f}%)")
    else:
        best = None
        print("\nNo feasible point (all corr < +0.30) — sweep may need to widen.")

    # Determine verdict.
    if best is not None and abs(best["bias"] - 1.0) < 0.05 and best["corr"] >= 0.50 \
       and best["median_abs_rel_err"] < 0.10:
        verdict = "✅ Success — Stop Condition 1 reached"
    elif best is not None and abs(best["bias"] - 1.0) < 0.05 and best["corr"] < 0.50:
        verdict = "⚠️ Trade-off characterised — Stop Condition 2"
    else:
        # Check if any candidate has corr >> baseline.
        max_corr = max((r["corr"] for r in rows if np.isfinite(r["corr"])), default=float("nan"))
        if max_corr < 0.45:
            verdict = "🔬 Theory falsified — Stop Condition 3 (no η_o_DO produces corr ≥ +0.45)"
        else:
            verdict = "⚠️ Partial — refine sweep (Strategy B)"

    # Write CSV
    if out_csv is None:
        out_csv = Path("data/investigations/phase4_5_eta_recalib/artifacts/eta_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6f}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    print(f"\nCSV written : {out_csv}")

    # Write markdown report
    out_md = out_csv.with_suffix(".md")
    lines = [
        "# Phase 4.5 — η_o_DO sweep results",
        "",
        f"> n_flights = {n_flights}. n_cached samples = {len(cached['valid_kin'])}.",
        f"> Sweep values : {sweep_values}.",
        "",
        f"**Verdict** : {verdict}",
        "",
        "## Sweep table",
        "",
        "| η_o_DO | n | corr Pearson | bias (pred/qar) | median \\|Δ\\| % | p90 \\|Δ\\| % | p99 \\|Δ\\| % | η_PS median | mdot_pred median (kg/s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['eta_o_do']:.4f} | {r['n']} | {r['corr']:+.4f} | "
            f"{r['bias']:.4f} | {r['median_abs_rel_err']*100:.2f} | "
            f"{r['p90_abs_rel_err']*100:.2f} | {r['p99_abs_rel_err']*100:.2f} | "
            f"{r['eta_ps_median']:.4f} | {r['mdot_pred_median']:.4f} |"
        )
    if best is not None:
        lines.extend([
            "",
            "## Best η_o_DO* (min |bias - 1|, corr ≥ +0.30)",
            "",
            f"- **η_o_DO* = {best['eta_o_do']:.4f}**",
            f"- bias = {best['bias']:.4f}",
            f"- corr Pearson = {best['corr']:+.4f}",
            f"- median |Δ| = {best['median_abs_rel_err']*100:.2f} %",
            f"- p99 |Δ| = {best['p99_abs_rel_err']*100:.2f} %",
        ])
    lines.extend([
        "",
        "## Compliance",
        "",
        "- R1 (no retraining) : ✅ — only the validation script touches η_o_DO.",
        "- R2 (no QAR in training) : ✅ — script is read-only on QAR data.",
        "- R6 (corr + bias + median |Δ| reported together) : ✅.",
    ])
    out_md.write_text("\n".join(lines))
    print(f"Markdown report : {out_md}")
    print(f"\nVerdict : {verdict}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AC7 QAR fuel-flow validation + Phase 4.5 η_o_DO sweep"
    )
    parser.add_argument(
        "--eta-o-do",
        type=float,
        default=None,
        help="Single η_o_DO override (default = textbook 0.309).",
    )
    parser.add_argument(
        "--sweep-values",
        type=str,
        default=None,
        help="Comma-separated list of η_o_DO values (sweep mode).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path for sweep mode.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/models/_comparison/phase4_fuelflow_v14_qar_validation.md"),
        help="Output markdown report (single-run mode).",
    )
    args = parser.parse_args()

    sweep_mode = args.sweep_values is not None
    if sweep_mode:
        try:
            sweep_values = [float(v.strip()) for v in args.sweep_values.split(",")]
        except ValueError as exc:
            print(f"Invalid --sweep-values : {exc}")
            return 1
        if not sweep_values:
            print("--sweep-values empty")
            return 1
        print(f"Sweep mode : {len(sweep_values)} η_o_DO values = {sweep_values}")
    else:
        single_eta = args.eta_o_do  # may be None → textbook
        print(
            f"Single-run mode : η_o_DO = {single_eta if single_eta is not None else ETA_O_DO} (textbook default)"
        )

    paths = sorted(QAR_DIR.glob("*A320*.parquet"))
    print(f"Found {len(paths)} A320 QAR files")
    if not paths:
        print("No QAR files — abort")
        return 1

    all_data: list[dict[str, np.ndarray]] = []
    for i, p in enumerate(paths):
        if i % 50 == 0:
            print(f"  [{i + 1}/{len(paths)}] {p.name}")
        d = process_flight(p)
        if d is not None:
            all_data.append(d)
    if not all_data:
        print("No cruise-stable flights — abort")
        return 1

    # Cache concatenated arrays across flights (kinematics + QAR FF only,
    # not yet η- or mdot_pred-dependent so the sweep can replay cheaply).
    cached = {
        "alt_m": np.concatenate([d["alt_m"] for d in all_data]),
        "tas_ms": np.concatenate([d["tas_ms"] for d in all_data]),
        "mach": np.concatenate([d["mach"] for d in all_data]),
        "mass_kg": np.concatenate([d["mass_kg"] for d in all_data]),
        "q_pa": np.concatenate([d["q_pa"] for d in all_data]),
        "drag_n": np.concatenate([d["drag_n"] for d in all_data]),
        "t_obs_n": np.concatenate([d["t_obs_n"] for d in all_data]),
        "c_t_inst": np.concatenate([d["c_t_inst"] for d in all_data]),
        "mdot_f_qar_kg_s": np.concatenate([d["mdot_f_qar_kg_s"] for d in all_data]),
        "valid_kin": np.concatenate([d["valid_kin"] for d in all_data]),
    }
    print(f"\nCached {len(cached['valid_kin'])} samples across {len(all_data)} flights")

    if sweep_mode:
        return _run_sweep(cached, sweep_values, args.out, len(all_data))

    # === Single-run mode (existing Phase 4 path) ===
    eta_o_do = args.eta_o_do  # may be None → textbook
    metrics = compute_metrics_for_eta_o_do(
        cached,
        eta_o_do if eta_o_do is not None else ETA_O_DO,
    )
    n = metrics["n"]
    if n < 100:
        print("Too few valid samples — abort")
        return 1

    # Re-derive mp / mq for distribution reporting (already inside metrics).
    eta_ps_full = eta_o_np(cached["c_t_inst"], cached["mach"], eta_o_do_override=eta_o_do)
    mdot_pred = cached["t_obs_n"] * cached["tas_ms"] / np.maximum(eta_ps_full * LCV_KEROSENE, 1e-6)
    valid = cached["valid_kin"] & (eta_ps_full > 0.05) & (eta_ps_full < 0.55)
    mp = mdot_pred[valid]
    mq = cached["mdot_f_qar_kg_s"][valid]
    eta_ps = eta_ps_full

    # === Test 1 : Pearson corr (AC7 central gate) ===
    corr_pearson = metrics["corr"]
    print("\n=== AC7 — corr(mdot_f_pred, mdot_f_qar) ===")
    print(f"Pearson corr : {corr_pearson:+.4f}")

    if corr_pearson >= 0.50:
        ac7_verdict = "✅ PASS"
    elif corr_pearson >= 0.30:
        ac7_verdict = "⚠️ partial"
    else:
        ac7_verdict = "❌ FAIL"
    print(f"AC7 verdict : {ac7_verdict}")

    # === Test 2 : relative error distribution ===
    median_abs_rel_err = metrics["median_abs_rel_err"]
    p99_abs_rel_err = metrics["p99_abs_rel_err"]
    p90_abs_rel_err = metrics["p90_abs_rel_err"]
    print("\n=== Bonus — relative error distribution ===")
    print(f"median |Δmdot_f| / mdot_f_qar : {median_abs_rel_err * 100:.1f} %  (target < 15 %)")
    print(f"p90    |Δmdot_f| / mdot_f_qar : {p90_abs_rel_err * 100:.1f} %")
    print(f"p99    |Δmdot_f| / mdot_f_qar : {p99_abs_rel_err * 100:.1f} %  (target < 50 %)")

    # === Stop Condition 3 : theory falsified if median |Δ| > 50 %
    theory_falsified = median_abs_rel_err > 0.50
    if theory_falsified:
        verdict_text = "🔬 STOP CONDITION 3 FIRED — Eq 19 theory falsified vs QAR"
    else:
        verdict_text = "✅ Theory tenable — Eq 19 reproduces QAR fuel flow to within median 50 %"
    print(f"\n{verdict_text}")

    # === Bonus : eta_PS distribution + mdot stats ===
    print("\n=== eta_PS distribution (cruise) ===")
    print(
        f"  min={eta_ps[valid].min():.3f}  median={np.median(eta_ps[valid]):.3f}  "
        f"max={eta_ps[valid].max():.3f}"
    )
    print("\n=== mdot_f distributions (cruise, kg/s) ===")
    print(
        f"  pred  median={np.median(mp):.3f}  p10={np.percentile(mp, 10):.3f}  p90={np.percentile(mp, 90):.3f}"
    )
    print(
        f"  qar   median={np.median(mq):.3f}  p10={np.percentile(mq, 10):.3f}  p90={np.percentile(mq, 90):.3f}"
    )
    print(f"  bias  pred/qar (median ratio) = {np.median(mp / mq):.3f}")

    # === Write report ===
    report = args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 — AC7 QAR fuel-flow validation",
        "",
        f"> The central scientific gate of Phase 4. n = {n} valid cruise-stable",
        f"> QAR samples across {len(all_data)} A320 flights.",
        "",
        "## Method",
        "",
        "For each cruise-stable QAR sample (alt > 8500 m, |dh/dt|<0.5 m/s, |dV/dt|<0.3 m/s²) :",
        "",
        "1. Reconstruct kinematic T-observable :",
        "   `T_obs = m·dV/dt + D_PS(C_L_qar) + m·g·sin γ`",
        "2. Compute analytical η_PS from PSEfficiencyLayer :",
        "   `η_PS(C_T_inst, M)` with `C_T_inst = T_obs / (q · S_ref)`",
        "3. Compute predicted fuel flow (Eq 19) :",
        "   `mdot_f_pred = T_obs · V / (η_PS · LCV)`",
        "4. Compare to ground truth :",
        "   `mdot_f_qar = (FUEL__FF_LEFT + FUEL__FF_RIGHT) / 3600`",
        "",
        "## Results — AC7 central gate",
        "",
        f"- Pearson corr(`mdot_f_pred`, `mdot_f_qar`) : **{corr_pearson:+.4f}**",
        "- Target : ≥ +0.50",
        f"- Verdict : **{ac7_verdict}**",
        "",
        "## Relative error distribution",
        "",
        f"- median |Δmdot_f| / mdot_f_qar : **{median_abs_rel_err * 100:.1f} %**  (target < 15 %)",
        f"- p90  : **{p90_abs_rel_err * 100:.1f} %**",
        f"- p99  : **{p99_abs_rel_err * 100:.1f} %**  (target < 50 %)",
        "",
        f"## Stop Condition 3 : **{verdict_text}**",
        "",
        "## Distributions (cruise, kg/s)",
        "",
        "| Quantity | median | p10 | p90 |",
        "|---|---:|---:|---:|",
        f"| mdot_f_pred | {np.median(mp):.3f} | {np.percentile(mp, 10):.3f} | {np.percentile(mp, 90):.3f} |",
        f"| mdot_f_qar  | {np.median(mq):.3f} | {np.percentile(mq, 10):.3f} | {np.percentile(mq, 90):.3f} |",
        "",
        f"Bias (median `mdot_f_pred / mdot_f_qar`) : **{np.median(mp / mq):.3f}** "
        "(1.00 = unbiased).",
        "",
        "## eta_PS distribution",
        "",
        f"- min : {eta_ps[valid].min():.3f}",
        f"- median : {np.median(eta_ps[valid]):.3f}",
        f"- max : {eta_ps[valid].max():.3f}",
        "- Target R8 cruise : ∈ [0.20, 0.45]",
        "",
        "## Methodology notes",
        "",
        "- The validation uses **kinematic T_obs** (reconstructed from QAR) ",
        "  as input to the η formula, not the NN-predicted T from v14. This ",
        "  isolates the **Eq 19 analytical structure** + the η_PS prior, ",
        "  independent of the v14 training quality.",
        "- C_L for the drag is computed from `m_qar · g / (q · S)` (level-flight ",
        "  identity), so AC7 measures the joint quality of PSDragLayer + ",
        "  PSEfficiencyLayer + Eq 19 against the most independent observable ",
        "  in the project (real QAR fuel flow).",
        "- R5-safe : no QAR sample enters training. This is validation only.",
    ]
    report.write_text("\n".join(lines))
    print(f"\nWrote {report}")

    return 0 if corr_pearson >= 0.50 and not theory_falsified else 1


if __name__ == "__main__":
    sys.exit(main())
