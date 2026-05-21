"""Phase 3 — Exp 15 : T_PS vs QAR T-observable diagnostic.

Tests stop condition 3 from PHASE_3_THRUST_PS_TICKET §4 :

> 🔬 Theory falsified on QAR : `T_PS` off de >30 % vs T-implicit observable
>  (`T = D + m·dV/dt + m·g·sin γ` reconstructed from QAR kinematics).

For each cruise-stable QAR sample, we reconstruct **T-observable**
from the kinematics + analytical drag prior, then ask :

    1. **Is the ML throttle inversion χ_ML ∈ [0.1, 1.0] ?**

       χ_ML = T_obs / (h_2(M_qar) · C_T_DO · q_qar · S_ref)

       If yes : the linear-throttle PSThrustLayer model has the
       *capacity* to represent T_obs by choice of throttle (the v13
       saturation is a hyperparameter / training problem, not a
       theory problem). If `χ_ML > 1.0` typically : T_PS literally
       cannot reach T_obs even at full throttle — theory falsified.

    2. **How does `χ_ML` compare to QAR ground truth
       `CTL__THROTT_POS_{L,R}` ?**

       This is a *bonus* test that wasn't in the original ticket —
       the QAR provides the actual pilot throttle setting, which
       should match `χ_ML` if PSThrustLayer is well-calibrated.

R5-safe : QAR is used for *validation only*, never for training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")

# A320 + CFM56-5B4_P (same constants as PSThrustLayer).
M_DO = 0.753
C_T_DO = 0.0347
S_REF = 122.6
C_T_RATIO_K = 0.55
G = 9.80665

# A320 polar (same as PSDragLayer).
PSI_0 = 7.846
SPAN_M = 34.10
S_REF_LIB = 122.4
SWEEP_DEG = 25.0
AR = SPAN_M * SPAN_M / S_REF_LIB  # ~9.50
MTF_AC = 0.87
J_1 = 11.0
J_2 = 1.0
E_LS = 0.778
K_INDUCED = 1.0 / (np.pi * AR * E_LS)
SKIN_A = 0.0269
SKIN_B = 0.14
L_REF = np.sqrt(S_REF)

# Sutherland's law.
MU_REF = 1.716e-5
T_REF_K = 273.15
S_SUTH = 110.4

R_GAS = 287.05
GAMMA_AIR = 1.4

# Unit conversions.
FT_PER_M = 3.28084
KTS_PER_MS = 1.9438445


def h_2(mach: np.ndarray) -> np.ndarray:
    """Eq 28 P&S Part 3 : (C_T)_etaB / (C_T)_DO as a function of Mach."""
    m_safe = np.maximum(mach, 0.1)
    num = 1.0 + C_T_RATIO_K * m_safe
    den = 1.0 + C_T_RATIO_K * M_DO
    return (num / den) * (M_DO / m_safe) ** 2


def t_ps_at_chi(chi: np.ndarray, mach: np.ndarray, q_pa: np.ndarray) -> np.ndarray:
    """Forward PSThrustLayer (same equations as layers/ps_thrust.py)."""
    return chi * h_2(mach) * C_T_DO * q_pa * S_REF


def c_d_ps(c_l: np.ndarray, mach: np.ndarray, temp_k: np.ndarray,
           q_pa: np.ndarray, tas_ms: np.ndarray) -> np.ndarray:
    """Analytical C_D Part 3 §4 — same as PSDragLayer."""
    temp_safe = np.maximum(temp_k, 180.0)
    mu = (
        MU_REF * (temp_safe / T_REF_K).clip(min=0.1) ** 1.5
        * (T_REF_K + S_SUTH) / (temp_safe + S_SUTH)
    )
    tas_safe = np.maximum(tas_ms, 50.0)
    rho = (2.0 * np.maximum(q_pa, 100.0)) / (tas_safe * tas_safe)
    re_ac = np.maximum(rho * tas_safe * L_REF / np.maximum(mu, 1e-7), 1e3)
    c_f = SKIN_A / re_ac ** SKIN_B
    c_d0 = PSI_0 * c_f
    cl_safe = np.clip(c_l, 0.0, 2.0)
    cs = np.cos(np.radians(SWEEP_DEG))
    m_cc = MTF_AC - 0.10 * cl_safe / (cs * cs)
    m_cc_safe = np.maximum(m_cc, 0.30)
    x = mach * cs / m_cc_safe
    c_dw = (cs ** 3) * J_1 * np.maximum(x - J_2, 0.0) ** 2
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
    """Return per-cruise-sample arrays for the diagnostic."""
    df = pl.read_parquet(parquet_path)
    # Skip flights without required columns.
    needed = {"ALT__STD", "SPD__TAS", "SPD__MACH", "SYS__GW",
              "ATT__PITCH", "TEMP__SAT",
              "CTL__THROTT_ANGL", "CTL__THROTT_ANGR"}
    if not needed.issubset(df.columns):
        return None

    # Filter to cruise-stable : alt > 28 000 ft, smooth speed and altitude.
    df = df.with_columns([
        (pl.col("ALT__STD") / FT_PER_M).alias("alt_m"),
        (pl.col("SPD__TAS") / KTS_PER_MS).alias("tas_ms"),
        (pl.col("TEMP__SAT") + 273.15).alias("temp_k"),
    ])
    # Add derivatives (1 Hz sampling assumed).
    df = df.with_columns([
        pl.col("alt_m").diff().alias("dalt_m_s"),
        pl.col("tas_ms").diff().alias("dtas_ms2"),
        pl.col("ATT__PITCH").diff().alias("dpitch_rad"),
    ])
    df = df.drop_nulls(["alt_m", "tas_ms", "dtas_ms2", "dalt_m_s", "SPD__MACH"])
    # Cruise mask : alt > 28 kft, smooth d_alt + d_tas + finite Mach.
    df_cruise = df.filter(
        (pl.col("alt_m") > 8500.0)
        & (pl.col("alt_m") < 12500.0)
        & (pl.col("dalt_m_s").abs() < 0.5)  # |dh/dt| < 0.5 m/s (level)
        & (pl.col("dtas_ms2").abs() < 0.3)  # |dV/dt| < 0.3 m/s² (steady)
        & (pl.col("SPD__MACH") > 0.6)
        & (pl.col("SPD__MACH") < 0.85)
        & (pl.col("SYS__GW") > 40000)
        & (pl.col("SYS__GW") < 80000)
    )
    if len(df_cruise) < 50:
        return None

    alt_m = df_cruise["alt_m"].to_numpy()
    tas_ms = df_cruise["tas_ms"].to_numpy()
    mach = df_cruise["SPD__MACH"].to_numpy()
    mass_kg = df_cruise["SYS__GW"].to_numpy()
    temp_k = df_cruise["temp_k"].to_numpy()
    pitch_rad = df_cruise["ATT__PITCH"].to_numpy()
    dalt = df_cruise["dalt_m_s"].to_numpy()
    dtas = df_cruise["dtas_ms2"].to_numpy()
    throt_l = df_cruise["CTL__THROTT_ANGL"].to_numpy().astype(np.float64)
    throt_r = df_cruise["CTL__THROTT_ANGR"].to_numpy().astype(np.float64)

    # Flight-path angle from pitch and TAS climb rate (small-angle).
    # gamma = arcsin(dh/dt / V) but at cruise dh/dt ~ 0 so gamma ~ 0.
    gamma = np.arcsin(np.clip(dalt / np.maximum(tas_ms, 50.0), -0.3, 0.3))

    # Dynamic pressure from ISA (since SAT gives true T, p from ISA at alt).
    _, p_pa = isa_temp_pressure(alt_m)
    rho = p_pa / (R_GAS * temp_k)
    q_pa = 0.5 * rho * tas_ms * tas_ms

    # Analytical drag (Phase 2 PSDragLayer)
    c_l = mass_kg * G / (q_pa * S_REF)  # level-flight CL
    cd = c_d_ps(c_l, mach, temp_k, q_pa, tas_ms)
    drag_n = q_pa * S_REF * cd

    # T-observable from kinematics : T = m·dV/dt + D + m·g·sin γ
    t_obs = mass_kg * dtas + drag_n + mass_kg * G * np.sin(gamma)

    # T_PS at χ=1 (max envelope per the analytical model)
    t_ps_chi1 = t_ps_at_chi(np.ones_like(mach), mach, q_pa)

    # Inverse : χ_ML = T_obs / T_PS(χ=1)
    chi_ml = t_obs / np.maximum(t_ps_chi1, 1.0)

    # Ground truth throttle from QAR (mean of L/R, normalised to [0, 1]).
    # CTL__THROTT_POS_{L,R} is typically in degrees [-20, 60] or %; needs
    # range check + normalisation.
    throt_avg = 0.5 * (throt_l + throt_r)

    return {
        "alt_m": alt_m,
        "tas_ms": tas_ms,
        "mach": mach,
        "mass_kg": mass_kg,
        "q_pa": q_pa,
        "drag_n": drag_n,
        "t_obs_n": t_obs,
        "t_ps_chi1_n": t_ps_chi1,
        "chi_ml": chi_ml,
        "throt_qar": throt_avg,
    }


def main() -> int:
    paths = sorted(QAR_DIR.glob("*A320*.parquet"))
    print(f"Found {len(paths)} A320 QAR files")
    if not paths:
        print("No QAR files — abort")
        return 1

    all_data: list[dict[str, np.ndarray]] = []
    for i, p in enumerate(paths):
        if i % 50 == 0:
            print(f"  [{i+1}/{len(paths)}] {p.name}")
        d = process_flight(p)
        if d is not None:
            all_data.append(d)
    if not all_data:
        print("No cruise-stable flights — abort")
        return 1

    chi_ml = np.concatenate([d["chi_ml"] for d in all_data])
    t_obs = np.concatenate([d["t_obs_n"] for d in all_data])
    t_ps_chi1 = np.concatenate([d["t_ps_chi1_n"] for d in all_data])
    throt_qar = np.concatenate([d["throt_qar"] for d in all_data])
    mach = np.concatenate([d["mach"] for d in all_data])
    drag_n = np.concatenate([d["drag_n"] for d in all_data])

    n = len(chi_ml)
    print(f"\n{n} cruise-stable samples across {len(all_data)} flights")
    print()

    # --- Test 1 : is χ_ML ∈ [0.1, 1.0] ?
    in_range = (chi_ml >= 0.1) & (chi_ml <= 1.0)
    pct_in_range = 100.0 * in_range.sum() / n
    print(f"=== Stop Condition 3 (T-PS theory test) ===")
    print(f"chi_ML in [0.1, 1.0] : {pct_in_range:.1f} % of samples")
    print(f"chi_ML stats : median={np.median(chi_ml):.3f}  "
          f"p10={np.percentile(chi_ml, 10):.3f}  "
          f"p90={np.percentile(chi_ml, 90):.3f}  "
          f"max={chi_ml.max():.3f}")
    print()
    # Compute T_PS at chi_ML and compare to T_obs (ought to be ~equal where chi_ML ∈ [0.1, 1])
    t_ps_at_ml = np.clip(chi_ml, 0.0, 1.0) * t_ps_chi1
    rel_err_pct = (t_ps_at_ml - t_obs) / np.maximum(np.abs(t_obs), 1.0) * 100.0
    median_abs_err = float(np.median(np.abs(rel_err_pct)))
    print(f"With chi clamped to [0, 1] : median |T_PS - T_obs| / |T_obs| = {median_abs_err:.1f} %")
    print(f"  (clamping kicks in for {100*(~in_range).sum()/n:.1f} % of samples where chi_ML > 1)")
    print()
    # On samples where chi_ML is in range, T_PS = T_obs by construction, err = 0.
    # On samples where chi_ML > 1, clamping caps T_PS at chi=1·(C_T)_ηB, so err = T_obs - T_PS_max.
    deficient = chi_ml > 1.0
    if deficient.sum() > 0:
        deficit_pct = (t_obs[deficient] - t_ps_chi1[deficient]) / np.maximum(t_obs[deficient], 1.0) * 100.0
        print(f"Samples where T_obs > T_PS(chi=1) : {deficient.sum()} ({100*deficient.sum()/n:.1f} %)")
        print(f"  T_obs deficit vs T_PS_max : median {np.median(deficit_pct):.1f} % , p90 {np.percentile(deficit_pct, 90):.1f} %")
    print()

    # --- Stop Condition 3 verdict
    # Original wording : T_PS off de >30 % vs T_obs.
    # Operationalised : if >50 % of cruise samples have chi_ML > 1
    #                   OR median |T_PS_at_ml - T_obs| > 30 %
    #                   the linear-throttle T_PS literally cannot represent reality.
    chi_oob_pct = 100.0 * deficient.sum() / n
    theory_falsified = (chi_oob_pct > 50.0) or (median_abs_err > 30.0)
    if theory_falsified:
        verdict = "🔬 STOP CONDITION 3 FIRED — theory falsified"
    else:
        verdict = "✅ Theory tenable — T_PS can represent T_obs by throttle choice"
    print(verdict)
    print()

    # --- Bonus : compare chi_ML to QAR ground truth throttle
    # Try to detect throttle units. QAR throttle is often 0-100% or pilot angle [0, 60].
    print(f"=== Bonus : chi_ML vs QAR ground-truth throttle ===")
    print(f"QAR CTL__THROTT_POS_(L,R) avg stats:")
    print(f"  min={throt_qar.min():.2f}  median={np.median(throt_qar):.2f}  "
          f"max={throt_qar.max():.2f}")
    # Normalize to [0, 1] if it looks like percent.
    if throt_qar.max() > 1.5:
        # Assume degrees [0, ~70] for pilot lever angle.
        throt_norm = (throt_qar - throt_qar.min()) / (throt_qar.max() - throt_qar.min() + 1e-6)
        print("  Normalising QAR throttle (range > 1.5, treat as scaled angle).")
    else:
        throt_norm = throt_qar
    # Sample stats
    good = (chi_ml > 0.05) & (chi_ml < 1.5) & (throt_norm > 0) & (throt_norm < 1.0)
    if good.sum() > 100:
        corr = float(np.corrcoef(chi_ml[good], throt_norm[good])[0, 1])
        print(f"  pearson corr(chi_ML, throt_qar_norm) on {good.sum()} samples = {corr:+.3f}")
        print(f"  chi_ML median where good = {np.median(chi_ml[good]):.3f}")
        print(f"  throt_qar_norm median where good = {np.median(throt_norm[good]):.3f}")

    # --- Write report
    report = Path("data/models/_comparison/phase3_thrust_v13_qar_t_observable.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 3 — Exp 15 : QAR T-observable diagnostic",
        "",
        f"> Stop condition 3 test. n = {n} cruise-stable QAR samples across",
        f"> {len(all_data)} A320 flights.",
        "",
        "## Method",
        "",
        "For each cruise-stable QAR sample (alt > 8500 m, |dh/dt|<0.5 m/s, |dV/dt|<0.3 m/s²) :",
        "1. Reconstruct T-observable : `T_obs = m·dV/dt + D_PS(C_L_qar) + m·g·sin γ`",
        "2. Compute T_PS at χ=1 (max envelope) and back-solve χ_ML = T_obs / T_PS_max",
        "3. Test whether χ_ML ∈ [0.1, 1.0] (theory tenable) or χ_ML > 1 (theory falsified)",
        "",
        "## Results",
        "",
        f"- χ_ML in [0.1, 1.0] : **{pct_in_range:.1f} %** of samples",
        f"- χ_ML stats : median {np.median(chi_ml):.3f}, p10 {np.percentile(chi_ml, 10):.3f}, p90 {np.percentile(chi_ml, 90):.3f}, max {chi_ml.max():.3f}",
        f"- Samples with χ_ML > 1 (T_obs > T_PS_max) : **{chi_oob_pct:.1f} %**",
        f"- Median |T_PS clamped - T_obs| / |T_obs| : **{median_abs_err:.1f} %**",
        "",
        f"## Verdict",
        "",
        f"**{verdict}**",
        "",
    ]
    if theory_falsified:
        lines += [
            "More than 50 % of samples need χ > 1 to match T_obs, or median",
            "error after clamping exceeds 30 %. The linear-throttle PSThrustLayer",
            "model **cannot** represent QAR T-observable across the cruise envelope.",
            "Phase 3 should not proceed without upgrading T_PS (Eqs E3-E5).",
        ]
    else:
        lines += [
            "The linear-throttle PSThrustLayer can represent T-observable across",
            "most of the cruise envelope by appropriate choice of χ ∈ [0.1, 1.0].",
            "The v13 val_loss gap (+21.5 %) is therefore not a *theory* problem",
            "but a training / hyperparameter problem (the ±5 % NN correction bound",
            "is too tight to absorb the deviation between learned χ and ML χ).",
        ]

    report.write_text("\n".join(lines))
    print(f"\nWrote {report}")

    return 0 if not theory_falsified else 3  # exit code 3 = stop condition 3 fired


if __name__ == "__main__":
    sys.exit(main())
