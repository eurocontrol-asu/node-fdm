"""Phase 1.8a — Pre-validation gate for P&S mass inversion.

Strategy A : single-window inversion. For each cruise-stable QAR flight,
extract the first stable 60-s cruise window, then solve

    min_(chi, m)  Σ_i (chi · h_2(M_i) · C_T_DO · q_i · S
                       - q_i · S · C_D(m·g/(q_i·S), M_i)) ** 2

over (chi, m) bounded by [0.1, 1.0] × [OEW, MTOW] via L-BFGS-B.

Compare `m_inv` to `SYS__GW` ground truth at the same window start.

R5-safe : QAR is used ONLY to measure the proxy quality (corr).
The mass inversion itself uses ONLY ADS-B-equivalent observables
(q, mach) — no QAR signal enters the inversion formula. This script
does NOT inject QAR into any training pipeline ; it answers
"is the inversion accurate enough to be worth wiring into the
training loader?" before we commit to retraining.

Verdict thresholds (cf. ticket §6) :
    corr ≥ +0.60 : Strategy A viable, proceed to retrain v15.
    +0.40 ≤ corr < +0.60 : Strategy A marginal, escalate to B.
    corr < +0.40 : Strategy A insufficient, escalate to B without retrain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

# Resolve project root and path for shared P&S helpers.
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the numpy P&S helpers from the AC7 script (h_2, c_d_ps, isa_temp_pressure,
# constants). They are textbook (η_o_DO=0.309, etc.) per R3.
from phase4_fuelflow_qar_validation import (  # noqa: E402
    C_T_DO,
    FT_PER_M,
    G,
    KTS_PER_MS,
    M_DO,
    R_GAS,
    S_REF,
    c_d_ps,
    h_2,
    isa_temp_pressure,
)

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")

# A320 mass bounds (TCDS).
A320_OEW = 40_000.0  # ~OEW + reserves bound for inversion (loose)
A320_MTOW = 80_000.0


def invert_m_chi_single_window(
    q: np.ndarray,
    mach: np.ndarray,
    temp_k: np.ndarray,
    tas_ms: np.ndarray,
) -> tuple[float, float, float]:
    """Solve Newton in stable cruise for (chi, m).

    Cost function : Σ_i (T_PS(chi, M_i, q_i) − D_PS(m·g/(q_i·S), M_i)) ** 2 / σ²
    with chi ∈ [0.1, 1.0] and m ∈ [OEW, MTOW].

    Returns (chi_inv, m_inv, residual).
    """
    def residual(params: np.ndarray) -> float:
        chi, m = params
        t_pred = chi * h_2(mach) * C_T_DO * q * S_REF
        c_l = m * G / np.maximum(q * S_REF, 1.0)
        c_d = c_d_ps(c_l, mach, temp_k, q, tas_ms)
        d_pred = q * S_REF * c_d
        return float(np.sum((t_pred - d_pred) ** 2))

    # Initial guess : cruise-typical (Phase 3 AC6 measured chi_cruise ~0.84,
    # mean A320 cruise mass ~62 t).
    x0 = np.array([0.80, 62_000.0])
    bounds = [(0.1, 1.0), (A320_OEW, A320_MTOW)]
    result = minimize(residual, x0, method="L-BFGS-B", bounds=bounds)
    return float(result.x[0]), float(result.x[1]), float(result.fun)


def _extract_phase_window(
    df: pl.DataFrame,
    *,
    alt_low: float,
    alt_high: float,
    direction: str,  # "climb", "cruise", "descent"
    n_samples: int = 30,
) -> dict[str, np.ndarray] | None:
    """Extract a smoothed window with sufficient samples in the given altitude band + direction."""
    # Smooth dh/dt and dV/dt with running mean (10 s window).
    df_sm = df.with_columns([
        pl.col("dalt_m_s").rolling_mean(window_size=10).alias("dalt_smooth"),
        pl.col("dtas_ms2").rolling_mean(window_size=10).alias("dtas_smooth"),
    ])
    if direction == "climb":
        mask = (
            (pl.col("alt_m") >= alt_low)
            & (pl.col("alt_m") <= alt_high)
            & (pl.col("dalt_smooth") > 1.0)
            & (pl.col("SPD__MACH") > 0.3)
            & (pl.col("SPD__MACH") < 0.85)
        )
    elif direction == "cruise":
        mask = (
            (pl.col("alt_m") >= alt_low)
            & (pl.col("alt_m") <= alt_high)
            & (pl.col("dalt_smooth").abs() < 0.5)
            & (pl.col("dtas_smooth").abs() < 0.3)
            & (pl.col("SPD__MACH") > 0.6)
            & (pl.col("SPD__MACH") < 0.85)
        )
    elif direction == "descent":
        mask = (
            (pl.col("alt_m") >= alt_low)
            & (pl.col("alt_m") <= alt_high)
            & (pl.col("dalt_smooth") < -1.0)
            & (pl.col("SPD__MACH") > 0.3)
            & (pl.col("SPD__MACH") < 0.85)
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")

    df_sel = df_sm.filter(mask)
    if len(df_sel) < n_samples:
        return None
    win = df_sel.head(n_samples)
    alt_m = win["alt_m"].to_numpy()
    tas_ms = win["tas_ms"].to_numpy()
    mach = win["SPD__MACH"].to_numpy()
    temp_k = win["temp_k"].to_numpy()
    dalt = win["dalt_smooth"].to_numpy()
    dtas = win["dtas_smooth"].to_numpy()
    gamma = np.arcsin(np.clip(dalt / np.maximum(tas_ms, 50.0), -0.3, 0.3))
    _, p_pa = isa_temp_pressure(alt_m)
    rho = p_pa / (R_GAS * temp_k)
    q_pa = 0.5 * rho * tas_ms * tas_ms
    return {
        "alt_m": alt_m,
        "tas_ms": tas_ms,
        "mach": mach,
        "temp_k": temp_k,
        "q_pa": q_pa,
        "gamma": gamma,
        "dtas": dtas,
    }


def invert_m_chi_multiphase(
    climb: dict[str, np.ndarray] | None,
    cruise: dict[str, np.ndarray] | None,
    descent: dict[str, np.ndarray] | None,
    *,
    normalize_per_phase: bool = True,
) -> tuple[float, float, float, float, dict[str, float]]:
    """Solve Newton across climb + cruise + descent for (chi_clb, chi_cru, chi_des, m).

    With `normalize_per_phase=True` (Strategy C v2), residuals are divided
    by a typical T magnitude per phase (climb T ~ 80 kN, cruise T ~ 30 kN,
    descent T ~ 10 kN) before squaring. This prevents climb's large
    absolute residual from dominating the optimization.
    """
    # Typical T_PS magnitudes per phase (used as scaling factors).
    t_typical_climb = 80_000.0
    t_typical_cruise = 35_000.0
    t_typical_descent = 12_000.0

    def newton_residual_phase(
        phase: dict[str, np.ndarray] | None,
        chi: float,
        m: float,
        is_cruise: bool,
    ) -> np.ndarray:
        if phase is None:
            return np.array([0.0])
        q = phase["q_pa"]
        mach = phase["mach"]
        temp_k = phase["temp_k"]
        tas_ms = phase["tas_ms"]
        gamma = phase["gamma"]
        dtas = phase["dtas"]
        t_pred = chi * h_2(mach) * C_T_DO * q * S_REF
        c_l = m * G / np.maximum(q * S_REF, 1.0)
        c_d = c_d_ps(c_l, mach, temp_k, q, tas_ms)
        d_pred = q * S_REF * c_d
        if is_cruise:
            return t_pred - d_pred
        return t_pred - d_pred - m * (dtas + G * np.sin(gamma))

    def total_residual(params: np.ndarray) -> float:
        chi_clb, chi_cru, chi_des, m = params
        r_clb = newton_residual_phase(climb, chi_clb, m, is_cruise=False)
        r_cru = newton_residual_phase(cruise, chi_cru, m, is_cruise=True)
        r_des = newton_residual_phase(descent, chi_des, m, is_cruise=False)
        n_clb = len(r_clb) if climb is not None else 1
        n_cru = len(r_cru) if cruise is not None else 1
        n_des = len(r_des) if descent is not None else 1
        if normalize_per_phase:
            r_clb = r_clb / t_typical_climb
            r_cru = r_cru / t_typical_cruise
            r_des = r_des / t_typical_descent
        return (
            float(np.sum(r_clb ** 2)) / max(n_clb, 1)
            + float(np.sum(r_cru ** 2)) / max(n_cru, 1)
            + float(np.sum(r_des ** 2)) / max(n_des, 1)
        )

    x0 = np.array([0.90, 0.80, 0.30, 62_000.0])
    bounds = [(0.5, 1.0), (0.5, 1.0), (0.05, 0.5), (A320_OEW, A320_MTOW)]
    result = minimize(total_residual, x0, method="L-BFGS-B", bounds=bounds)
    chi_clb, chi_cru, chi_des, m_inv = result.x
    diag = {
        "n_climb": len(climb["mach"]) if climb is not None else 0,
        "n_cruise": len(cruise["mach"]) if cruise is not None else 0,
        "n_descent": len(descent["mach"]) if descent is not None else 0,
        "residual": float(result.fun),
        "n_total": (
            (len(climb["mach"]) if climb is not None else 0)
            + (len(cruise["mach"]) if cruise is not None else 0)
            + (len(descent["mach"]) if descent is not None else 0)
        ),
    }
    return float(chi_clb), float(chi_cru), float(chi_des), float(m_inv), diag


def process_flight_for_m_inv_strategy_c(parquet_path: Path) -> dict[str, float] | None:
    """Strategy C : climb + cruise + descent multi-phase inversion."""
    df = pl.read_parquet(parquet_path)
    needed = {"ALT__STD", "SPD__TAS", "SPD__MACH", "SYS__GW",
              "ATT__PITCH", "TEMP__SAT"}
    if not needed.issubset(df.columns):
        return None
    df = df.with_columns([
        (pl.col("ALT__STD") / FT_PER_M).alias("alt_m"),
        (pl.col("SPD__TAS") / KTS_PER_MS).alias("tas_ms"),
        (pl.col("TEMP__SAT") + 273.15).alias("temp_k"),
    ])
    df = df.with_columns([
        pl.col("alt_m").diff().alias("dalt_m_s"),
        pl.col("tas_ms").diff().alias("dtas_ms2"),
    ])
    df = df.drop_nulls(["alt_m", "tas_ms", "dtas_ms2", "dalt_m_s", "SPD__MACH"])

    climb = _extract_phase_window(df, alt_low=3000.0, alt_high=8000.0, direction="climb", n_samples=30)
    cruise = _extract_phase_window(df, alt_low=8500.0, alt_high=12500.0, direction="cruise", n_samples=30)
    descent = _extract_phase_window(df, alt_low=3000.0, alt_high=8000.0, direction="descent", n_samples=30)

    # Need at least 2 phases to break degeneracy.
    n_phases = sum(p is not None for p in (climb, cruise, descent))
    if n_phases < 2:
        return None

    # Ground truth : take SYS__GW at climb start (or cruise start if no climb).
    if climb is not None:
        # Find first sample where alt_m ≈ climb start ; SYS__GW already filtered.
        df_climb_start = df.filter(
            (pl.col("alt_m") >= 3000.0)
            & (pl.col("alt_m") <= 4000.0)
            & (pl.col("SYS__GW") > 40000)
        )
        if len(df_climb_start) > 0:
            m_qar = float(df_climb_start["SYS__GW"].head(1).to_numpy()[0])
        else:
            return None
    else:
        df_cru_start = df.filter(
            (pl.col("alt_m") >= 8500.0)
            & (pl.col("alt_m") <= 9500.0)
            & (pl.col("SYS__GW") > 40000)
        )
        if len(df_cru_start) > 0:
            m_qar = float(df_cru_start["SYS__GW"].head(1).to_numpy()[0])
        else:
            return None

    chi_clb, chi_cru, chi_des, m_inv, diag = invert_m_chi_multiphase(climb, cruise, descent)

    return {
        "m_inv": m_inv,
        "m_qar": m_qar,
        "chi_clb": chi_clb,
        "chi_cru": chi_cru,
        "chi_des": chi_des,
        "residual": diag["residual"],
        "n_climb": float(diag["n_climb"]),
        "n_cruise": float(diag["n_cruise"]),
        "n_descent": float(diag["n_descent"]),
        "n_total": float(diag["n_total"]),
    }


def _extract_cruise_windows_separated(
    df: pl.DataFrame,
    *,
    n_windows_target: int = 5,
    window_samples: int = 30,
    min_separation_s: float = 1200.0,
) -> list[dict[str, np.ndarray]]:
    """Find up to N cruise-stable windows separated by ≥ min_separation_s."""
    df_sm = df.with_columns([
        pl.col("alt_m").diff().alias("dalt_m_s"),
        pl.col("tas_ms").diff().alias("dtas_ms2"),
    ])
    df_sm = df_sm.drop_nulls(["dalt_m_s", "dtas_ms2"])
    df_cruise = df_sm.filter(
        (pl.col("alt_m") > 8500.0)
        & (pl.col("alt_m") < 12500.0)
        & (pl.col("dalt_m_s").abs() < 0.5)
        & (pl.col("dtas_ms2").abs() < 0.3)
        & (pl.col("SPD__MACH") > 0.6)
        & (pl.col("SPD__MACH") < 0.85)
    )
    if len(df_cruise) < window_samples * n_windows_target:
        return []
    # Add a row index so we can space windows by sample count (≈ time).
    df_cruise = df_cruise.with_row_index("row_idx")
    windows = []
    last_idx = -1e9
    arr_row_idx = df_cruise["row_idx"].to_numpy()
    n_total = len(arr_row_idx)
    i = 0
    while i < n_total - window_samples and len(windows) < n_windows_target:
        cur_row_idx = arr_row_idx[i]
        if cur_row_idx - last_idx < min_separation_s:
            i += 1
            continue
        # Take this window.
        win = df_cruise.slice(i, window_samples)
        alt_m = win["alt_m"].to_numpy()
        tas_ms = win["tas_ms"].to_numpy()
        mach = win["SPD__MACH"].to_numpy()
        temp_k = win["temp_k"].to_numpy()
        _, p_pa = isa_temp_pressure(alt_m)
        rho = p_pa / (R_GAS * temp_k)
        q_pa = 0.5 * rho * tas_ms * tas_ms
        windows.append({
            "alt_m": alt_m,
            "tas_ms": tas_ms,
            "mach": mach,
            "temp_k": temp_k,
            "q_pa": q_pa,
            "t_center": float(cur_row_idx) + window_samples / 2.0,
        })
        last_idx = cur_row_idx
        i += window_samples
    return windows


# eta_o(C_T, M) inline (avoid circular import).
ETA_O_DO_PHASE_B = 0.309  # textbook (R3 strict)
BPR_PHASE_B = 5.6
ETA_2_PHASE_B = 0.65 * (1.0 - 0.035 * BPR_PHASE_B)
H0_CURVATURE_PHASE_B = 0.43
LCV_PHASE_B = 43.0e6


def _eta_o_local(c_t: np.ndarray, mach: np.ndarray) -> np.ndarray:
    m_safe = np.maximum(mach, 0.1)
    h_1 = (m_safe / M_DO) ** ETA_2_PHASE_B
    eta_o_b = h_1 * ETA_O_DO_PHASE_B
    c_t_eta_b = h_2(m_safe) * C_T_DO
    ratio = c_t / np.maximum(c_t_eta_b, 1e-6)
    omega = np.where(m_safe >= 0.4, 0.0,
                     np.where(m_safe >= 0.2, 1.30 * (0.4 - m_safe),
                              1.30 * (0.4 - 0.2)))
    arg = ratio - 1.0
    h_0 = (1.0 - H0_CURVATURE_PHASE_B * arg * arg) * (1.0 + omega * arg * arg)
    return h_0 * eta_o_b


def invert_m_chi_multi_window_chained(
    windows: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, float, float, int]:
    """Solve Newton across K cruise windows with chained mass loss linkage.

    Unknowns : (chi_1, ..., chi_K, m_0). K+1 total.
    Residual : Σ_k Σ_i (T_k_i - D_k_i(m_k))² / N_k
        where m_k = m_0 - Σ_{j<k} mdot_f_pred(chi_j, m_j, V_j, ...) · Δt_jk.
    """
    K = len(windows)
    if K < 2:
        return np.array([0.0] * (K + 1)), float("inf"), 0.0, K

    def cumulative_burn(chis: np.ndarray, m_0: float) -> np.ndarray:
        """Return m_k for k=0..K-1 given chi_1..K and m_0."""
        m_arr = np.zeros(K)
        m_arr[0] = m_0
        for k in range(K - 1):
            w = windows[k]
            # mean mdot_f over window k applied over Δt = t_{k+1} - t_k (in seconds, ≈ row indices @ 1Hz).
            mach = w["mach"]
            q = w["q_pa"]
            tas = w["tas_ms"]
            t_max_pred = chis[k] * h_2(mach) * C_T_DO * q * S_REF
            c_t_inst = t_max_pred / np.maximum(q * S_REF, 1.0)
            eta_ps = _eta_o_local(c_t_inst, mach)
            mdot_f = t_max_pred * tas / np.maximum(eta_ps * LCV_PHASE_B, 1e-6)
            mdot_mean = float(np.mean(mdot_f))
            dt = windows[k + 1]["t_center"] - w["t_center"]  # ≈ seconds
            m_arr[k + 1] = m_arr[k] - mdot_mean * dt
        return m_arr

    def residual_full(params: np.ndarray) -> float:
        chis = params[:K]
        m_0 = params[K]
        m_arr = cumulative_burn(chis, m_0)
        total = 0.0
        for k, w in enumerate(windows):
            q = w["q_pa"]
            mach = w["mach"]
            temp_k = w["temp_k"]
            tas = w["tas_ms"]
            t_pred = chis[k] * h_2(mach) * C_T_DO * q * S_REF
            c_l = m_arr[k] * G / np.maximum(q * S_REF, 1.0)
            c_d = c_d_ps(c_l, mach, temp_k, q, tas)
            d_pred = q * S_REF * c_d
            total += float(np.sum((t_pred - d_pred) ** 2)) / max(len(q), 1)
        return total

    x0 = np.array([0.80] * K + [62_000.0])
    bounds = [(0.1, 1.0)] * K + [(A320_OEW, A320_MTOW)]
    result = minimize(residual_full, x0, method="L-BFGS-B", bounds=bounds)
    return result.x, float(result.fun), float(result.x[K]), K


def process_flight_for_m_inv_strategy_b(parquet_path: Path) -> dict[str, float] | None:
    """Strategy B : multi-window cruise + chained mass-loss inversion."""
    df = pl.read_parquet(parquet_path)
    needed = {"ALT__STD", "SPD__TAS", "SPD__MACH", "SYS__GW",
              "ATT__PITCH", "TEMP__SAT"}
    if not needed.issubset(df.columns):
        return None
    df = df.with_columns([
        (pl.col("ALT__STD") / FT_PER_M).alias("alt_m"),
        (pl.col("SPD__TAS") / KTS_PER_MS).alias("tas_ms"),
        (pl.col("TEMP__SAT") + 273.15).alias("temp_k"),
    ])

    windows = _extract_cruise_windows_separated(df, n_windows_target=5, window_samples=30, min_separation_s=600.0)
    if len(windows) < 3:
        return None

    # Ground truth : SYS__GW at top of climb (first cruise sample).
    df_toc = df.filter(
        (pl.col("alt_m") >= 8500.0)
        & (pl.col("alt_m") <= 10500.0)
        & (pl.col("SYS__GW") > 40000)
    )
    if len(df_toc) == 0:
        return None
    m_qar = float(df_toc["SYS__GW"].head(1).to_numpy()[0])

    params, residual, m_0_inv, K = invert_m_chi_multi_window_chained(windows)
    chi_array = params[:K]

    return {
        "m_inv": m_0_inv,
        "m_qar": m_qar,
        "chi_mean": float(np.mean(chi_array)),
        "chi_min": float(np.min(chi_array)),
        "chi_max": float(np.max(chi_array)),
        "residual": residual,
        "n_windows": float(K),
        "t_span_s": float(windows[-1]["t_center"] - windows[0]["t_center"]),
    }


def process_flight_for_m_inv(parquet_path: Path) -> dict[str, float] | None:
    """Return (m_inv, m_qar, chi_inv, n_samples) for the first cruise window."""
    df = pl.read_parquet(parquet_path)
    needed = {"ALT__STD", "SPD__TAS", "SPD__MACH", "SYS__GW",
              "ATT__PITCH", "TEMP__SAT"}
    if not needed.issubset(df.columns):
        return None

    df = df.with_columns([
        (pl.col("ALT__STD") / FT_PER_M).alias("alt_m"),
        (pl.col("SPD__TAS") / KTS_PER_MS).alias("tas_ms"),
        (pl.col("TEMP__SAT") + 273.15).alias("temp_k"),
    ])
    df = df.with_columns([
        pl.col("alt_m").diff().alias("dalt_m_s"),
        pl.col("tas_ms").diff().alias("dtas_ms2"),
    ])
    df = df.drop_nulls(["alt_m", "tas_ms", "dtas_ms2", "dalt_m_s", "SPD__MACH"])

    df_cruise = df.filter(
        (pl.col("alt_m") > 8500.0)
        & (pl.col("alt_m") < 12500.0)
        & (pl.col("dalt_m_s").abs() < 0.5)
        & (pl.col("dtas_ms2").abs() < 0.3)
        & (pl.col("SPD__MACH") > 0.6)
        & (pl.col("SPD__MACH") < 0.85)
        & (pl.col("SYS__GW") > 40000)
        & (pl.col("SYS__GW") < 80000)
    )
    if len(df_cruise) < 60:  # need at least 60 samples for stable window
        return None

    # Take the first 60-sample window (60 s @ 1 Hz typical).
    win = df_cruise.head(60)
    alt_m = win["alt_m"].to_numpy()
    tas_ms = win["tas_ms"].to_numpy()
    mach = win["SPD__MACH"].to_numpy()
    mass_kg = win["SYS__GW"].to_numpy()
    temp_k = win["temp_k"].to_numpy()

    # Dynamic pressure from ISA + observed temp (same as Phase 4 AC7).
    _, p_pa = isa_temp_pressure(alt_m)
    rho = p_pa / (R_GAS * temp_k)
    q_pa = 0.5 * rho * tas_ms * tas_ms

    # Run the inversion.
    chi_inv, m_inv, residual = invert_m_chi_single_window(q_pa, mach, temp_k, tas_ms)

    # Ground truth = mean SYS__GW over the window (mass varies < 0.1 % per 60 s).
    m_qar = float(np.mean(mass_kg))

    return {
        "m_inv": m_inv,
        "m_qar": m_qar,
        "chi_inv": chi_inv,
        "residual": residual,
        "alt_m_mean": float(np.mean(alt_m)),
        "mach_mean": float(np.mean(mach)),
        "n_samples": 60,
    }


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(
        description="Phase 1.8a pre-validation gate: corr(m_inv, m_qar) on 517 QAR flights"
    )
    parser.add_argument(
        "--strategy", choices=["A", "B", "C"], default="A",
        help="Inversion strategy (only A implemented in this script)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = Path(f"data/investigations/phase1_8a_ps_mass_inversion/artifacts/preval_strategy_{args.strategy.lower()}.md")
    if args.out_csv is None:
        args.out_csv = Path(f"data/investigations/phase1_8a_ps_mass_inversion/artifacts/preval_strategy_{args.strategy.lower()}.csv")

    if args.strategy not in ("A", "B", "C"):
        print(f"Strategy {args.strategy} not yet implemented (A, B, C).")
        return 1

    paths = sorted(QAR_DIR.glob("*A320*.parquet"))
    print(f"Found {len(paths)} A320 QAR files (Strategy {args.strategy})")
    if not paths:
        print("No QAR files — abort")
        return 1

    if args.strategy == "A":
        process_fn = process_flight_for_m_inv
    elif args.strategy == "B":
        process_fn = process_flight_for_m_inv_strategy_b
    else:
        process_fn = process_flight_for_m_inv_strategy_c

    rows: list[dict[str, float]] = []
    n_skipped_no_cols = 0
    n_skipped_no_cruise = 0
    for i, p in enumerate(paths):
        if i % 50 == 0:
            print(f"  [{i+1}/{len(paths)}] {p.name}")
        d = process_fn(p)
        if d is None:
            try:
                df_head = pl.read_parquet(p, n_rows=1)
                if {"ALT__STD", "SPD__TAS", "SPD__MACH", "SYS__GW",
                    "ATT__PITCH", "TEMP__SAT"}.issubset(df_head.columns):
                    n_skipped_no_cruise += 1
                else:
                    n_skipped_no_cols += 1
            except Exception:  # noqa: BLE001
                n_skipped_no_cols += 1
            continue
        rows.append(d)

    print(f"\nProcessed {len(rows)} flights ({n_skipped_no_cols} missing cols, {n_skipped_no_cruise} insufficient phases)")

    if len(rows) < 100:
        print("Too few flights — abort")
        return 1

    m_inv = np.array([r["m_inv"] for r in rows])
    m_qar = np.array([r["m_qar"] for r in rows])
    residual = np.array([r["residual"] for r in rows])
    if args.strategy == "A":
        chi_inv = np.array([r["chi_inv"] for r in rows])
    elif args.strategy == "B":
        chi_inv = np.array([r["chi_mean"] for r in rows])
    else:
        chi_inv = np.array([r["chi_cru"] for r in rows])

    # === Stats ===
    corr = float(np.corrcoef(m_inv, m_qar)[0, 1])
    bias = float(np.median(m_inv / m_qar))
    mae = float(np.mean(np.abs(m_inv - m_qar)))
    rmse = float(np.sqrt(np.mean((m_inv - m_qar) ** 2)))
    abs_rel = np.abs((m_inv - m_qar) / m_qar)
    median_abs_rel = float(np.median(abs_rel))
    p99_abs_rel = float(np.percentile(abs_rel, 99))

    print(f"\n=== Phase 1.8a Pre-validation gate (Strategy A) ===")
    print(f"n_flights : {len(rows)}")
    print(f"corr(m_inv, m_qar) Pearson : {corr:+.4f}")
    print(f"bias = median(m_inv / m_qar) : {bias:.4f}")
    print(f"MAE = {mae:.1f} kg")
    print(f"RMSE = {rmse:.1f} kg")
    print(f"median |Δm| / m_qar : {median_abs_rel*100:.2f} %")
    print(f"p99 |Δm| / m_qar : {p99_abs_rel*100:.2f} %")
    print(f"\nchi_inv : median={np.median(chi_inv):.3f}, p10={np.percentile(chi_inv, 10):.3f}, p90={np.percentile(chi_inv, 90):.3f}")
    print(f"residual : median={np.median(residual):.1f} N², p99={np.percentile(residual, 99):.1f} N²")
    print(f"\nm_qar : median={np.median(m_qar):.0f} kg, p10={np.percentile(m_qar, 10):.0f}, p90={np.percentile(m_qar, 90):.0f}")
    print(f"m_inv : median={np.median(m_inv):.0f} kg, p10={np.percentile(m_inv, 10):.0f}, p90={np.percentile(m_inv, 90):.0f}")

    # === Verdict ===
    if corr >= 0.60:
        verdict = "✅ Strategy A viable — proceed to Exp 2 (retrain v15)"
    elif corr >= 0.40:
        verdict = "⚠️ Strategy A marginal — escalate to Strategy B (multi-window)"
    else:
        verdict = "❌ Strategy A insufficient — escalate to Strategy B without retrain"
    print(f"\nVerdict : {verdict}")

    # === Write CSV ===
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" for k in keys) + "\n")
    print(f"\nCSV : {args.out_csv}")

    # === Markdown report ===
    lines = [
        "# Phase 1.8a — Pre-validation gate (Strategy A : single-window inversion)",
        "",
        f"> R5-safe : QAR utilisé UNIQUEMENT pour mesurer la qualité du proxy m_inv.",
        f"> Aucun signal QAR n'entre dans l'inversion (qui utilise seulement q, M, alt_m).",
        f"> n_flights = {len(rows)}.",
        "",
        "## Stats globales",
        "",
        f"- **corr Pearson(m_inv, m_qar) = {corr:+.4f}**  (target ≥ +0.60)",
        f"- bias median(m_inv / m_qar) = {bias:.4f}",
        f"- MAE = {mae:.1f} kg",
        f"- RMSE = {rmse:.1f} kg",
        f"- median |Δm|/m_qar = {median_abs_rel*100:.2f} %",
        f"- p99 |Δm|/m_qar = {p99_abs_rel*100:.2f} %",
        "",
        "## Distributions",
        "",
        "| Quantity | median | p10 | p90 |",
        "|---|---:|---:|---:|",
        f"| m_qar (kg) | {np.median(m_qar):.0f} | {np.percentile(m_qar, 10):.0f} | {np.percentile(m_qar, 90):.0f} |",
        f"| m_inv (kg) | {np.median(m_inv):.0f} | {np.percentile(m_inv, 10):.0f} | {np.percentile(m_inv, 90):.0f} |",
        f"| chi_inv | {np.median(chi_inv):.3f} | {np.percentile(chi_inv, 10):.3f} | {np.percentile(chi_inv, 90):.3f} |",
        "",
        f"## Verdict : **{verdict}**",
        "",
        "## Compliance",
        "",
        "- R1 (no QAR in training) : ✅ — script is read-only validation.",
        "- R2 (no BADA) : ✅ — pure P&S + scipy.optimize.",
        "- R3 (textbook η_o_DO in code) : ✅ — η not used in inversion (cruise: T=D).",
        "- R10 (identifiability) : measure χ_inv distribution + residual to flag degeneracy.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Markdown : {args.out}")

    return 0 if corr >= 0.60 else (2 if corr >= 0.40 else 1)


if __name__ == "__main__":
    sys.exit(main())
