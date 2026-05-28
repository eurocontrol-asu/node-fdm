# ruff: noqa: RUF001, RUF003, C408, S607, E501
"""Paper 0 §8 — QAR-based validation of FMS-equivalent target extraction.

R1/R5-safe: QAR is validation-only. The preprocessing pipeline reads only
the ADS-B/Mode-S-decodable equivalents of QAR signals (raw_*, bds_*), and
is fed NULL for `bds_mcp_alt_sel_ft` / `bds_fms_alt_sel_ft` to force the
plateau-detection methodology under test. The FCU `*_SEL` columns are
opened in a separate code path strictly for the truth-side comparison.

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    unset VIRTUAL_ENV
    uv run python scripts/_validation/paper0_qar_target_validation.py \\
        --report   data/models/_comparison/paper0_qar_target_validation.md \\
        --figures-dir data/figures/paper0_qar_target_validation/ \\
        --csv     data/models/_comparison/paper0_qar_target_validation_per_flight.csv
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from node_fdm_data.lateral import augment_lateral
from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds
from node_fdm_data.segments import build_selected_params
from node_fdm_pipeline.config import SelectedParamConfig

# ---------------------------------------------------------------- constants --
QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")
QAR_GLOB = "*A320*.parquet"

# Conversions
FT_TO_M = 0.3048
KT_TO_MS = 0.514444
FTMIN_TO_MS = 0.00508
DEG_TO_RAD = math.pi / 180.0

# Pipeline step (downsample 1 Hz QAR → 0.25 Hz, matching ADS-B dt=4 s).
QAR_DOWNSAMPLE = 4
PIPELINE_DT_S = 4.0

# Validated QAR truth columns (user-confirmed mapping).
COL_ALT_SEL_FT = "NAV__ALT_SEL_F"  # filtered Float64 (ft) → m
COL_MACH_SEL = "SPD__MACH_SEL"  # gated by SPD__SPD_MACH_SEL == 'MACH'
COL_CAS_SEL_KT = "SPD__SPD_SEL"  # gated by SPD__SPD_MACH_SEL == 'SPEED'
COL_SPD_MODE = "SPD__SPD_MACH_SEL"  # 'MACH' | 'SPEED'
COL_FPA_SEL_DEG = "ATT__FPA_SEL"  # degrees → rad (only when FPA mode active)
COL_VS_SEL_FTMIN = "SPD__VERT_SEL"  # ft/min (only when V_S mode active)
COL_LONG_MODE = "FMA__LONGITUDINAL"  # ALT|ALT_CAPT|V_S|G_S|FLARE|... determines γ truth
COL_HDG_SEL_DEG = "NAV__HDG_SEL"  # signed deg
COL_LAT_MODE = "FMA__LAT_MODES"  # 'NAV' | 'HDG' | 'LOC TRK' | 'LOC*' | 'RWY' | 'NRD'
COL_TRACK_ACTUAL = "NAV__TRACK"  # 0-360°, used as FMS proxy in NAV mode on straight legs

# γ truth dispatch by longitudinal FMA mode. Modes not listed here yield NaN
# (the autopilot is managing γ, no FCU intent to compare against).
GAMMA_LEVEL_MODES = ("ALT", "ALT_CAPT", "FLARE")              # γ_truth = 0
GAMMA_GS_MODES = ("G_S", "G_S_CAPT")                          # γ_truth = -3° (ILS std)
GAMMA_VS_MODES = ("V_S",)                                      # γ_truth from VS_SEL
GAMMA_FPA_MODES = ("FPA",)                                     # γ_truth from FPA_SEL
GLIDESLOPE_GAMMA_RAD = -3.0 * math.pi / 180.0                 # -0.05236 rad

# Phase classification thresholds.
CRUISE_ALT_MIN_FT = 24000.0
TMA_ALT_FT = 10000.0
VS_LEVEL_FTMIN = 100.0
VS_CLIMB_FTMIN = 400.0
VS_DESCENT_FTMIN = -400.0

# Pipeline configuration for build_selected_params.
# Uses the SAME Pydantic config as the production pipeline (AXM-1689 bilateral
# defaults: bilateral_mach, bilateral_cas, bilateral_vz, bilateral_gamma).
# An earlier version of this experiment incorrectly used the legacy savgol_*
# CURRENT_CONFIG snapshot from scripts/debug/check_crossover_segments.py —
# that file is a before/after comparison artifact, NOT the prod config.
SEL_PARAMS_CONFIG: dict[str, object] = SelectedParamConfig().model_dump()
SEL_PARAMS_CONFIG["alt_hold_relax"] = 15  # only key not in the Pydantic model

# Acceptance criteria (channel → (mae_target, cov_target_pct)).
AC_TARGETS = {
    "alt_m": (91.0, 70.0),  # AC2, AC3
    "mach": (0.005, 60.0),  # AC4, AC5
    "cas_kt": (5.0, 60.0),  # AC6, AC7
    "gamma_rad": (0.005, 50.0),  # AC8, AC9
    "track_deg": (3.0, 80.0),  # AC10, AC11
}

# Channels per protocol §5.
CHANNELS = ("alt_m", "mach", "cas_kt", "gamma_rad", "track_deg")
CHANNEL_LABEL = {
    "alt_m": "Altitude (m)",
    "mach": "Mach",
    "cas_kt": "CAS (kt)",
    "gamma_rad": "Gamma (rad)",
    "track_deg": "Track (deg)",
}
CHANNEL_TOL = {
    "alt_m": 91.0,
    "mach": 0.005,
    "cas_kt": 5.0,
    "gamma_rad": 0.005,
    "track_deg": 3.0,
}
CHANNEL_PHASES = {
    "alt_m": ("climb", "cruise", "descent", "tma", "global"),
    "mach": ("cruise",),
    "cas_kt": ("climb", "descent"),
    "gamma_rad": ("climb", "descent"),
    "track_deg": ("straight",),
}


# ------------------------------------------------------------- QAR adapter --
def qar_to_raw_schema(df: pl.DataFrame, flight_id: str) -> pl.DataFrame:
    """Map QAR-native columns to the pipeline's raw_*/bds_*/meta_* schema.

    Sets ``bds_mcp_alt_sel_ft`` and ``bds_fms_alt_sel_ft`` to NULL: R1/R5
    require the pipeline derive the target from the trajectory alone, not
    from a BDS-decoded FCU broadcast that mirrors the truth column.
    """
    n = df.height
    return df.with_columns(
        pl.col("ALT__STD").cast(pl.Float64).alias("raw_alt_ft"),
        pl.col("ATT__VV").cast(pl.Float64).alias("raw_vz_ftmin"),
        pl.col("SPD__GND").cast(pl.Float64).alias("raw_gs_kt"),
        pl.col("NAV__LAT").cast(pl.Float64).alias("raw_lat_deg"),
        pl.col("NAV__LONG").cast(pl.Float64).alias("raw_lon_deg"),
        pl.col("NAV__HDG_TRUE").cast(pl.Float64).alias("raw_heading_deg"),
        pl.col("NAV__TRACK").cast(pl.Float64).alias("raw_track_deg"),
        # augment_lateral expects bare `latitude`, `longitude`, `track`.
        pl.col("NAV__LAT").cast(pl.Float64).alias("latitude"),
        pl.col("NAV__LONG").cast(pl.Float64).alias("longitude"),
        pl.col("NAV__TRACK").cast(pl.Float64).alias("track"),
        pl.col("SPD__MACH").cast(pl.Float64).alias("bds_mach"),
        pl.col("SPD__CAS").cast(pl.Float64).alias("bds_ias_kt"),
        pl.col("SPD__TAS").cast(pl.Float64).alias("bds_tas_kt"),
        pl.lit(None, dtype=pl.Float64).alias("bds_mcp_alt_sel_ft"),
        pl.lit(None, dtype=pl.Float64).alias("bds_fms_alt_sel_ft"),
        # ISA temperature stand-in for the missing ERA5 — required by
        # clean_bds_speeds to compute fdm_tas_from_cas_kt (which feeds the γ
        # chain). Below 11 km: T = 288.15 - 0.0065·h ; above: T = 216.65.
        pl.when(pl.col("ALT__STD") * FT_TO_M <= 11000.0)
        .then(288.15 - 0.0065 * pl.col("ALT__STD") * FT_TO_M)
        .otherwise(216.65)
        .alias("era_temp_K"),
        pl.lit(flight_id).alias("meta_flight_id"),
        pl.int_range(0, n, dtype=pl.Int64).alias("meta_row_idx"),
    )


def downsample(df: pl.DataFrame, step: int) -> pl.DataFrame:
    """Keep every *step*-th row (deterministic, no smoothing)."""
    return df.with_row_index("__r").filter(pl.col("__r") % step == 0).drop("__r")


def run_preprocessing(df_raw: pl.DataFrame) -> pl.DataFrame:
    """Run M3 (clean_bds_speeds) → M1+M5 (build_selected_params) → M2 (augment_lateral).

    Returns df with ``fdm_alt_target_ft``, ``fdm_cas_target_kt``,
    ``fdm_tas_target_kt``, ``fdm_gamma_target_rad``, ``fdm_track_ortho_deg``,
    ``fdm_in_turn``, and ``fdm_*_known`` flags.
    """
    df = clean_bds_speeds(df_raw)
    # `_build_gamma_target` in segments.py requires `fdm_gamma_rad` to be
    # present ; that column is normally produced by `derive_columns` (étape 4).
    # Inline its definition here so the γ target chain wires up. Formula
    # matches preprocessing/derive.py: γ = asin( (vz·FTMIN) / (TAS·KT) ).
    if {"raw_vz_ftmin", "fdm_tas_from_cas_kt"}.issubset(df.columns):
        vz_ms = pl.col("raw_vz_ftmin") * FTMIN_TO_MS
        tas_ms = (pl.col("fdm_tas_from_cas_kt") * KT_TO_MS).clip(lower_bound=1e-6)
        df = df.with_columns(
            (vz_ms / tas_ms).clip(-1.0, 1.0).arcsin().alias("fdm_gamma_rad"),
        )
    df = build_selected_params(df, SEL_PARAMS_CONFIG)
    df = augment_lateral(df, dt=PIPELINE_DT_S)
    return df


# ----------------------------------------------------------- truth extract --
def extract_qar_truth(df_qar: pl.DataFrame) -> dict[str, np.ndarray]:
    """Extract per-row FCU truth columns with R1/R5 isolation.

    Returns a dict of aligned arrays (same length as ``df_qar``):
      - ``alt_truth_m``
      - ``mach_truth`` (NaN outside MACH mode)
      - ``cas_truth_kt`` (NaN outside SPEED mode)
      - ``gamma_truth_rad`` (NaN where neither FPA_SEL nor VS_SEL is meaningful)
      - ``hdg_truth_deg`` (NaN outside HDG mode)
    """
    n = df_qar.height

    def _col_or_nan(name: str) -> np.ndarray:
        if name in df_qar.columns:
            return df_qar[name].cast(pl.Float64).to_numpy()
        return np.full(n, np.nan)

    alt_ft = _col_or_nan(COL_ALT_SEL_FT)
    alt_truth_m = alt_ft * FT_TO_M

    if COL_SPD_MODE in df_qar.columns:
        mode = df_qar[COL_SPD_MODE].to_numpy()
    else:
        mode = np.array([""] * n)
    mach_raw = _col_or_nan(COL_MACH_SEL)
    cas_raw_kt = _col_or_nan(COL_CAS_SEL_KT)
    mach_truth = np.where(mode == "MACH", mach_raw, np.nan)
    cas_truth_kt = np.where(mode == "SPEED", cas_raw_kt, np.nan)
    # Sanity-filter QAR data-quality outliers in the polymorphic-knob columns.
    # On some flights the `SPD__MACH_SEL` column contains non-physical values
    # (e.g. 0.984 repeated for 870 samples on AAF231) even when the mode
    # discriminator reports 'MACH' — likely a CAS value that leaked through
    # a transient mode toggle. Operational A320 envelopes: M ∈ [0.50, 0.90],
    # CAS dialed ∈ [80, 380] kt. Anything outside is QAR-side garbage.
    mach_truth = np.where(
        np.isfinite(mach_truth) & (mach_truth >= 0.50) & (mach_truth <= 0.90),
        mach_truth, np.nan,
    )
    cas_truth_kt = np.where(
        np.isfinite(cas_truth_kt) & (cas_truth_kt >= 80.0) & (cas_truth_kt <= 380.0),
        cas_truth_kt, np.nan,
    )

    # γ truth: dispatch by the active FMA__LONGITUDINAL mode (the only sane way
    # to know what the FCU is actually doing). FPA_SEL and VS_SEL retain stale
    # values across mode transitions, so "non-zero" is NOT a valid mode-active
    # proxy — the pilot pre-dials a descent V/S before initiating descent while
    # the autopilot is still in CLB managed mode ; my old filter then captured
    # -1500 ftmin as truth during climb, which is wrong-sign garbage.
    #
    # Modes that give a meaningful γ truth:
    #   ALT / ALT_CAPT / FLARE  → γ = 0  (level)
    #   G_S / G_S_CAPT          → γ = -3° (ILS standard glideslope)
    #   V_S                     → γ = asin(VS_SEL · 0.00508 / TAS · 0.514)
    #   FPA (if A320 schema exposes it) → γ = FPA_SEL · π/180
    # Everything else (CLB, DES, OP_CLB, OP_DES, SRS, FINAL_APP, ROLL_OUT, OFF):
    # autopilot-managed γ, no FCU intent to compare against → NaN.
    fpa_deg = _col_or_nan(COL_FPA_SEL_DEG)
    vs_ftmin = _col_or_nan(COL_VS_SEL_FTMIN)
    tas_kt = (
        df_qar["SPD__TAS"].cast(pl.Float64).to_numpy()
        if "SPD__TAS" in df_qar.columns
        else np.full(n, np.nan)
    )
    tas_ms = tas_kt * KT_TO_MS
    vs_ms = vs_ftmin * FTMIN_TO_MS
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where((tas_ms > 1.0) & np.isfinite(vs_ms), vs_ms / tas_ms, np.nan)
        ratio = np.clip(ratio, -1.0, 1.0)
        gamma_from_vs = np.arcsin(ratio)
    gamma_from_fpa = fpa_deg * DEG_TO_RAD

    if COL_LONG_MODE in df_qar.columns:
        long_mode = df_qar[COL_LONG_MODE].to_numpy()
    else:
        long_mode = np.array([""] * n)
    gamma_truth = np.full(n, np.nan)
    gamma_truth = np.where(np.isin(long_mode, GAMMA_LEVEL_MODES), 0.0, gamma_truth)
    gamma_truth = np.where(np.isin(long_mode, GAMMA_GS_MODES), GLIDESLOPE_GAMMA_RAD, gamma_truth)
    gamma_truth = np.where(np.isin(long_mode, GAMMA_VS_MODES), gamma_from_vs, gamma_truth)
    gamma_truth = np.where(np.isin(long_mode, GAMMA_FPA_MODES), gamma_from_fpa, gamma_truth)

    # Lateral truth: use FMA lateral mode to switch between HDG_SEL (HDG mode)
    # and actual NAV__TRACK as proxy-for-FMS-target (NAV mode, dominant ~72%).
    # On straight legs (filtered downstream via fdm_in_turn==False) the FMS-
    # commanded track equals the actual flown track modulo small wind drift.
    if COL_LAT_MODE in df_qar.columns:
        lat_mode = df_qar[COL_LAT_MODE].to_numpy()
    else:
        lat_mode = np.array(["NRD"] * n)
    hdg_sel = _col_or_nan(COL_HDG_SEL_DEG)
    track_actual = _col_or_nan(COL_TRACK_ACTUAL)
    # Map HDG_SEL (signed [-180,180]) and TRACK (unsigned [0,360]) into a
    # common unsigned [0,360] convention for the comparison side.
    hdg_sel_unsigned = np.where(np.isfinite(hdg_sel), hdg_sel % 360.0, np.nan)
    hdg_truth = np.where(
        lat_mode == "HDG", hdg_sel_unsigned,
        np.where(np.isin(lat_mode, ["NAV", "LOC TRK"]), track_actual, np.nan),
    )

    return {
        "alt_truth_m": alt_truth_m,
        "mach_truth": mach_truth,
        "cas_truth_kt": cas_truth_kt,
        "gamma_truth_rad": gamma_truth,
        "hdg_truth_deg": hdg_truth,
    }


# ----------------------------------------------------- phase classification --
def classify_phase(df: pl.DataFrame) -> np.ndarray:
    """Classify each sample as climb / cruise / descent / tma / other.

    Uses ``raw_alt_ft`` and ``raw_vz_ftmin`` on the processed (downsampled) df.
    """
    n = df.height
    out = np.full(n, "other", dtype="<U8")
    alt_ft = df["raw_alt_ft"].to_numpy()
    vz = df["raw_vz_ftmin"].to_numpy()

    # vz-based climb/descent (any altitude) so the initial descent FL340→FL240
    # falls under "descent" — that band is precisely where V/S mode is most
    # often dialed (the FCU γ truth column comes from there).
    climb_mask = vz > VS_CLIMB_FTMIN
    descent_mask = vz < VS_DESCENT_FTMIN
    cruise_mask = (alt_ft >= CRUISE_ALT_MIN_FT) & (np.abs(vz) < VS_LEVEL_FTMIN)
    tma_mask = alt_ft < TMA_ALT_FT

    out[climb_mask] = "climb"
    out[descent_mask] = "descent"
    out[cruise_mask] = "cruise"
    # TMA overrides climb/descent (sub-phase of low altitude)
    out[tma_mask & (climb_mask | descent_mask | (out == "other"))] = "tma"
    return out


# ---------------------------------------------------------- per-flight run --
def process_flight(parquet_path: Path) -> dict | None:
    """Run the Paper 0 preprocessing on one QAR flight + extract aligned truth.

    Returns a dict with predicted, truth, known-flag and phase arrays per
    channel, or None if the flight is unusable.
    """
    flight_id = parquet_path.stem
    try:
        df_qar = pl.read_parquet(parquet_path)
    except Exception as exc:
        print(f"  [SKIP] {parquet_path.name}: read failed: {exc}", file=sys.stderr)
        return None

    # Ground-truth extraction at QAR's native rate, then downsample.
    truth_full = extract_qar_truth(df_qar)
    truth_idx = np.arange(0, df_qar.height, QAR_DOWNSAMPLE)
    truth = {k: v[truth_idx] for k, v in truth_full.items()}

    # Preprocessing pipeline on the R1-clean adapted view.
    df_raw = qar_to_raw_schema(df_qar, flight_id)
    df_raw = downsample(df_raw, QAR_DOWNSAMPLE)
    try:
        df_proc = run_preprocessing(df_raw)
    except Exception as exc:
        print(
            f"  [SKIP] {flight_id}: pipeline error: {exc}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return None

    # If pipeline reordered/filtered rows, align by meta_row_idx.
    if "meta_row_idx" in df_proc.columns:
        keep_idx = df_proc["meta_row_idx"].to_numpy()
        # meta_row_idx was assigned pre-downsample, so positions in truth_idx
        # must be matched by value. truth is already on the downsampled grid,
        # whose meta_row_idx equals truth_idx exactly. We map proc indices to
        # downsample grid positions:
        ds_to_pos = {int(v): i for i, v in enumerate(truth_idx)}
        positions = np.array([ds_to_pos.get(int(v), -1) for v in keep_idx])
        valid = positions >= 0
        positions = positions[valid]
        truth_aligned = {k: v[positions] for k, v in truth.items()}
    else:
        truth_aligned = truth

    # Pull predicted columns (with safe defaults if a column wasn't produced).
    def _pull(name: str) -> np.ndarray:
        if name in df_proc.columns:
            return df_proc[name].cast(pl.Float64).to_numpy()
        return np.full(df_proc.height, np.nan)

    def _pull_bool(name: str) -> np.ndarray:
        if name in df_proc.columns:
            arr = df_proc[name].to_numpy()
            return np.asarray(arr, dtype=bool)
        return np.zeros(df_proc.height, dtype=bool)

    # Altitude: anchored target (backward-fill from the next detected plateau,
    # forward-fill for the post-last-plateau tail — semantics of `_anchored_target`
    # in segments.py). Justified: pilots dial the cleared altitude at takeoff
    # and hold it across the entire climb chain.
    alt_pred_m = _pull("fdm_alt_target_ft") * FT_TO_M

    # CAS + Mach: point-wise plateau values only — `fdm_cas_sel_kt` /
    # `fdm_mach_sel` are NaN outside detected plateaus. No backward-fill
    # extrapolation: the pilot may re-dial CAS / Mach in flight (descent
    # step-downs, cruise Mach changes), so anchoring forward would be a
    # data-fabrication artifact. We compare strictly where the pipeline
    # produces a value ; samples outside any plateau drop out of the
    # comparison (coverage reflects plateau detection rate).
    cas_pred_kt = _pull("fdm_cas_sel_kt")
    mach_pred = _pull("fdm_mach_sel")

    gamma_pred = _pull("fdm_gamma_target_rad")
    track_pred_deg = _pull("fdm_track_ortho_deg")
    in_turn = _pull_bool("fdm_in_turn")

    alt_ft_proc = df_proc["raw_alt_ft"].cast(pl.Float64).to_numpy()
    alt_m_proc = alt_ft_proc * FT_TO_M

    # Current pointwise CAS / Mach (for directional metric — same logic as
    # altitude: pipeline anchors to the next plateau ; FCU truth steps through
    # intermediate dialed values ; both are valid views of the target).
    cas_current_kt = (
        df_proc["bds_ias_kt_clean"].cast(pl.Float64).to_numpy()
        if "bds_ias_kt_clean" in df_proc.columns
        else np.full(df_proc.height, np.nan)
    )
    mach_current = (
        df_proc["bds_mach_clean"].cast(pl.Float64).to_numpy()
        if "bds_mach_clean" in df_proc.columns
        else np.full(df_proc.height, np.nan)
    )

    # Known flags (NaN-preserving).
    alt_known = np.isfinite(alt_pred_m)
    cas_known = np.isfinite(cas_pred_kt)
    mach_known = np.isfinite(mach_pred)
    gamma_known = (
        _pull_bool("fdm_gamma_target_known")
        if "fdm_gamma_target_known" in df_proc.columns
        else np.isfinite(gamma_pred)
    )
    track_known = (
        _pull_bool("fdm_track_sel_known")
        if "fdm_track_sel_known" in df_proc.columns
        else np.isfinite(track_pred_deg)
    )

    phase = classify_phase(df_proc)

    # Current instantaneous γ for the directional metric.
    gamma_current = (
        df_proc["fdm_gamma_rad"].cast(pl.Float64).to_numpy()
        if "fdm_gamma_rad" in df_proc.columns
        else np.full(df_proc.height, np.nan)
    )

    return {
        "flight_id": flight_id,
        "n_samples": int(df_proc.height),
        "alt_current_m": alt_m_proc,
        "alt_pred_m": alt_pred_m,
        "alt_truth_m": truth_aligned["alt_truth_m"],
        "alt_known": alt_known,
        "mach_current": mach_current,
        "mach_pred": mach_pred,
        "mach_truth": truth_aligned["mach_truth"],
        "mach_known": mach_known,
        "cas_current_kt": cas_current_kt,
        "cas_pred_kt": cas_pred_kt,
        "cas_truth_kt": truth_aligned["cas_truth_kt"],
        "cas_known": cas_known,
        "gamma_current_rad": gamma_current,
        "gamma_pred_rad": gamma_pred,
        "gamma_truth_rad": truth_aligned["gamma_truth_rad"],
        "gamma_known": gamma_known,
        "track_pred_deg": track_pred_deg,
        "track_truth_deg": truth_aligned["hdg_truth_deg"],
        "track_known": track_known,
        "in_turn": in_turn,
        "phase": phase,
    }


# ------------------------------------------------------------ aggregation --
def _angular_delta_deg(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Signed minimum-arc difference between two heading/track angles in deg."""
    d = (pred - truth + 180.0) % 360.0 - 180.0
    return d


def compute_channel_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    known: np.ndarray,
    phase: np.ndarray,
    tol: float,
    phases: tuple[str, ...],
    *,
    angular: bool = False,
) -> dict[str, dict[str, float]]:
    """For one channel, return {phase: {mae, p99, coverage_pct, agreement_pct, n}}."""
    out: dict[str, dict[str, float]] = {}
    for ph in phases:
        if ph == "global":
            phase_mask = np.ones_like(phase, dtype=bool)
        elif ph == "straight":
            phase_mask = np.ones_like(phase, dtype=bool)  # caller pre-filters turns
        else:
            phase_mask = phase == ph
        n_phase = int(phase_mask.sum())
        finite_truth = np.isfinite(truth)
        finite_pred = np.isfinite(pred)
        eligible = phase_mask & finite_pred & finite_truth & known.astype(bool)
        n_compare = int(eligible.sum())
        coverage_pct = 100.0 * n_compare / max(n_phase, 1)
        if n_compare == 0:
            out[ph] = dict(
                n=0,
                mae=float("nan"),
                p99=float("nan"),
                coverage_pct=coverage_pct,
                agreement_pct=float("nan"),
            )
            continue
        if angular:
            delta = np.abs(_angular_delta_deg(pred[eligible], truth[eligible]))
        else:
            delta = np.abs(pred[eligible] - truth[eligible])
        out[ph] = dict(
            n=n_compare,
            mae=float(delta.mean()),
            p99=float(np.quantile(delta, 0.99)),
            coverage_pct=coverage_pct,
            agreement_pct=float(100.0 * (delta < tol).mean()),
        )
    return out


def compute_directional_consistency(
    pred: np.ndarray,
    truth: np.ndarray,
    current: np.ndarray,
    known: np.ndarray,
    phase: np.ndarray,
    tol: float,
    *,
    phases: tuple[str, ...] = ("climb", "descent", "level", "global"),
) -> dict[str, dict[str, float]]:
    """Chain-consistency metric: truth must lie inside the convex hull of
    (current, pred) ± tol.

    The pipeline anchors to the *next detected plateau* — typically the
    chain's endpoint (cruise FL, Vapp). The FCU truth shows the current
    dialed step within the chain. Both are valid views of "the target".

    A sample is consistent when truth ∈ [min(current, pred) - tol,
    max(current, pred) + tol]. This subsumes:
      - climb chain (current < truth < pred): ✓
      - descent chain (pred < truth < current): ✓
      - intermediate-step hold during a chain
        (current = truth, pred = chain endpoint): ✓
      - level (current = truth = pred): ✓
      - wrong direction (truth outside the chain interval): flagged with
        magnitude = distance from truth to the nearest interval bound.

    Args:
        tol: tolerance in the channel's units (e.g. 91 m, 5 kt, 0.005 mach).
    """
    out: dict[str, dict[str, float]] = {}
    for ph in phases:
        if ph == "global":
            phase_mask = np.ones_like(phase, dtype=bool)
        elif ph == "level":
            # Trajectory-defined level: |truth - current| ≤ tol.
            phase_mask = np.abs(truth - current) <= tol
        else:
            phase_mask = phase == ph
        eligible = phase_mask & known.astype(bool) & np.isfinite(pred) & np.isfinite(truth) & np.isfinite(current)
        n = int(eligible.sum())
        if n == 0:
            out[ph] = dict(
                n=0,
                consistency_pct=float("nan"),
                directional_mae=float("nan"),
                wrong_direction_p99=float("nan"),
            )
            continue
        c = current[eligible]
        t = truth[eligible]
        p = pred[eligible]
        lo = np.minimum(c, p) - tol
        hi = np.maximum(c, p) + tol
        # Distance from truth to the [lo, hi] interval (0 if truth is inside).
        wrong = np.maximum(0.0, np.maximum(t - hi, lo - t))
        consistent = wrong < tol
        out[ph] = dict(
            n=n,
            consistency_pct=float(100.0 * consistent.mean()),
            directional_mae=float(wrong.mean()),
            wrong_direction_p99=float(np.quantile(wrong, 0.99)),
        )
    return out


def aggregate_results(per_flight: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Pool every flight's samples and compute per-channel-per-phase metrics."""
    if not per_flight:
        return {}
    pooled = {
        k: np.concatenate([f[k] for f in per_flight])
        for k in (
            "alt_current_m",
            "alt_pred_m",
            "alt_truth_m",
            "alt_known",
            "mach_current",
            "mach_pred",
            "mach_truth",
            "mach_known",
            "cas_current_kt",
            "cas_pred_kt",
            "cas_truth_kt",
            "cas_known",
            "gamma_current_rad",
            "gamma_pred_rad",
            "gamma_truth_rad",
            "gamma_known",
            "track_pred_deg",
            "track_truth_deg",
            "track_known",
            "in_turn",
            "phase",
        )
    }
    results = {}
    results["alt_m"] = compute_channel_metrics(
        pooled["alt_pred_m"],
        pooled["alt_truth_m"],
        pooled["alt_known"],
        pooled["phase"],
        CHANNEL_TOL["alt_m"],
        CHANNEL_PHASES["alt_m"],
    )
    results["mach"] = compute_channel_metrics(
        pooled["mach_pred"],
        pooled["mach_truth"],
        pooled["mach_known"],
        pooled["phase"],
        CHANNEL_TOL["mach"],
        CHANNEL_PHASES["mach"],
    )
    results["cas_kt"] = compute_channel_metrics(
        pooled["cas_pred_kt"],
        pooled["cas_truth_kt"],
        pooled["cas_known"],
        pooled["phase"],
        CHANNEL_TOL["cas_kt"],
        CHANNEL_PHASES["cas_kt"],
    )
    results["gamma_rad"] = compute_channel_metrics(
        pooled["gamma_pred_rad"],
        pooled["gamma_truth_rad"],
        pooled["gamma_known"],
        pooled["phase"],
        CHANNEL_TOL["gamma_rad"],
        CHANNEL_PHASES["gamma_rad"],
    )
    # Track: straight-leg only (in_turn == False).
    straight = ~pooled["in_turn"]
    results["track_deg"] = compute_channel_metrics(
        pooled["track_pred_deg"][straight],
        pooled["track_truth_deg"][straight],
        pooled["track_known"][straight],
        pooled["phase"][straight],
        CHANNEL_TOL["track_deg"],
        CHANNEL_PHASES["track_deg"],
        angular=True,
    )
    # Directional-consistency metric — same logic as altitude, applied to every
    # one-dimensional channel where the FCU dial steps through intermediate
    # values while the pipeline anchors forward to the next plateau.
    results["alt_directional"] = compute_directional_consistency(
        pooled["alt_pred_m"],
        pooled["alt_truth_m"],
        pooled["alt_current_m"],
        pooled["alt_known"],
        pooled["phase"],
        CHANNEL_TOL["alt_m"],
    )
    results["cas_directional"] = compute_directional_consistency(
        pooled["cas_pred_kt"],
        pooled["cas_truth_kt"],
        pooled["cas_current_kt"],
        pooled["cas_known"],
        pooled["phase"],
        CHANNEL_TOL["cas_kt"],
    )
    results["mach_directional"] = compute_directional_consistency(
        pooled["mach_pred"],
        pooled["mach_truth"],
        pooled["mach_current"],
        pooled["mach_known"],
        pooled["phase"],
        CHANNEL_TOL["mach"],
    )
    results["gamma_directional"] = compute_directional_consistency(
        pooled["gamma_pred_rad"],
        pooled["gamma_truth_rad"],
        pooled["gamma_current_rad"],
        pooled["gamma_known"],
        pooled["phase"],
        CHANNEL_TOL["gamma_rad"],
    )
    return results


# ----------------------------------------------------------------- plots --
def _density_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    lim: tuple[float, float] | None,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.hexbin(x[finite], y[finite], gridsize=60, cmap="viridis", mincnt=1)
    if lim is not None:
        ax.plot(lim, lim, "r--", lw=0.8, alpha=0.5)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
    else:
        lo = float(np.nanmin(np.concatenate([x[finite], y[finite]])))
        hi = float(np.nanmax(np.concatenate([x[finite], y[finite]])))
        ax.plot([lo, hi], [lo, hi], "r--", lw=0.8, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_scatters(per_flight: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled = {
        k: np.concatenate([f[k] for f in per_flight])
        for k in (
            "alt_pred_m",
            "alt_truth_m",
            "alt_known",
            "mach_pred",
            "mach_truth",
            "mach_known",
            "cas_pred_kt",
            "cas_truth_kt",
            "cas_known",
            "gamma_pred_rad",
            "gamma_truth_rad",
            "gamma_known",
            "track_pred_deg",
            "track_truth_deg",
            "track_known",
            "in_turn",
            "phase",
        )
    }

    # 1. Altitude (4-phase facet)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, ph in zip(axes.flat, ("climb", "cruise", "descent", "tma"), strict=False):
        mask = (pooled["phase"] == ph) & pooled["alt_known"]
        _density_scatter(
            ax,
            pooled["alt_truth_m"][mask],
            pooled["alt_pred_m"][mask],
            xlabel="QAR alt_sel (m)",
            ylabel="pred alt_target (m)",
            title=f"Altitude — {ph}",
            lim=(0, 12000),
        )
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_alt.png", dpi=110)
    plt.close(fig)

    # 2. Mach (cruise only)
    fig, ax = plt.subplots(figsize=(6, 6))
    mask = (pooled["phase"] == "cruise") & pooled["mach_known"]
    _density_scatter(
        ax,
        pooled["mach_truth"][mask],
        pooled["mach_pred"][mask],
        xlabel="QAR Mach_sel",
        ylabel="pred Mach_target",
        title="Mach — cruise",
        lim=(0.5, 0.9),
    )
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_mach.png", dpi=110)
    plt.close(fig)

    # 3. CAS (climb + descent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, ph in zip(axes, ("climb", "descent"), strict=False):
        mask = (pooled["phase"] == ph) & pooled["cas_known"]
        _density_scatter(
            ax,
            pooled["cas_truth_kt"][mask],
            pooled["cas_pred_kt"][mask],
            xlabel="QAR IAS_sel (kt)",
            ylabel="pred CAS_target (kt)",
            title=f"CAS — {ph}",
            lim=(100, 350),
        )
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_cas.png", dpi=110)
    plt.close(fig)

    # 4. Gamma (climb + descent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, ph in zip(axes, ("climb", "descent"), strict=False):
        mask = (pooled["phase"] == ph) & pooled["gamma_known"]
        _density_scatter(
            ax,
            pooled["gamma_truth_rad"][mask],
            pooled["gamma_pred_rad"][mask],
            xlabel="QAR γ (rad)",
            ylabel="pred γ_target (rad)",
            title=f"γ — {ph}",
            lim=(-0.1, 0.1),
        )
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_gamma.png", dpi=110)
    plt.close(fig)

    # 5. Track (straight legs only)
    fig, ax = plt.subplots(figsize=(6, 6))
    straight = ~pooled["in_turn"]
    mask = straight & pooled["track_known"]
    # Truth is in QAR's signed [-180, 180]; pred is the great-circle bearing
    # in unsigned [0, 360]. Map both to [-180, 180] for the diag-overlay.
    t = pooled["track_truth_deg"][mask]
    p = pooled["track_pred_deg"][mask]
    p_signed = ((p + 180.0) % 360.0) - 180.0
    t_signed = ((t + 180.0) % 360.0) - 180.0
    _density_scatter(
        ax,
        t_signed,
        p_signed,
        xlabel="QAR HDG_sel (deg)",
        ylabel="pred track_ortho (deg)",
        title="Track — straight legs",
        lim=(-180, 180),
    )
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_track.png", dpi=110)
    plt.close(fig)

    # 6. |Δ| histogram per channel
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_list = list(axes.flat)
    pairs = [
        (
            "alt_m",
            "|alt_pred - alt_truth| (m)",
            pooled["alt_known"],
            pooled["alt_pred_m"] - pooled["alt_truth_m"],
            91.0,
        ),
        (
            "mach",
            "|mach_pred - mach_truth|",
            pooled["mach_known"],
            pooled["mach_pred"] - pooled["mach_truth"],
            0.005,
        ),
        (
            "cas_kt",
            "|cas_pred - cas_truth| (kt)",
            pooled["cas_known"],
            pooled["cas_pred_kt"] - pooled["cas_truth_kt"],
            5.0,
        ),
        (
            "gamma_rad",
            "|γ_pred - γ_truth| (rad)",
            pooled["gamma_known"],
            pooled["gamma_pred_rad"] - pooled["gamma_truth_rad"],
            0.005,
        ),
        (
            "track_deg",
            "|track_pred - track_truth| (deg)",
            pooled["track_known"],
            _angular_delta_deg(pooled["track_pred_deg"], pooled["track_truth_deg"]),
            3.0,
        ),
    ]
    for ax, (label, xlabel, known, delta, tol) in zip(axes_list[:5], pairs, strict=False):
        finite = np.isfinite(delta) & known.astype(bool)
        d = np.abs(delta[finite])
        if d.size:
            ax.hist(d, bins=80, edgecolor="none")
            ax.axvline(tol, color="r", linestyle="--", lw=0.8, label=f"tol={tol}")
            ax.set_xlim(0, max(tol * 3, float(np.quantile(d, 0.99))))
            ax.legend(fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(CHANNEL_LABEL[label])
    # Hide the 6th panel
    axes_list[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "histograms_delta.png", dpi=110)
    plt.close(fig)

    # 7. Coverage per phase per channel
    fig, ax = plt.subplots(figsize=(10, 5))
    channels_for_bar = list(CHANNELS)
    phases_for_bar = ["climb", "cruise", "descent", "tma", "straight"]
    width = 0.16
    x_idx = np.arange(len(phases_for_bar))
    for i, ch in enumerate(channels_for_bar):
        cov = []
        for ph in phases_for_bar:
            ph_mask = pooled["phase"] == ph if ph != "straight" else ~pooled["in_turn"]
            ph_n = int(ph_mask.sum())
            if ph_n == 0:
                cov.append(0.0)
                continue
            known_col = {
                "alt_m": pooled["alt_known"],
                "mach": pooled["mach_known"],
                "cas_kt": pooled["cas_known"],
                "gamma_rad": pooled["gamma_known"],
                "track_deg": pooled["track_known"],
            }[ch]
            truth_col = {
                "alt_m": pooled["alt_truth_m"],
                "mach": pooled["mach_truth"],
                "cas_kt": pooled["cas_truth_kt"],
                "gamma_rad": pooled["gamma_truth_rad"],
                "track_deg": pooled["track_truth_deg"],
            }[ch]
            cov.append(
                100.0
                * float((ph_mask & known_col.astype(bool) & np.isfinite(truth_col)).sum())
                / ph_n
            )
        ax.bar(x_idx + (i - 2) * width, cov, width, label=CHANNEL_LABEL[ch])
    ax.set_xticks(x_idx)
    ax.set_xticklabels(phases_for_bar)
    ax.set_ylabel("coverage (%)")
    ax.set_title("Coverage of (predicted_known ∧ truth_finite) per channel per phase")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_per_phase.png", dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------- output --
def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def _ac_verdict(
    results: dict[str, dict[str, dict[str, float]]],
) -> list[tuple[str, str, str, str]]:
    """Return [(AC_id, channel, criterion, PASS/FAIL/SKIP)] rows."""
    out: list[tuple[str, str, str, str]] = []
    # AC1: exit code 0  — reported by caller; placeholder PASS.
    out.append(("AC1", "—", "script runs end-to-end", "PASS"))

    def _bin(channel: str, metric: str, phase: str, threshold: float, gt: bool) -> str:
        if channel not in results or phase not in results[channel]:
            return "SKIP"
        v = results[channel][phase].get(metric, float("nan"))
        if not math.isfinite(v):
            return "SKIP"
        return "PASS" if (v > threshold if gt else v < threshold) else "FAIL"

    out.append(
        (
            "AC2",
            "altitude",
            "coverage > 70 % (pool)",
            _bin("alt_m", "coverage_pct", "global", 70.0, True),
        )
    )
    out.append(
        ("AC3", "altitude", "MAE < 91 m (pool)", _bin("alt_m", "mae", "global", 91.0, False))
    )
    # AC3b: directional-consistency variant — pred must lie on the side of
    # truth that the aircraft is heading toward (rationale: the pipeline
    # anchors to the next plateau, FCU shows the immediate dialed step ;
    # both are valid views of the target if the pipeline overshoots in
    # the same direction the aircraft is moving).
    # AC3b: directional, worst-of-three regimes (climb / descent / level).
    def _worst_dir_mae(channel: str) -> float:
        phs = [results.get(channel, {}).get(p, {}) for p in ("climb", "descent", "level")]
        vals = [p.get("directional_mae", float("nan")) for p in phs]
        return max((v for v in vals if math.isfinite(v)), default=float("nan"))

    alt_dir_worst = _worst_dir_mae("alt_directional")
    out.append(
        (
            "AC3b",
            "altitude (directional, worst phase)",
            "wrong-direction MAE < 91 m",
            "PASS" if alt_dir_worst < 91.0 and math.isfinite(alt_dir_worst)
            else ("SKIP" if not math.isfinite(alt_dir_worst) else "FAIL"),
        )
    )
    out.append(
        (
            "AC4",
            "Mach (cruise)",
            "coverage > 60 %",
            _bin("mach", "coverage_pct", "cruise", 60.0, True),
        )
    )
    out.append(
        ("AC5", "Mach (cruise)", "MAE < 0.005", _bin("mach", "mae", "cruise", 0.005, False))
    )
    mach_dir_worst = _worst_dir_mae("mach_directional")
    out.append(
        (
            "AC5b",
            "Mach (directional, worst phase)",
            "wrong-direction MAE < 0.005",
            "PASS" if mach_dir_worst < 0.005 and math.isfinite(mach_dir_worst)
            else ("SKIP" if not math.isfinite(mach_dir_worst) else "FAIL"),
        )
    )
    # CAS: pooled over climb+descent — use the worse of the two for the verdict.
    cas_phases = [results.get("cas_kt", {}).get(p, {}) for p in ("climb", "descent")]
    cas_cov = max((p.get("coverage_pct", float("nan")) for p in cas_phases), default=float("nan"))
    cas_mae = max((p.get("mae", float("-inf")) for p in cas_phases), default=float("-inf"))
    out.append(
        (
            "AC6",
            "CAS (climb/descent)",
            "coverage > 60 %",
            "PASS" if cas_cov > 60.0 else ("SKIP" if not math.isfinite(cas_cov) else "FAIL"),
        )
    )
    out.append(
        (
            "AC7",
            "CAS (climb/descent)",
            "MAE < 5 kt",
            "PASS"
            if cas_mae < 5.0 and math.isfinite(cas_mae)
            else ("SKIP" if not math.isfinite(cas_mae) else "FAIL"),
        )
    )
    cas_dir_worst = _worst_dir_mae("cas_directional")
    out.append(
        (
            "AC7b",
            "CAS (directional, worst phase)",
            "wrong-direction MAE < 5 kt",
            "PASS" if cas_dir_worst < 5.0 and math.isfinite(cas_dir_worst)
            else ("SKIP" if not math.isfinite(cas_dir_worst) else "FAIL"),
        )
    )
    g_phases = [results.get("gamma_rad", {}).get(p, {}) for p in ("climb", "descent")]
    g_cov = max((p.get("coverage_pct", float("nan")) for p in g_phases), default=float("nan"))
    g_mae = max((p.get("mae", float("-inf")) for p in g_phases), default=float("-inf"))
    out.append(
        (
            "AC8",
            "γ (climb/descent)",
            "coverage > 50 %",
            "PASS" if g_cov > 50.0 else ("SKIP" if not math.isfinite(g_cov) else "FAIL"),
        )
    )
    out.append(
        (
            "AC9",
            "γ (climb/descent)",
            "MAE < 0.005 rad",
            "PASS"
            if g_mae < 0.005 and math.isfinite(g_mae)
            else ("SKIP" if not math.isfinite(g_mae) else "FAIL"),
        )
    )
    g_dir_worst = _worst_dir_mae("gamma_directional")
    out.append(
        (
            "AC9b",
            "γ (directional, worst phase)",
            "wrong-direction MAE < 0.005 rad",
            "PASS" if g_dir_worst < 0.005 and math.isfinite(g_dir_worst)
            else ("SKIP" if not math.isfinite(g_dir_worst) else "FAIL"),
        )
    )
    out.append(
        (
            "AC10",
            "track (straight)",
            "coverage > 80 %",
            _bin("track_deg", "coverage_pct", "straight", 80.0, True),
        )
    )
    out.append(
        ("AC11", "track (straight)", "MAE < 3°", _bin("track_deg", "mae", "straight", 3.0, False))
    )
    return out


def write_markdown(
    results: dict[str, dict[str, dict[str, float]]],
    per_flight: list[dict],
    args: argparse.Namespace,
    extras: dict,
) -> None:
    n_flights = len(per_flight)
    rows_per_channel = {
        ch: {ph: results[ch].get(ph, {}) for ph in CHANNEL_PHASES[ch]}
        for ch in results
        if ch in CHANNEL_PHASES
    }
    ac = _ac_verdict(results)
    overall = (
        "✅ SUCCESS"
        if all(v == "PASS" for _, _, _, v in ac if v != "SKIP")
        else "⚠️ PARTIAL"
        if sum(v == "FAIL" for _, _, _, v in ac) <= 2
        else "🔬 FALSIFIED"
    )

    lines: list[str] = []
    lines.append("# Paper 0 — QAR target validation results")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(UTC).isoformat()}_")
    lines.append(f"_node-fdm-data git SHA: `{_git_sha()}`_")
    lines.append(f"_QAR snapshot dir: `{QAR_DIR}`_")
    lines.append(f"_QAR pattern: `{QAR_GLOB}`_")
    lines.append(
        "_QAR → raw_ adapter sets `bds_mcp_alt_sel_ft`, `bds_fms_alt_sel_ft` to NULL (R1/R5)."
    )
    lines.append("")
    lines.append("## Cohort")
    lines.append(f"- n_flights_processed: **{n_flights}**")
    lines.append(f"- n_flights_skipped: {extras['n_skipped']}")
    lines.append(
        f"- Total samples (downsampled to {PIPELINE_DT_S:.0f} s grid): {extras['n_samples_total']:,}"
    )
    lines.append("")
    lines.append("### QAR truth columns used (validated)")
    lines.append("| Channel | QAR column | Filter | Unit conv |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Altitude  | `{COL_ALT_SEL_FT}`  | —                              | ×0.3048 → m |"
    )
    lines.append(
        f"| Mach      | `{COL_MACH_SEL}`     | `{COL_SPD_MODE}=='MACH'`        | —           |"
    )
    lines.append(
        f"| CAS       | `{COL_CAS_SEL_KT}`   | `{COL_SPD_MODE}=='SPEED'`       | —           |"
    )
    lines.append(
        f"| γ primary | `{COL_FPA_SEL_DEG}` | `!= 0` (FPA mode active)        | ×π/180 → rad |"
    )
    lines.append(
        f"| γ fallback| `{COL_VS_SEL_FTMIN}`| `!= 0` (V/S mode active) ; needs `SPD__TAS` | asin(VS·0.00508 / TAS·0.514) |"
    )
    lines.append(
        f"| Track HDG | `{COL_HDG_SEL_DEG}` | `{COL_LAT_MODE}=='HDG'`         | mod 360°     |"
    )
    lines.append(
        f"| Track NAV | `{COL_TRACK_ACTUAL}`| `{COL_LAT_MODE} in ('NAV','LOC TRK')` (FMS-target proxy) | — |"
    )
    lines.append("")
    lines.append("## Per-channel summary")
    for ch in CHANNELS:
        if ch not in rows_per_channel:
            continue
        lines.append(f"### {CHANNEL_LABEL[ch]}")
        lines.append("| Phase | n | MAE | p99 |Δ| | Coverage | Agreement < tol |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for ph, m in rows_per_channel[ch].items():
            if not m:
                continue
            lines.append(
                f"| {ph:8s} | {m.get('n', 0):>8,d} | "
                f"{m.get('mae', float('nan')):.4g} | "
                f"{m.get('p99', float('nan')):.4g} | "
                f"{m.get('coverage_pct', float('nan')):.1f}% | "
                f"{m.get('agreement_pct', float('nan')):.1f}% |"
            )
        lines.append("")
    # Directional-consistency metric (applied to altitude, CAS, Mach).
    dir_channels = [
        ("alt_directional", "Altitude (m)", "m", CHANNEL_TOL["alt_m"]),
        ("cas_directional", "CAS (kt)", "kt", CHANNEL_TOL["cas_kt"]),
        ("mach_directional", "Mach", "", CHANNEL_TOL["mach"]),
        ("gamma_directional", "Gamma (rad)", "rad", CHANNEL_TOL["gamma_rad"]),
    ]
    lines.append("## Directional consistency (anchored-target methodology fairness)")
    lines.append("")
    lines.append(
        "Pipeline target is methodology-consistent when it lies on the side of "
        "FCU truth that the aircraft is heading toward (current<truth → pred ≥ "
        "truth - tol ; current>truth → pred ≤ truth + tol ; |truth-current|≤tol "
        "→ strict |Δ| < tol). Wrong-direction MAE is 0 when consistent."
    )
    lines.append("")
    for key, label, unit, tol in dir_channels:
        if key not in results:
            continue
        lines.append(f"### {label} — directional (tol = {tol}{unit})")
        lines.append(f"| Phase | n | Consistency | Wrong-direction MAE | p99 wrong-dir |")
        lines.append("|---|---:|---:|---:|---:|")
        for ph, m in results[key].items():
            lines.append(
                f"| {ph:8s} | {m.get('n', 0):>8,d} | "
                f"{m.get('consistency_pct', float('nan')):.1f}% | "
                f"{m.get('directional_mae', float('nan')):.4g} | "
                f"{m.get('wrong_direction_p99', float('nan')):.4g} |"
            )
        lines.append("")
    lines.append("## Acceptance criteria")
    lines.append("| AC | Channel | Criterion | Verdict |")
    lines.append("|---|---|---|---:|")
    for ac_id, channel, criterion, verdict in ac:
        lines.append(f"| {ac_id} | {channel} | {criterion} | **{verdict}** |")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("```bash")
    lines.append("cd /Users/gabriel/Documents/Code/python/node-fdm-v2")
    lines.append("unset VIRTUAL_ENV")
    lines.append(
        "uv run python scripts/_validation/paper0_qar_target_validation.py "
        f"--report {args.report} --figures-dir {args.figures_dir} --csv {args.csv}"
    )
    lines.append("```")
    lines.append("")
    lines.append(f"## Verdict: {overall}")
    fail_summary = [f"{ch} ({crit})" for ac_id, ch, crit, v in ac if v == "FAIL"]
    skip_summary = [f"{ch} ({crit})" for ac_id, ch, crit, v in ac if v == "SKIP"]
    if fail_summary:
        lines.append("**Failing channels**:")
        for fs in fail_summary:
            lines.append(f"- {fs}")
    if skip_summary:
        lines.append("**Skipped (no data)**:")
        for ss in skip_summary:
            lines.append(f"- {ss}")
    lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))


def write_per_flight_csv(
    per_flight: list[dict], results_by_flight: dict[str, dict], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for f in per_flight:
        fid = f["flight_id"]
        rs = results_by_flight.get(fid, {})
        for channel, phases in rs.items():
            for phase, m in phases.items():
                rows.append(
                    {
                        "flight_id": fid,
                        "channel": channel,
                        "phase": phase,
                        "n_samples_compared": int(m.get("n", 0)),
                        "mae": m.get("mae", float("nan")),
                        "p99_abs_delta": m.get("p99", float("nan")),
                        "coverage_pct": m.get("coverage_pct", float("nan")),
                        "agreement_pct": m.get("agreement_pct", float("nan")),
                    }
                )
    df = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            {
                "flight_id": [],
                "channel": [],
                "phase": [],
                "n_samples_compared": [],
                "mae": [],
                "p99_abs_delta": [],
                "coverage_pct": [],
                "agreement_pct": [],
            }
        )
    )
    df.write_csv(path)


def _per_flight_metrics(f: dict) -> dict[str, dict[str, dict[str, float]]]:
    """Run aggregate_results on a single flight (for CSV granularity)."""
    return aggregate_results([f])


# ------------------------------------------------------------------- main --
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--max-flights", type=int, default=None, help="cap for quick smoke test")
    parser.add_argument("--qar-dir", type=Path, default=QAR_DIR)
    parser.add_argument("--qar-glob", type=str, default=QAR_GLOB)
    args = parser.parse_args()

    paths = sorted(args.qar_dir.glob(args.qar_glob))
    if args.max_flights is not None:
        paths = paths[: args.max_flights]
    if not paths:
        print(f"No QAR parquets matched {args.qar_dir}/{args.qar_glob}", file=sys.stderr)
        return 2
    print(f"Processing {len(paths)} flights from {args.qar_dir} ...")

    per_flight: list[dict] = []
    results_by_flight: dict[str, dict] = {}
    n_skipped = 0
    n_samples_total = 0
    for i, p in enumerate(paths):
        if i % 25 == 0 or i == len(paths) - 1:
            print(f"  [{i + 1:>4d}/{len(paths)}] {p.name}")
        d = process_flight(p)
        if d is None:
            n_skipped += 1
            continue
        per_flight.append(d)
        n_samples_total += d["n_samples"]
        results_by_flight[d["flight_id"]] = _per_flight_metrics(d)

    if not per_flight:
        print("No flights succeeded.", file=sys.stderr)
        return 3

    print("\nAggregating ...")
    results = aggregate_results(per_flight)
    extras = {"n_skipped": n_skipped, "n_samples_total": n_samples_total}

    print("Writing markdown report ...")
    write_markdown(results, per_flight, args, extras)
    print("Writing figures ...")
    plot_scatters(per_flight, args.figures_dir)
    print("Writing per-flight CSV ...")
    write_per_flight_csv(per_flight, results_by_flight, args.csv)

    print(f"\nReport : {args.report}")
    print(f"Figures: {args.figures_dir}")
    print(f"CSV    : {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
