# ruff: noqa: E501, S603, T201, BLE001
"""Paper 0 §8 — Bilateral detector hyperparameter tuning (100/100 cohort).

Runs an Optuna TPE study per channel (mach / cas / vz / alt / gamma) to find
better ``bilateral_*`` filter hyperparameters than the production defaults.

Strategy (see PROMPT):

1. Pre-cache (qar_to_raw_schema + downsample) for the 200 cohort flights to
   ``/tmp/p0_finetune_cache/{stem}.parquet`` so per-trial work is just
   ``clean_bds_speeds → build_selected_params → augment_lateral`` (~0.3-0.5 s).
2. TPE sampler + MedianPruner (n_warmup_steps=5, n_startup_trials=10).
3. Channels run sequentially. Top-3 candidates re-evaluated on a disjoint
   validation set (next 100 flights).
4. Results dumped to ``/tmp/p0_finetune_results.json`` and a markdown report
   at the path passed via ``--report``.

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    unset VIRTUAL_ENV
    uv run python scripts/_validation/paper0_qar_target_finetune.py \\
        --tuning-flights 100 --validation-flights 100 \\
        --channels mach,cas,vz,alt,gamma \\
        --trials 50 \\
        --report /Users/gabriel/axm/04-papers/PS_MODEL/PAPER_0_DATA_PREPROCESSING/experiments/03_qar_target_finetuning.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Reuse the validation helpers (cohort glob, schema adapter, metrics).
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import paper0_qar_target_validation as v  # noqa: E402

from node_fdm_data.lateral import augment_lateral  # noqa: E402
from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds  # noqa: E402
from node_fdm_data.segments import build_selected_params  # noqa: E402
from node_fdm_pipeline.config import (  # noqa: E402
    AltFilterConfig,
    CasFilterConfig,
    GammaFilterConfig,
    MachFilterConfig,
    SelectedParamConfig,
    VzFilterConfig,
)

# ---------------------------------------------------------------- constants --
CACHE_DIR = Path("/tmp/p0_finetune_cache")
RESULTS_JSON = Path("/tmp/p0_finetune_results.json")
QAR_DIR = v.QAR_DIR
QAR_GLOB = v.QAR_GLOB
CHANNEL_TOL = v.CHANNEL_TOL
FT_TO_M = v.FT_TO_M
FTMIN_TO_MS = v.FTMIN_TO_MS
KT_TO_MS = v.KT_TO_MS
PIPELINE_DT_S = v.PIPELINE_DT_S

# Per-channel: which columns are predicted vs which raw signal is the
# "self-consistency" reference, plus default param block.
CHANNELS = ("mach", "cas", "vz", "alt", "gamma")

# Quiet down chatty deps.
logging.getLogger().setLevel(logging.WARNING)


# ----------------------------------------------------------- cache builder --
def _cohort_paths(n_total: int) -> list[Path]:
    """Return the first ``n_total`` A320 parquet paths, sorted."""
    paths = sorted(QAR_DIR.glob(QAR_GLOB))
    if len(paths) < n_total:
        msg = f"Cohort too small: requested {n_total}, found {len(paths)}"
        raise RuntimeError(msg)
    return paths[:n_total]


def _build_cache(paths: list[Path]) -> list[Path]:
    """Pre-compute ``qar_to_raw_schema → downsample`` once per flight.

    Returns the list of cached parquet paths (one per input). Failed flights
    are skipped (returned list may be shorter than input).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    print(f"[cache] preparing {len(paths)} flights at {CACHE_DIR}", file=sys.stderr)
    t0 = time.time()
    for i, p in enumerate(paths):
        flight_id = p.stem
        cache_path = CACHE_DIR / f"{flight_id}.parquet"
        truth_path = CACHE_DIR / f"{flight_id}.truth.parquet"
        if cache_path.exists() and truth_path.exists():
            out.append(cache_path)
            continue
        try:
            df_qar = pl.read_parquet(p)
            truth_full = v.extract_qar_truth(df_qar)
            truth_idx = np.arange(0, df_qar.height, v.QAR_DOWNSAMPLE)
            truth = {k: val[truth_idx] for k, val in truth_full.items()}
            df_raw = v.qar_to_raw_schema(df_qar, flight_id)
            df_raw = v.downsample(df_raw, v.QAR_DOWNSAMPLE)
            df_raw.write_parquet(cache_path)
            # Persist truth as a parquet with float columns.
            truth_df = pl.DataFrame({k: pl.Series(k, val) for k, val in truth.items()})
            truth_df.write_parquet(truth_path)
            out.append(cache_path)
        except Exception as exc:
            print(f"  [SKIP cache] {p.name}: {exc}", file=sys.stderr)
            continue
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [cache] {i + 1}/{len(paths)} ({elapsed:.0f}s)", file=sys.stderr)
    print(f"[cache] {len(out)} usable / {len(paths)} requested in {time.time() - t0:.0f}s", file=sys.stderr)
    return out


# -------------------------------------------------------- per-flight engine --
def _process_cached(cache_path: Path, sel_params: dict[str, Any]) -> dict | None:
    """Run preprocessing + metric prep on a cached flight, using ``sel_params``.

    Mirrors v.process_flight from the ``df_raw = downsample(...)`` step
    forward — uses the cached pair (raw, truth).
    """
    flight_id = cache_path.stem
    truth_path = cache_path.with_suffix(".truth.parquet")
    try:
        df_raw = pl.read_parquet(cache_path)
        truth_df = pl.read_parquet(truth_path)
    except Exception as exc:
        print(f"  [SKIP load] {flight_id}: {exc}", file=sys.stderr)
        return None
    truth_aligned = {c: truth_df[c].to_numpy() for c in truth_df.columns}
    try:
        df_proc = _run_preprocessing_with_cfg(df_raw, sel_params)
    except Exception as exc:
        print(
            f"  [SKIP pipe] {flight_id}: {exc}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return None

    # Re-align truth to processed rows by meta_row_idx if pipeline reordered.
    truth_idx_full = np.arange(0, len(truth_aligned[next(iter(truth_aligned))]) * v.QAR_DOWNSAMPLE, v.QAR_DOWNSAMPLE)
    if "meta_row_idx" in df_proc.columns:
        keep_idx = df_proc["meta_row_idx"].to_numpy()
        ds_to_pos = {int(val): i for i, val in enumerate(truth_idx_full)}
        positions = np.array([ds_to_pos.get(int(val), -1) for val in keep_idx])
        valid = positions >= 0
        positions = positions[valid]
        truth_aligned = {k: val[positions] for k, val in truth_aligned.items()}

    def _pull(name: str) -> np.ndarray:
        if name in df_proc.columns:
            return df_proc[name].cast(pl.Float64).to_numpy()
        return np.full(df_proc.height, np.nan)

    def _pull_bool(name: str) -> np.ndarray:
        if name in df_proc.columns:
            return np.asarray(df_proc[name].to_numpy(), dtype=bool)
        return np.zeros(df_proc.height, dtype=bool)

    alt_pred_m = _pull("fdm_alt_target_ft") * FT_TO_M
    alt_pred_sel_m = _pull("fdm_alt_sel_ft") * FT_TO_M
    cas_pred_kt = _pull("fdm_cas_sel_kt")
    mach_pred = _pull("fdm_mach_sel")
    vz_pred_ftmin = _pull("fdm_vz_sel_ftmin")
    gamma_pred = _pull("fdm_gamma_target_rad")

    alt_ft_proc = df_proc["raw_alt_ft"].cast(pl.Float64).to_numpy()
    alt_m_proc = alt_ft_proc * FT_TO_M

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
    gamma_current = (
        df_proc["fdm_gamma_rad"].cast(pl.Float64).to_numpy()
        if "fdm_gamma_rad" in df_proc.columns
        else np.full(df_proc.height, np.nan)
    )
    vz_current_ftmin = df_proc["raw_vz_ftmin"].cast(pl.Float64).to_numpy()

    alt_known = np.isfinite(alt_pred_m)
    cas_known = np.isfinite(cas_pred_kt)
    mach_known = np.isfinite(mach_pred)
    vz_known = np.isfinite(vz_pred_ftmin)
    gamma_known = (
        _pull_bool("fdm_gamma_target_known")
        if "fdm_gamma_target_known" in df_proc.columns
        else np.isfinite(gamma_pred)
    )

    phase = v.classify_phase(df_proc)

    return {
        "flight_id": flight_id,
        "alt_current_m": alt_m_proc,
        "alt_pred_m": alt_pred_m,
        "alt_pred_sel_m": alt_pred_sel_m,
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
        "vz_current_ftmin": vz_current_ftmin,
        "vz_pred_ftmin": vz_pred_ftmin,
        "vz_truth_ftmin": truth_aligned["vz_truth_ftmin"],
        "vz_known": vz_known,
        "phase": phase,
    }


def _run_preprocessing_with_cfg(df_raw: pl.DataFrame, sel_params: dict[str, Any]) -> pl.DataFrame:
    """Same as v.run_preprocessing but takes the sel_params dict explicitly."""
    df = clean_bds_speeds(df_raw)
    if {"raw_vz_ftmin", "fdm_tas_from_cas_kt"}.issubset(df.columns):
        vz_ms = pl.col("raw_vz_ftmin") * FTMIN_TO_MS
        tas_ms = (pl.col("fdm_tas_from_cas_kt") * KT_TO_MS).clip(lower_bound=1e-6)
        df = df.with_columns(
            (vz_ms / tas_ms).clip(-1.0, 1.0).arcsin().alias("fdm_gamma_rad"),
        )
    df = build_selected_params(df, sel_params)
    df = augment_lateral(df, dt=PIPELINE_DT_S)
    return df


# ---------------------------------------------------------- channel metric --
def _channel_J(per_flight: list[dict], channel: str) -> tuple[float, dict[str, float]]:
    """Compute J(params) for one channel from a list of per-flight dicts.

    Returns ``(J, extras)`` where extras contains the raw rates feeding J.
    """
    if not per_flight:
        return 10.0, {"n_level": 0, "self_pct": 0.0, "chain_pct": 0.0, "wrong_mae": 0.0}

    if channel == "mach":
        pred_key, truth_key, current_key, known_key, tol = (
            "mach_pred",
            "mach_truth",
            "mach_current",
            "mach_known",
            CHANNEL_TOL["mach"],
        )
        # mach uses pointwise pred for self-consistency.
        raw_key = "mach_current"
        angular = False
    elif channel == "cas":
        pred_key, truth_key, current_key, known_key, tol = (
            "cas_pred_kt",
            "cas_truth_kt",
            "cas_current_kt",
            "cas_known",
            CHANNEL_TOL["cas_kt"],
        )
        raw_key = "cas_current_kt"
        angular = False
    elif channel == "vz":
        pred_key, truth_key, current_key, known_key, tol = (
            "vz_pred_ftmin",
            "vz_truth_ftmin",
            "vz_current_ftmin",
            "vz_known",
            CHANNEL_TOL["vz_ftmin"],
        )
        raw_key = "vz_current_ftmin"
        angular = False
    elif channel == "alt":
        pred_key, truth_key, current_key, known_key, tol = (
            "alt_pred_m",
            "alt_truth_m",
            "alt_current_m",
            "alt_known",
            CHANNEL_TOL["alt_m"],
        )
        # Altitude self-consistency uses the point-wise plateau (alt_pred_sel_m)
        # not the anchored target (which has 100% by construction).
        raw_key = "alt_current_m"
        angular = False
    elif channel == "gamma":
        pred_key, truth_key, current_key, known_key, tol = (
            "gamma_pred_rad",
            "gamma_truth_rad",
            "gamma_current_rad",
            "gamma_known",
            CHANNEL_TOL["gamma_rad"],
        )
        raw_key = "gamma_current_rad"
        angular = False
    else:
        msg = f"Unknown channel: {channel}"
        raise ValueError(msg)

    pooled_pred = np.concatenate([f[pred_key] for f in per_flight])
    pooled_truth = np.concatenate([f[truth_key] for f in per_flight])
    pooled_current = np.concatenate([f[current_key] for f in per_flight])
    pooled_known = np.concatenate([f[known_key] for f in per_flight]).astype(bool)
    pooled_phase = np.concatenate([f["phase"] for f in per_flight])
    pooled_raw = np.concatenate([f[raw_key] for f in per_flight])

    # For altitude self-consistency we want the *point-wise* alt_sel, not
    # the anchored target.
    if channel == "alt":
        pooled_self_pred = np.concatenate([f["alt_pred_sel_m"] for f in per_flight])
        pooled_self_known = np.isfinite(pooled_self_pred)
    else:
        pooled_self_pred = pooled_pred
        pooled_self_known = pooled_known

    # Chain consistency (level phase).
    chain = v.compute_directional_consistency(
        pooled_pred,
        pooled_truth,
        pooled_current,
        pooled_known,
        pooled_phase,
        tol,
        phases=("level",),
    )["level"]

    # Self consistency (level phase).
    self_c = v.compute_self_consistency(
        pooled_self_pred,
        pooled_raw,
        pooled_self_known,
        pooled_phase,
        tol,
        phases=("level",),
        angular=angular,
    )["level"]

    n_eligible = int(chain["n"])
    if n_eligible < 100:
        return 10.0, {
            "n_level": n_eligible,
            "self_pct": float(self_c.get("self_consistency_pct", float("nan"))),
            "chain_pct": float(chain.get("consistency_pct", float("nan"))),
            "wrong_mae": float(chain.get("directional_mae", float("nan"))),
        }

    wrong_mae = float(chain["directional_mae"])
    chain_rate = float(chain["consistency_pct"]) / 100.0
    self_rate = float(self_c["self_consistency_pct"]) / 100.0
    if not np.isfinite(self_rate):
        self_rate = 0.0
    if not np.isfinite(chain_rate):
        chain_rate = 0.0
    if not np.isfinite(wrong_mae):
        wrong_mae = tol * 5.0  # large penalty

    J = wrong_mae / tol + 0.5 * (1.0 - self_rate) + 0.5 * (1.0 - chain_rate)
    return float(J), {
        "n_level": n_eligible,
        "self_pct": float(self_c["self_consistency_pct"]),
        "chain_pct": float(chain["consistency_pct"]),
        "wrong_mae": wrong_mae,
    }


def _all_phase_metrics(per_flight: list[dict], channel: str) -> dict[str, dict[str, float]]:
    """Compute self+chain consistency at level / climb / descent / global.

    Returns a nested ``{phase: {self_pct, chain_pct, wrong_mae, n}}`` dict.
    """
    if not per_flight:
        return {}

    if channel == "mach":
        pred_key, truth_key, current_key, known_key, tol = (
            "mach_pred", "mach_truth", "mach_current", "mach_known", CHANNEL_TOL["mach"],
        )
        raw_key = "mach_current"
    elif channel == "cas":
        pred_key, truth_key, current_key, known_key, tol = (
            "cas_pred_kt", "cas_truth_kt", "cas_current_kt", "cas_known", CHANNEL_TOL["cas_kt"],
        )
        raw_key = "cas_current_kt"
    elif channel == "vz":
        pred_key, truth_key, current_key, known_key, tol = (
            "vz_pred_ftmin", "vz_truth_ftmin", "vz_current_ftmin", "vz_known", CHANNEL_TOL["vz_ftmin"],
        )
        raw_key = "vz_current_ftmin"
    elif channel == "alt":
        pred_key, truth_key, current_key, known_key, tol = (
            "alt_pred_m", "alt_truth_m", "alt_current_m", "alt_known", CHANNEL_TOL["alt_m"],
        )
        raw_key = "alt_current_m"
    elif channel == "gamma":
        pred_key, truth_key, current_key, known_key, tol = (
            "gamma_pred_rad", "gamma_truth_rad", "gamma_current_rad", "gamma_known", CHANNEL_TOL["gamma_rad"],
        )
        raw_key = "gamma_current_rad"
    else:
        msg = f"Unknown channel: {channel}"
        raise ValueError(msg)

    pooled_pred = np.concatenate([f[pred_key] for f in per_flight])
    pooled_truth = np.concatenate([f[truth_key] for f in per_flight])
    pooled_current = np.concatenate([f[current_key] for f in per_flight])
    pooled_known = np.concatenate([f[known_key] for f in per_flight]).astype(bool)
    pooled_phase = np.concatenate([f["phase"] for f in per_flight])
    pooled_raw = np.concatenate([f[raw_key] for f in per_flight])

    if channel == "alt":
        pooled_self_pred = np.concatenate([f["alt_pred_sel_m"] for f in per_flight])
        pooled_self_known = np.isfinite(pooled_self_pred)
    else:
        pooled_self_pred = pooled_pred
        pooled_self_known = pooled_known

    phases = ("level", "climb", "descent", "global")
    chain = v.compute_directional_consistency(
        pooled_pred, pooled_truth, pooled_current, pooled_known, pooled_phase, tol, phases=phases,
    )
    self_c = v.compute_self_consistency(
        pooled_self_pred, pooled_raw, pooled_self_known, pooled_phase, tol, phases=phases,
    )
    out: dict[str, dict[str, float]] = {}
    for ph in phases:
        out[ph] = {
            "n_chain": int(chain[ph]["n"]),
            "self_pct": float(self_c[ph]["self_consistency_pct"]),
            "self_mae": float(self_c[ph]["self_consistency_mae"]),
            "chain_pct": float(chain[ph]["consistency_pct"]),
            "wrong_mae": float(chain[ph]["directional_mae"]),
        }
    return out


# ------------------------------------------------------- optuna integration --
def _build_sel_params(channel: str, params: dict[str, Any]) -> dict[str, Any]:
    """Build a full SEL_PARAMS_CONFIG dict with ``channel`` overridden.

    Other channels keep their Pydantic defaults.
    """
    base = SelectedParamConfig().model_dump()
    base["alt_hold_relax"] = 15
    if channel == "mach":
        base["mach"] = MachFilterConfig(**params).model_dump()
    elif channel == "cas":
        base["cas"] = CasFilterConfig(**params).model_dump()
    elif channel == "vz":
        base["vz"] = VzFilterConfig(**params).model_dump()
    elif channel == "alt":
        base["alt"] = AltFilterConfig(**params).model_dump()
    elif channel == "gamma":
        base["gamma"] = GammaFilterConfig(**params).model_dump()
    else:
        msg = f"Unknown channel: {channel}"
        raise ValueError(msg)
    return base


def _sample_params(channel: str, trial: Any) -> dict[str, Any]:
    """Sample hyperparameters for a given channel from the trial's search space."""
    if channel == "mach":
        return {
            "mode": "bilateral_mach",
            "sigma_s": trial.suggest_float("sigma_s", 4.0, 15.0),
            "sigma_r": trial.suggest_float("sigma_r", 0.02, 0.20),
            "n_passes": trial.suggest_int("n_passes", 1, 4),
            "slope_tol": trial.suggest_float("slope_tol", 3.25e-4, 1.3e-3, log=True),
            "flat_tol": trial.suggest_float("flat_tol", 2.5e-2, 1.0e-1, log=True),
            "min_len": trial.suggest_int("min_len", 5, 30),
        }
    if channel == "cas":
        return {
            "mode": "bilateral_cas",
            "sigma_s": trial.suggest_float("sigma_s", 4.0, 15.0),
            "sigma_r": trial.suggest_float("sigma_r", 5.0, 30.0),
            "n_passes": trial.suggest_int("n_passes", 1, 4),
            "slope_tol": trial.suggest_float("slope_tol", 0.125, 0.5),
            "flat_tol": trial.suggest_float("flat_tol", 10.0, 40.0),
            "min_len": trial.suggest_int("min_len", 5, 30),
        }
    if channel == "vz":
        return {
            "mode": "bilateral_vz",
            "sigma_s": trial.suggest_float("sigma_s", 4.0, 15.0),
            "sigma_r": trial.suggest_float("sigma_r", 100.0, 500.0),
            "slope_tol": trial.suggest_float("slope_tol", 7.5, 30.0),
            "flat_tol": trial.suggest_float("flat_tol", 50.0, 200.0),
            "min_len": trial.suggest_int("min_len", 5, 30),
            "min_abs_value": trial.suggest_float("min_abs_value", 0.0, 100.0),
        }
    if channel == "alt":
        return {
            "mode": "bilateral_vz",
            "sigma_s": trial.suggest_float("sigma_s", 4.0, 15.0),
            "sigma_r": trial.suggest_float("sigma_r", 100.0, 500.0),
            "n_passes": trial.suggest_int("n_passes", 1, 4),
            "tol_ftmin": trial.suggest_float("tol_ftmin", 75.0, 300.0),
            "min_len": trial.suggest_int("min_len", 5, 30),
        }
    if channel == "gamma":
        return {
            "mode": "bilateral_gamma",
            "sigma_s": trial.suggest_float("sigma_s", 4.0, 15.0),
            "sigma_r": trial.suggest_float("sigma_r", 5e-3, 3e-2, log=True),
            "slope_tol": trial.suggest_float("slope_tol", 1.5e-4, 6e-4, log=True),
            "flat_tol": trial.suggest_float("flat_tol", 1e-3, 4e-3, log=True),
            "abs_min": trial.suggest_float("abs_min", 0.0, 0.01),
            "min_len": trial.suggest_int("min_len", 5, 30),
        }
    msg = f"Unknown channel: {channel}"
    raise ValueError(msg)


def _evaluate(channel: str, params: dict[str, Any], cache_paths: list[Path], *, prune_hook=None) -> tuple[float, dict[str, float]]:
    """Run all cached flights with given channel params; return (J, extras).

    If ``prune_hook`` is provided, called with (i, partial_J, n_flights) after
    each flight — may raise ``optuna.TrialPruned``.
    """
    sel_params = _build_sel_params(channel, params)
    per_flight: list[dict] = []
    for i, cp in enumerate(cache_paths):
        out = _process_cached(cp, sel_params)
        if out is not None:
            per_flight.append(out)
        if prune_hook is not None:
            # Compute partial J on what we have so far.
            partial_J, _extras = _channel_J(per_flight, channel)
            prune_hook(i, partial_J)
    J, extras = _channel_J(per_flight, channel)
    return J, extras


def _baseline_params(channel: str) -> dict[str, Any]:
    """Return the production default params for a given channel."""
    if channel == "mach":
        return MachFilterConfig().model_dump()
    if channel == "cas":
        return CasFilterConfig().model_dump()
    if channel == "vz":
        return VzFilterConfig().model_dump()
    if channel == "alt":
        return AltFilterConfig().model_dump()
    if channel == "gamma":
        return GammaFilterConfig().model_dump()
    msg = f"Unknown channel: {channel}"
    raise ValueError(msg)


def _baseline_evaluate(channel: str, cache_paths: list[Path]) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
    """Evaluate baseline params; return (J, extras, all_phase_metrics)."""
    sel_params = _build_sel_params(channel, _baseline_params(channel))
    per_flight: list[dict] = []
    for cp in cache_paths:
        out = _process_cached(cp, sel_params)
        if out is not None:
            per_flight.append(out)
    J, extras = _channel_J(per_flight, channel)
    all_phases = _all_phase_metrics(per_flight, channel)
    return J, extras, all_phases


def _tune_channel(channel: str, tune_paths: list[Path], val_paths: list[Path], n_trials: int) -> dict[str, Any]:
    """Run a TPE study on a single channel; return the result dict."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\n=== Tuning channel: {channel} ({n_trials} trials, {len(tune_paths)} tuning flights) ===", file=sys.stderr)
    t0 = time.time()

    # Baseline on tuning set.
    base_J_tune, base_extras_tune, _ = _baseline_evaluate(channel, tune_paths)
    print(f"  [baseline tune] J={base_J_tune:.4f} extras={base_extras_tune}", file=sys.stderr)

    sampler = TPESampler(seed=42, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def objective(trial: Any) -> float:
        params = _sample_params(channel, trial)
        sel_params = _build_sel_params(channel, params)
        per_flight: list[dict] = []
        for i, cp in enumerate(tune_paths):
            out = _process_cached(cp, sel_params)
            if out is not None:
                per_flight.append(out)
            if (i + 1) % 10 == 0 and (i + 1) >= 20:
                partial_J, _extras = _channel_J(per_flight, channel)
                trial.report(partial_J, step=i + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        J, _extras = _channel_J(per_flight, channel)
        return J

    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except KeyboardInterrupt:
        print(f"  [interrupted] using best-so-far for {channel}", file=sys.stderr)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params["mode"] = _baseline_params(channel)["mode"]  # mode is fixed
    best_J_tune = float(best_trial.value)
    print(f"  [best tune]    J={best_J_tune:.4f} params={best_params}", file=sys.stderr)

    # Pick top-3 completed trials for validation.
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value if t.value is not None else float("inf"))
    top_trials = completed[:3]

    # Validation pass.
    val_results = []
    for rank, t in enumerate(top_trials):
        params = dict(t.params)
        params["mode"] = _baseline_params(channel)["mode"]
        sel_params = _build_sel_params(channel, params)
        per_flight = []
        for cp in val_paths:
            out = _process_cached(cp, sel_params)
            if out is not None:
                per_flight.append(out)
        J_val, extras_val = _channel_J(per_flight, channel)
        all_phases_val = _all_phase_metrics(per_flight, channel)
        val_results.append({
            "rank": rank,
            "J_tune": float(t.value),
            "J_val": J_val,
            "params": params,
            "extras_val": extras_val,
            "all_phases_val": all_phases_val,
        })
        print(f"  [val rank {rank}]  J_tune={t.value:.4f} J_val={J_val:.4f}", file=sys.stderr)

    # Baseline on validation set (with full phase metrics).
    base_J_val, base_extras_val, base_phases_val = _baseline_evaluate(channel, val_paths)
    print(f"  [baseline val] J={base_J_val:.4f}", file=sys.stderr)

    # Pick the val-best (lowest J on val) — that's the "honest" winner.
    if val_results:
        val_best = min(val_results, key=lambda r: r["J_val"])
    else:
        val_best = {"rank": -1, "J_tune": float("nan"), "J_val": float("nan"), "params": best_params, "extras_val": {}, "all_phases_val": {}}

    elapsed = time.time() - t0
    print(f"  [done] {channel} in {elapsed:.0f}s", file=sys.stderr)

    return {
        "channel": channel,
        "n_trials_completed": len(completed),
        "n_trials_pruned": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
        "baseline_J_tune": base_J_tune,
        "baseline_extras_tune": base_extras_tune,
        "baseline_J_val": base_J_val,
        "baseline_extras_val": base_extras_val,
        "baseline_phases_val": base_phases_val,
        "best_J_tune": best_J_tune,
        "best_params_tune": best_params,
        "val_results_top3": val_results,
        "val_best_params": val_best["params"],
        "val_best_J": val_best["J_val"],
        "val_best_phases": val_best["all_phases_val"],
        "elapsed_s": elapsed,
    }


# ------------------------------------------------------------- markdown out --
# Bilateral-relevant keys per channel (others are legacy savgol fields that
# would just add noise to the diff).
_RELEVANT_KEYS = {
    "mach":  ("sigma_s", "sigma_r", "n_passes", "slope_tol", "flat_tol", "min_len"),
    "cas":   ("sigma_s", "sigma_r", "n_passes", "slope_tol", "flat_tol", "min_len"),
    "vz":    ("sigma_s", "sigma_r", "slope_tol", "flat_tol", "min_len", "min_abs_value"),
    "alt":   ("sigma_s", "sigma_r", "n_passes", "tol_ftmin", "min_len"),
    "gamma": ("sigma_s", "sigma_r", "slope_tol", "flat_tol", "abs_min", "min_len"),
}


def _format_params_diff(baseline: dict[str, Any], best: dict[str, Any], channel: str | None = None) -> str:
    """Return a markdown bullet list of parameter changes (bilateral keys only)."""
    keys = _RELEVANT_KEYS.get(channel or "", tuple(sorted(set(baseline) | set(best))))
    lines = []
    for k in keys:
        if k == "mode":
            continue
        bv = baseline.get(k)
        nv = best.get(k)
        if bv is None and nv is None:
            continue
        marker = "" if bv == nv else " **"
        end = "" if bv == nv else "**"
        if isinstance(bv, (int, float)) and isinstance(nv, (int, float)):
            lines.append(f"  - `{k}`: {float(bv):.4g} ->{marker}{float(nv):.4g}{end}")
            continue
        lines.append(f"  - `{k}`: {bv} ->{marker}{nv}{end}")
    if not lines:
        return "  - (no change)"
    return "\n".join(lines)


def _format_phase_table(baseline_phases: dict[str, dict[str, float]], best_phases: dict[str, dict[str, float]]) -> str:
    """Return a markdown table comparing baseline vs best per phase."""
    rows = ["| Phase | n | Self% base | Self% best | Chain% base | Chain% best | wrongMAE base | wrongMAE best |",
            "|---|---|---|---|---|---|---|---|"]
    for ph in ("level", "climb", "descent", "global"):
        bm = baseline_phases.get(ph, {})
        nm = best_phases.get(ph, {})
        def _fmt(d: dict, k: str, prec: int = 1) -> str:
            v_ = d.get(k)
            if v_ is None:
                return "—"
            try:
                vf = float(v_)
            except (TypeError, ValueError):
                return "—"
            if not np.isfinite(vf):
                return "—"
            return f"{vf:.{prec}f}"
        rows.append(
            f"| {ph} | {bm.get('n_chain', '—')} | "
            f"{_fmt(bm, 'self_pct')} | {_fmt(nm, 'self_pct')} | "
            f"{_fmt(bm, 'chain_pct')} | {_fmt(nm, 'chain_pct')} | "
            f"{_fmt(bm, 'wrong_mae', 4)} | {_fmt(nm, 'wrong_mae', 4)} |"
        )
    return "\n".join(rows)


def write_markdown(results: list[dict[str, Any]], report_path: Path, *, n_tune: int, n_val: int) -> None:
    """Write the markdown finetune report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# Paper 0 §8 — QAR Bilateral Detector Hyperparameter Tuning\n")
    lines.append(f"_Generated: {now}_\n")
    lines.append(f"_Cohort: A320, first {n_tune} flights (tuning) + next {n_val} (validation, disjoint)._\n")
    lines.append(
        "Objective per channel — minimize on **level** phase:\n\n"
        "```\n"
        "J = wrong_dir_MAE / tol  +  0.5 * (1 - self_consistency_rate)  +  0.5 * (1 - chain_consistency_rate)\n"
        "```\n"
    )
    lines.append(
        "Phases <100 samples receive J=10. Top-3 tuning candidates are re-evaluated on "
        "the disjoint validation set; the `val_best` row is the honest winner.\n"
    )

    # Summary table.
    lines.append("## Summary (validation set)\n")
    lines.append("| Channel | J_base_val | J_best_val | Δ% | transfer_gap |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        j_base = r["baseline_J_val"]
        j_best = r["val_best_J"]
        delta_pct = 100.0 * (j_best - j_base) / max(abs(j_base), 1e-9)
        # Transfer gap = (J_val_best - J_tune_best) / J_tune_best.
        # When pruning kicks the tune-best params off the top-3, val_best may
        # be a different parameter set; we still report on val_best vs its own
        # tune J.
        j_tune_match = next(
            (vr["J_tune"] for vr in r["val_results_top3"] if vr.get("J_val") == j_best),
            r["best_J_tune"],
        )
        gap = (j_best - j_tune_match) / max(abs(j_tune_match), 1e-9)
        flag = " ⚠" if gap > 0.30 else ""
        lines.append(f"| {r['channel']} | {j_base:.4f} | {j_best:.4f} | {delta_pct:+.1f}% | {gap:+.2f}{flag} |")
    lines.append("")

    # Per-channel detail.
    for r in results:
        ch = r["channel"]
        lines.append(f"## {ch}\n")
        lines.append(
            f"- Trials: {r['n_trials_completed']} completed, {r['n_trials_pruned']} pruned "
            f"({r['elapsed_s']:.0f}s).\n"
            f"- J_base_tune={r['baseline_J_tune']:.4f}  J_best_tune={r['best_J_tune']:.4f}\n"
            f"- J_base_val={r['baseline_J_val']:.4f}  J_best_val={r['val_best_J']:.4f}\n"
        )
        lines.append("\n**Parameter change (baseline -> val-best):**\n")
        lines.append(_format_params_diff(_baseline_params(ch), r["val_best_params"], channel=ch))
        lines.append("\n\n**Per-phase metrics (validation set):**\n")
        lines.append(_format_phase_table(r["baseline_phases_val"], r["val_best_phases"]))
        lines.append("\n")

        # Recommendation hint.
        delta_J = r["baseline_J_val"] - r["val_best_J"]
        if delta_J > 0.02:
            lines.append(
                f"**Recommendation:** ship the val-best params for `{ch}` — "
                f"J drops {delta_J:.3f} on the held-out 100-flight set.\n"
            )
        elif delta_J > 0.0:
            lines.append(
                f"**Recommendation:** marginal gain on `{ch}` (ΔJ={delta_J:.3f}); "
                "keep defaults unless one of the per-phase rates improves a target you care about.\n"
            )
        else:
            lines.append(
                f"**Recommendation:** tuning did not beat defaults for `{ch}` "
                f"(ΔJ={delta_J:.3f}). Leave production config alone.\n"
            )
        lines.append("")

    lines.append("## Caveats\n")
    lines.append(
        f"- Tuning + validation sets are disjoint ({n_tune} flights each, A320 only).\n"
        "- Optuna budget per channel limited to ~50 trials with MedianPruner; "
        "deeper sweeps may find better optima.\n"
        "- `min_abs_value` on `vz` / `alt` is in the search space but the "
        "`bilateral_vz` plateau detector does **not** consume it "
        "(see ``_VZ_BILATERAL_KEYS`` in `segments.py`). The 75-ftmin floor is dead in "
        "bilateral mode — finding confirmed by code path inspection.\n"
        "- Transfer gap >30% (flagged ⚠) means the tuning-best overfit; the val-best "
        "is reported instead and the gap is computed against it.\n"
        "- Phase `level` here is defined trajectory-side as `|truth - current| ≤ tol` "
        "(not just FMA mode), so it pools cruise hold + intermediate-step holds.\n"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"[done] markdown report: {report_path}", file=sys.stderr)


# -------------------------------------------------------------------- main --
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tuning-flights", type=int, default=100)
    parser.add_argument("--validation-flights", type=int, default=100)
    parser.add_argument("--channels", type=str, default="mach,cas,vz,alt,gamma",
                        help="Comma-separated channel subset.")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per channel.")
    parser.add_argument("--report", type=Path,
                        default=Path("/Users/gabriel/axm/04-papers/PS_MODEL/PAPER_0_DATA_PREPROCESSING/experiments/03_qar_target_finetuning.md"))
    parser.add_argument("--results-json", type=Path, default=RESULTS_JSON)
    parser.add_argument("--no-cache-rebuild", action="store_true",
                        help="Skip the (qar→raw→downsample) cache step if all files already exist.")
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    for c in channels:
        if c not in CHANNELS:
            print(f"Unknown channel: {c} (valid: {CHANNELS})", file=sys.stderr)
            return 2

    n_total = args.tuning_flights + args.validation_flights
    paths = _cohort_paths(n_total)
    cache_all = _build_cache(paths)

    # Map back to original ordering for tune/val split.
    cache_map = {p.stem: p for p in cache_all}
    ordered_cache = [cache_map[p.stem] for p in paths if p.stem in cache_map]
    tune_cache = ordered_cache[: args.tuning_flights]
    val_cache = ordered_cache[args.tuning_flights : args.tuning_flights + args.validation_flights]
    print(f"[split] tune={len(tune_cache)}  val={len(val_cache)}", file=sys.stderr)

    results: list[dict[str, Any]] = []
    for ch in channels:
        try:
            r = _tune_channel(ch, tune_cache, val_cache, args.trials)
            results.append(r)
        except Exception as exc:
            print(f"[FAIL] channel {ch}: {exc}\n{traceback.format_exc()}", file=sys.stderr)
            results.append({
                "channel": ch,
                "error": str(exc),
                "baseline_J_val": float("nan"),
                "val_best_J": float("nan"),
                "baseline_phases_val": {},
                "val_best_phases": {},
                "val_best_params": _baseline_params(ch),
                "baseline_J_tune": float("nan"),
                "best_J_tune": float("nan"),
                "n_trials_completed": 0,
                "n_trials_pruned": 0,
                "elapsed_s": 0.0,
                "val_results_top3": [],
                "baseline_extras_tune": {},
                "baseline_extras_val": {},
            })
        # Persist incremental progress.
        args.results_json.write_text(json.dumps(_serialize(results), indent=2, default=str))
        write_markdown(results, args.report, n_tune=args.tuning_flights, n_val=args.validation_flights)

    print(f"[done] all results: {args.results_json}", file=sys.stderr)
    return 0


def _serialize(obj: Any) -> Any:
    """Convert numpy types so json.dumps doesn't choke."""
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj


if __name__ == "__main__":
    sys.exit(main())
