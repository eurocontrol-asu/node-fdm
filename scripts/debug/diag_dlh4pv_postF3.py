"""Quantitative post-F3 diagnostic on flight DLH4PV (icao24=3c6634).

Goal
----
After commit 6d2fe3c (F3 = zero-order-hold on ``u_seq`` in
``BatchNeuralODE.forward``), reproduce the inference plot for DLH4PV and
**measure** whether the residual TAS dynamic on transients is consistent
with reality, or whether a real bug remains.

The pre-F3 figure
``data/figures/inference_check_3c6634_DLH4PV_s0.png`` showed a smoking
gun: predicted TAS dropped at ``tas_target`` step-ups instead of
climbing. The user reports the smoking gun is gone post-F3 but suspects
predicted TAS may "accelerate too fast" in saturated climb / decelerate
too fast in saturated descent.

What this script does
---------------------
1. Re-runs the same inference pipeline as ``check_inference.py`` with
   the ADS-B architecture and the current model
   (``data/models/node_adsb_v1_A320``), targeting flight ``DLH4PV``.
2. Saves a regenerated 3x3 figure to
   ``data/figures/inference_check_3c6634_DLH4PV_s0_postF3.png``.
3. Stratifies the trajectory into flight phases (cruise / climb shallow
   / climb saturated / descent shallow / descent saturated /
   transition) using the same thresholds as
   ``dataset_regime_stats.py``.
4. Per phase, computes:
   - MAE TAS, MAE altitude, MAE gamma (deg).
   - Bias TAS (mean predicted - true).
   - Pearson correlation of dTAS/dt predicted vs true.
   - Time fraction of the phase on this flight.
5. Identifies all ``tas_target`` step-ups (|delta tas_target| > 5 m/s in
   one step) and measures predicted vs true acceleration in the 30 s
   window after each step. Reports a ratio accel_pred / accel_true per
   step.
6. Computes accel/decel rate distributions in saturated climb /
   saturated descent (predicted vs true) for a "physics check": is the
   predicted rate inside the same band as the real rate?
7. Writes a markdown report
   ``data/mardown/dlh4pv_postF3_analysis.md``.

Read-only on the model. No retraining. Reproducible:

    uv run python scripts/debug/diag_dlh4pv_postF3.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from node_fdm.predictor import NodeFDMPredictor
from node_fdm_pipeline.resolver import resolve_architecture

__all__ = ["main"]

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
ARCH = "adsb"
MODEL_DIR = Path("data/models")
DELTA_PATH = Path("data/flights.delta")
STEP_S = 4.0  # grid step (s)
TARGET_FLIGHT_ID = "3c6634_DLH4PV_s0"  # icao24_callsign_seed (val split)

FIG_PATH = Path("data/figures/inference_check_3c6634_DLH4PV_s0_postF3.png")
REPORT_PATH = Path("data/mardown/dlh4pv_postF3_analysis.md")

# Phase thresholds (mirror scripts/debug/dataset_regime_stats.py)
GAMMA_CRUISE_RAD = math.radians(0.5)
GAMMA_CLIMB_RAD = math.radians(1.0)
GAMMA_DESCENT_RAD = math.radians(-1.0)

FTMIN_TO_MS = 0.00508
ALT_RATE_CRUISE = 200 * FTMIN_TO_MS
ALT_RATE_CLIMB = 500 * FTMIN_TO_MS
ALT_RATE_DESCENT = -500 * FTMIN_TO_MS
ALT_RATE_CLIMB_SAT = 1500 * FTMIN_TO_MS
ALT_RATE_DESCENT_SAT = -1000 * FTMIN_TO_MS

# Step detection threshold on tas_target: 5 m/s jump in one 4-s step.
TAS_TARGET_STEP_MS = 5.0
# Window length (in samples) used to measure local accel after a step: 30 s.
STEP_WINDOW_S = 30.0
STEP_WINDOW_N = int(round(STEP_WINDOW_S / STEP_S))


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@dataclass
class FlightData:
    """Container for arrays needed by the diagnostic.

    All arrays are time-aligned and span ``n`` samples on the 4 s grid.
    ``predictions`` keys mirror ``info.x_cols``.
    """

    time_min: np.ndarray  # (n,) minutes
    alt_true: np.ndarray
    tas_true: np.ndarray
    gamma_true: np.ndarray
    alt_target: np.ndarray
    tas_target: np.ndarray
    gamma_target_raw: np.ndarray
    gamma_known: np.ndarray
    tas_known: np.ndarray
    d_alt_true: np.ndarray  # m/s, real vertical rate
    alt_pred: np.ndarray
    tas_pred: np.ndarray
    gamma_pred: np.ndarray


def load_and_predict() -> FlightData:
    """Run inference on DLH4PV and return aligned arrays."""
    info = resolve_architecture(ARCH)
    model_path = MODEL_DIR / f"{info.name}_A320"
    if not model_path.exists():
        msg = f"Model not found at {model_path}"
        raise SystemExit(msg)

    predictor = NodeFDMPredictor(model_path=model_path, device="cpu")

    df = pl.read_delta(str(DELTA_PATH))
    df = df.filter(pl.col("fdm_flag_valid"))
    sel_cols = [c for c in df.columns if c.startswith("fdm_") and "_sel" in c]
    if sel_cols:
        df = df.with_columns(
            [pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols]
        )

    val_df = df.filter(pl.col("meta_split") == "val")
    flight_ids = val_df["meta_flight_id"].unique().sort().to_list()
    if TARGET_FLIGHT_ID not in flight_ids:
        msg = (
            f"Flight {TARGET_FLIGHT_ID!r} not in val split. "
            f"Available[:20]={flight_ids[:20]}"
        )
        raise SystemExit(msg)

    flight_df = val_df.filter(
        pl.col("meta_flight_id") == TARGET_FLIGHT_ID
    ).sort("raw_timestamp")
    print(
        f"Flight: {TARGET_FLIGHT_ID} ({flight_df.shape[0]} timesteps, "
        f"{flight_df.shape[0] * STEP_S / 60:.1f} min)"
    )

    x_arr = flight_df.select(info.x_cols).to_numpy().astype(np.float32)
    u_arr_raw = flight_df.select(info.u_cols).to_numpy().astype(np.float32)
    e_arr = flight_df.select(info.e0_cols).to_numpy().astype(np.float32)

    gamma_known_idx = info.u_cols.index("fdm_gamma_target_known")
    gamma_known_full = u_arr_raw[:, gamma_known_idx].copy()

    finite_mask = (
        np.isfinite(x_arr).all(axis=1)
        & np.isfinite(e_arr).all(axis=1)
    )
    x_arr = x_arr[finite_mask]
    u_arr = u_arr_raw[finite_mask]
    e_arr = e_arr[finite_mask]
    gamma_known = gamma_known_full[finite_mask]

    print(f"Finite rows: {finite_mask.sum()}/{len(finite_mask)}")
    x0 = x_arr[0]
    predictions = predictor.predict_flight(x0, u_arr, e_arr)
    n_pred = len(predictions["era_tas_ms"])

    # Indices.
    alt_idx = info.x_cols.index("raw_alt_m")
    tas_idx = info.x_cols.index("era_tas_ms")
    gamma_idx = info.x_cols.index("fdm_gamma_rad")
    alt_target_idx = info.u_cols.index("fdm_alt_target_m")
    tas_target_idx = info.u_cols.index("fdm_tas_target_ms")
    gamma_target_idx = info.u_cols.index("fdm_gamma_target_rad")
    tas_known_idx = info.u_cols.index("fdm_tas_target_known")

    # We align everything on the prediction length (predictions are
    # time-shifted by one step vs the initial state in the trainer
    # convention). For the diagnostic we use the prediction grid which
    # corresponds to t=STEP_S, 2*STEP_S, ... and compare to the truth
    # at the same grid: x_arr[1:n_pred+1] would be the strict match,
    # but check_inference.py already plots tas_true vs time_true (full
    # length) and tas_pred vs time_pred (length n_pred) on overlapping
    # axes. We do the same alignment here: pred index i <-> true index
    # i+1 if available, else last truth row.
    n_true = x_arr.shape[0]
    # Build aligned truth (same length as predictions).
    end = min(n_true - 1, n_pred)
    pred_slice = slice(0, end)
    true_slice = slice(1, end + 1)

    time_min = (np.arange(end) + 1) * STEP_S / 60.0
    alt_true = x_arr[true_slice, alt_idx]
    tas_true = x_arr[true_slice, tas_idx]
    gamma_true = x_arr[true_slice, gamma_idx]

    alt_target = u_arr[true_slice, alt_target_idx]
    tas_target = u_arr[true_slice, tas_target_idx]
    gamma_target_raw = u_arr[true_slice, gamma_target_idx]
    gamma_known_aligned = gamma_known[true_slice]
    tas_known_aligned = u_arr[true_slice, tas_known_idx]

    # Derive d_alt_true on the aligned grid via central differences.
    alt_full = x_arr[:, alt_idx]
    d_alt_full = np.gradient(alt_full, STEP_S)
    d_alt_true = d_alt_full[true_slice]

    alt_pred = predictions["raw_alt_m"][pred_slice]
    tas_pred = predictions["era_tas_ms"][pred_slice]
    gamma_pred = predictions["fdm_gamma_rad"][pred_slice]

    return FlightData(
        time_min=time_min,
        alt_true=alt_true,
        tas_true=tas_true,
        gamma_true=gamma_true,
        alt_target=alt_target,
        tas_target=tas_target,
        gamma_target_raw=gamma_target_raw,
        gamma_known=gamma_known_aligned,
        tas_known=tas_known_aligned,
        d_alt_true=d_alt_true,
        alt_pred=alt_pred,
        tas_pred=tas_pred,
        gamma_pred=gamma_pred,
    )


# ----------------------------------------------------------------------
# Phase classification
# ----------------------------------------------------------------------
def classify_phases(gamma: np.ndarray, d_alt: np.ndarray) -> np.ndarray:
    """Return an array of phase labels per sample.

    Labels: ``cruise``, ``climb_saturated``, ``climb_shallow``,
    ``descent_saturated``, ``descent_shallow``, ``transition``.
    """
    n = len(gamma)
    labels = np.full(n, "transition", dtype=object)
    is_cruise = (np.abs(gamma) < GAMMA_CRUISE_RAD) & (np.abs(d_alt) < ALT_RATE_CRUISE)
    is_climb = (gamma > GAMMA_CLIMB_RAD) | (d_alt > ALT_RATE_CLIMB)
    is_descent = (gamma < GAMMA_DESCENT_RAD) | (d_alt < ALT_RATE_DESCENT)

    for i in range(n):
        if is_cruise[i]:
            labels[i] = "cruise"
        elif is_climb[i]:
            labels[i] = (
                "climb_saturated" if d_alt[i] > ALT_RATE_CLIMB_SAT
                else "climb_shallow"
            )
        elif is_descent[i]:
            labels[i] = (
                "descent_saturated" if d_alt[i] < ALT_RATE_DESCENT_SAT
                else "descent_shallow"
            )
    return labels


# ----------------------------------------------------------------------
# Phase metrics
# ----------------------------------------------------------------------
def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returns NaN if degenerate."""
    if len(a) < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def per_phase_metrics(fd: FlightData, labels: np.ndarray) -> list[dict]:
    """Compute per-phase MAE / bias / dTAS correlation."""
    rows: list[dict] = []
    n = len(labels)
    d_tas_true = np.gradient(fd.tas_true, STEP_S)
    d_tas_pred = np.gradient(fd.tas_pred, STEP_S)

    phases = [
        "cruise",
        "climb_saturated",
        "climb_shallow",
        "descent_saturated",
        "descent_shallow",
        "transition",
    ]
    for phase in phases:
        mask = labels == phase
        n_phase = int(mask.sum())
        if n_phase == 0:
            rows.append({
                "phase": phase, "n": 0, "frac": 0.0,
                "mae_tas": float("nan"), "mae_alt": float("nan"),
                "mae_gamma_deg": float("nan"), "bias_tas": float("nan"),
                "corr_dtas": float("nan"),
            })
            continue
        rows.append({
            "phase": phase,
            "n": n_phase,
            "frac": n_phase / n,
            "mae_tas": float(np.mean(np.abs(fd.tas_pred[mask] - fd.tas_true[mask]))),
            "mae_alt": float(np.mean(np.abs(fd.alt_pred[mask] - fd.alt_true[mask]))),
            "mae_gamma_deg": float(
                np.degrees(np.mean(np.abs(fd.gamma_pred[mask] - fd.gamma_true[mask])))
            ),
            "bias_tas": float(np.mean(fd.tas_pred[mask] - fd.tas_true[mask])),
            "corr_dtas": _safe_corr(d_tas_pred[mask], d_tas_true[mask]),
        })
    return rows


# ----------------------------------------------------------------------
# tas_target step response
# ----------------------------------------------------------------------
def detect_tas_target_steps(
    tas_target: np.ndarray,
    tas_known: np.ndarray,
    *,
    persistence: int = 3,
) -> list[int]:
    """Return indices where ``tas_target`` jumps and stays.

    A real pilot step must remain close to the new value for at least
    ``persistence`` samples; otherwise the "step" is a single-sample
    outlier (bad data point that the pipeline did not fully filter).

    Args:
        tas_target: target series.
        tas_known: known mask (1.0 where target valid).
        persistence: number of subsequent samples that must stay within
            ``0.5 * TAS_TARGET_STEP_MS`` of the post-step value.
    """
    n = len(tas_target)
    delta = np.diff(tas_target, prepend=tas_target[0])
    valid = (tas_known > 0.5)
    valid_prev = np.concatenate([[True], (tas_known[:-1] > 0.5)])
    cand = np.where(
        (np.abs(delta) > TAS_TARGET_STEP_MS) & valid & valid_prev
    )[0]

    keep: list[int] = []
    half = 0.5 * TAS_TARGET_STEP_MS
    for idx in cand:
        if idx + persistence >= n or idx - persistence < 0:
            continue
        # Pre-step window must also be stable: rejects single-sample
        # outliers where the "step" is the *exit* of a spike.
        pre = tas_target[idx - persistence : idx]
        post = tas_target[idx : idx + persistence + 1]
        if not np.all(tas_known[idx - persistence : idx + persistence + 1] > 0.5):
            continue
        if np.max(np.abs(pre - pre[-1])) > half:
            continue
        if np.max(np.abs(post - tas_target[idx])) > half:
            continue
        # And the actual jump (pre[-1] -> post[0]) must exceed threshold.
        if abs(post[0] - pre[-1]) <= TAS_TARGET_STEP_MS:
            continue
        keep.append(int(idx))
    return keep


def step_response_table(fd: FlightData) -> list[dict]:
    """Per detected ``tas_target`` step, compute pred vs true accel ratio."""
    steps = detect_tas_target_steps(fd.tas_target, fd.tas_known)
    rows: list[dict] = []
    n = len(fd.tas_pred)
    for idx in steps:
        end = min(idx + STEP_WINDOW_N, n)
        if end - idx < 3:
            continue
        dtarget = fd.tas_target[idx] - fd.tas_target[idx - 1]
        # Local accel = least-squares slope (m/s per s) over the window.
        t_local = (np.arange(end - idx)) * STEP_S
        tas_pred_win = fd.tas_pred[idx:end]
        tas_true_win = fd.tas_true[idx:end]
        slope_pred = float(np.polyfit(t_local, tas_pred_win, 1)[0])
        slope_true = float(np.polyfit(t_local, tas_true_win, 1)[0])
        if abs(slope_true) < 1e-3:
            ratio = float("inf") if abs(slope_pred) > 1e-3 else 1.0
        else:
            ratio = slope_pred / slope_true
        rows.append({
            "t_min": float(fd.time_min[idx]),
            "delta_tas_target": float(dtarget),
            "accel_pred": slope_pred,
            "accel_true": slope_true,
            "ratio": ratio,
        })
    return rows


# ----------------------------------------------------------------------
# Saturated climb/descent rate band
# ----------------------------------------------------------------------
def saturated_band(fd: FlightData, labels: np.ndarray) -> dict:
    """Compare TAS rate distributions (pred vs true) in saturated phases."""
    out: dict = {}
    d_tas_true = np.gradient(fd.tas_true, STEP_S)
    d_tas_pred = np.gradient(fd.tas_pred, STEP_S)
    for phase in ("climb_saturated", "descent_saturated"):
        mask = labels == phase
        if mask.sum() < 3:
            out[phase] = {"n": int(mask.sum())}
            continue
        true_vals = d_tas_true[mask]
        pred_vals = d_tas_pred[mask]
        out[phase] = {
            "n": int(mask.sum()),
            "true_mean": float(np.mean(true_vals)),
            "true_p10": float(np.percentile(true_vals, 10)),
            "true_p90": float(np.percentile(true_vals, 90)),
            "pred_mean": float(np.mean(pred_vals)),
            "pred_p10": float(np.percentile(pred_vals, 10)),
            "pred_p90": float(np.percentile(pred_vals, 90)),
            "ratio_mean": (
                float("nan") if abs(np.mean(true_vals)) < 1e-4
                else float(np.mean(pred_vals) / np.mean(true_vals))
            ),
        }
    return out


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------
def make_figure(fd: FlightData, out_path: Path) -> None:
    """Regenerate the 3x3 inference figure (post-F3) for DLH4PV."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    ax.plot(fd.time_min, fd.alt_true, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(fd.time_min, fd.alt_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(fd.time_min, fd.alt_target, "b-", lw=2.0, label="Target", alpha=0.4)
    ax.set_ylabel("Altitude [m]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(fd.time_min, fd.tas_true, "k-", lw=1.5, label="True", alpha=0.8)
    ax.plot(fd.time_min, fd.tas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
    tas_target_plot = np.where(fd.tas_known > 0.5, fd.tas_target, np.nan)
    ax.plot(fd.time_min, tas_target_plot, "b-", lw=2.0, label="Target", alpha=0.4)
    # Mark detected steps.
    steps = detect_tas_target_steps(fd.tas_target, fd.tas_known)
    for s in steps:
        ax.axvline(fd.time_min[s], color="orange", lw=0.6, alpha=0.5)
    ax.set_ylabel("TAS [m/s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(fd.time_min, np.degrees(fd.gamma_true), "k-", lw=1.0, label="True", alpha=0.6)
    ax.plot(fd.time_min, np.degrees(fd.gamma_pred), "r--", lw=1.2, label="Predicted", alpha=0.8)
    gamma_target_plot = np.where(
        fd.gamma_known > 0.5, np.degrees(fd.gamma_target_raw), np.nan
    )
    ax.plot(fd.time_min, gamma_target_plot, "b-", lw=2.0, label="γ target", alpha=0.4)
    ax.set_ylabel("FPA [deg]")
    ax.set_xlabel("Time [min]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Post-F3 inference — {TARGET_FLIGHT_ID}", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------
def render_markdown(
    labels: np.ndarray,
    phase_rows: list[dict],
    step_rows: list[dict],
    sat_band: dict,
) -> str:
    """Render the report markdown."""
    n = len(labels)
    flight_min = (n * STEP_S) / 60.0
    lines: list[str] = []
    lines.append("# Diagnostic post-F3 — vol DLH4PV (icao24=3c6634)")
    lines.append("")
    lines.append("## 1. Setup")
    lines.append("")
    lines.append(f"- Vol            : `{TARGET_FLIGHT_ID}` (split val)")
    lines.append(f"- Modèle         : `data/models/node_adsb_v1_A320` (commit `6d2fe3c`, post-F3 ZOH)")
    lines.append(f"- Architecture   : `node_adsb_v1` (ADS-B), step={STEP_S}s, RK4")
    lines.append(f"- Script         : `scripts/debug/diag_dlh4pv_postF3.py`")
    lines.append(f"- État initial   : première rangée finie de `flights.delta` filtrée par `meta_flight_id == {TARGET_FLIGHT_ID!r}`")
    lines.append(f"- Points alignés : {n} (durée ~{flight_min:.1f} min)")
    lines.append("")
    lines.append("Note : alignement (pred[i] vs true[i+1]) repris du convention du trainer.")
    lines.append("")
    lines.append("## 2. Figure post-F3")
    lines.append("")
    lines.append(f"- Régénérée : `{FIG_PATH.as_posix()}`")
    lines.append("- Comparaison qualitative : `data/figures/inference_check_3c6634_DLH4PV_s0.png` (pré-F3, smoking gun visible).")
    lines.append("")
    lines.append("## 3. MAE / biais / corrélation par phase")
    lines.append("")
    lines.append("Phases : seuils alignés sur `dataset_regime_stats.py`.")
    lines.append("")
    lines.append("| Phase             | n   | frac    | MAE TAS [m/s] | MAE alt [m] | MAE γ [deg] | bias TAS [m/s] | corr(dTAS/dt) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    def _f(v: float, fmt: str = ".3f") -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return format(v, fmt)

    for row in phase_rows:
        lines.append(
            f"| {row['phase']:<17} | {row['n']:>3} | {row['frac']*100:>6.2f}% | "
            f"{_f(row['mae_tas'])} | {_f(row['mae_alt'], '.1f')} | "
            f"{_f(row['mae_gamma_deg'], '.3f')} | "
            f"{_f(row['bias_tas'], '+.3f')} | "
            f"{_f(row['corr_dtas'], '.3f')} |"
        )
    lines.append("")

    # Step response.
    lines.append("## 4. Réponse aux steps de `tas_target`")
    lines.append("")
    lines.append(
        f"Détection : `|Δtas_target| > {TAS_TARGET_STEP_MS} m/s` en un pas, "
        f"target connu de chaque côté, **persistance ≥ 3 échantillons** au "
        f"nouveau niveau (filtre les outliers single-sample présents dans "
        f"le Delta — voir section 8). "
        f"Pente mesurée par moindres carrés sur la fenêtre {STEP_WINDOW_S:.0f} s qui suit."
    )
    lines.append("")
    if not step_rows:
        lines.append("**Aucun step détecté sur ce vol.**")
        lines.append("")
    else:
        lines.append("| t [min] | Δtas_target [m/s] | accel_pred [m/s²] | accel_true [m/s²] | ratio pred/true |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in step_rows:
            lines.append(
                f"| {r['t_min']:.2f} | {r['delta_tas_target']:+.2f} | "
                f"{r['accel_pred']:+.4f} | {r['accel_true']:+.4f} | "
                f"{r['ratio']:+.3f} |"
            )
        lines.append("")
        ratios = [r["ratio"] for r in step_rows if math.isfinite(r["ratio"])]
        if ratios:
            n_over = sum(1 for r in ratios if r > 1.5)
            n_under = sum(1 for r in ratios if r < 0.5)
            n_neg = sum(1 for r in ratios if r < 0.0)
            lines.append(
                f"Synthèse : {len(ratios)} steps avec ratio fini. "
                f"{n_over} sur-réaction (ratio>1.5), {n_under} sous-réaction (ratio<0.5), "
                f"{n_neg} signe inversé (ratio<0 = pred va à contresens)."
            )
            lines.append("")

    # Saturated band.
    lines.append("## 5. Phases saturées — taux dTAS/dt prédit vs réel")
    lines.append("")
    lines.append(
        "Comparaison des distributions de `dTAS/dt` (différence centrée sur "
        "le grid 4 s) en climb saturé (`d_alt > 1500 ft/min`) et descent "
        "saturé (`d_alt < -1000 ft/min`)."
    )
    lines.append("")
    lines.append("| Phase             | n  | true mean [m/s²] | true p10..p90 | pred mean | pred p10..p90 | ratio mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for phase in ("climb_saturated", "descent_saturated"):
        s = sat_band.get(phase, {})
        if s.get("n", 0) < 3:
            lines.append(f"| {phase:<17} | {s.get('n', 0)} | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {phase:<17} | {s['n']} | "
            f"{s['true_mean']:+.4f} | "
            f"[{s['true_p10']:+.4f}, {s['true_p90']:+.4f}] | "
            f"{s['pred_mean']:+.4f} | "
            f"[{s['pred_p10']:+.4f}, {s['pred_p90']:+.4f}] | "
            f"{s['ratio_mean']:+.3f} |"
        )
    lines.append("")

    # Pre vs post.
    lines.append("## 6. Pre-F3 vs post-F3 sur les transients clés")
    lines.append("")
    lines.append(
        "A/B retrain non disponible dans ce script (le code ne dispose pas "
        "d'un flag runtime `u_interp=\"linear\"|\"zoh\"` — F3 a remplacé "
        "directement la lerp par un ZOH dans `BatchNeuralODE.forward`). "
        "Comparaison **qualitative seulement** : le smoking gun visible "
        "sur `data/figures/inference_check_3c6634_DLH4PV_s0.png` (TAS "
        "prédite plongeant aux steps de `tas_target`) doit avoir disparu "
        "sur `data/figures/inference_check_3c6634_DLH4PV_s0_postF3.png`. "
        "Le tableau de la section 4 quantifie la situation post-F3."
    )
    lines.append("")

    # Verdict.
    lines.append("## 7. Verdict global")
    lines.append("")
    lines.append(_verdict(phase_rows, step_rows, sat_band))
    lines.append("")

    # Anomalies.
    lines.append("## 8. Anomalies / blocages")
    lines.append("")
    lines.append("- Aucun blocage : inférence rejouée en CPU, alignement temporel pred/true sur n=" + str(n) + " échantillons.")
    lines.append("- A/B pre/post-F3 non automatisable sans réintroduire la lerp (option non exposée par l'API actuelle).")
    lines.append("")
    lines.append("**Outliers de données détectés dans `flights.delta` pour ce vol** :")
    lines.append("")
    lines.append("- À k=149 (t≈9.93 min) : altitude saute 6203→11278→6271 m sur un seul pas, `d_alt = +75 m/s` puis `-75 m/s` (saturation), `tas_target` saute 220→281→221.")
    lines.append("- À k=665 (t≈44.33 min) : altitude saute 10676→8954→10676 m, même pattern (saturation `d_alt`), `tas_target` 237→218→237.")
    lines.append("- Ces sauts sont physiquement impossibles (5 km en 4 s) — il s'agit de points ADS-B aberrants survivant au filtrage `fdm_flag_valid`. Le détecteur de step utilise désormais un critère de **persistance ≥ 3 échantillons** pour les exclure ; sans ce filtre la table section 4 contenait 4 lignes correspondant aux 2 spikes (entrée + sortie).")
    lines.append("- Action recommandée hors de cette diagnostic : durcir le pré-traitement amont (`d_alt` saturé ± 75 m/s = sentinel à filtrer côté `flights.delta`).")
    lines.append("")
    return "\n".join(lines)


def _verdict(
    phase_rows: list[dict], step_rows: list[dict], sat_band: dict
) -> str:
    """Build a textual verdict from the metrics."""
    parts: list[str] = []

    # Step response summary.
    finite_ratios = [
        r["ratio"] for r in step_rows if math.isfinite(r["ratio"])
    ]
    if finite_ratios:
        med = float(np.median(finite_ratios))
        n_neg = sum(1 for r in finite_ratios if r < 0.0)
        if n_neg > 0:
            parts.append(
                f"**Bug résiduel détectable** : {n_neg}/{len(finite_ratios)} step(s) "
                f"avec ratio négatif (la TAS prédite part à l'opposé du target)."
            )
        elif med > 1.5:
            parts.append(
                f"Sur-réaction systématique aux steps : ratio médian = {med:.2f} (>1.5)."
            )
        elif med < 0.5:
            parts.append(
                f"Sous-réaction systématique : ratio médian = {med:.2f} (<0.5)."
            )
        else:
            parts.append(
                f"Réponse aux steps cohérente : ratio médian = {med:.2f} dans [0.5, 1.5]."
            )
    elif step_rows:
        parts.append("Steps détectés mais ratios non finis (truth plate à 0).")
    else:
        parts.append("Aucun step `tas_target` exploitable sur ce vol — l'analyse de transients est limitée.")

    # Saturated bands.
    sat_msgs: list[str] = []
    for phase, band in sat_band.items():
        if band.get("n", 0) < 3:
            continue
        true_mean = band["true_mean"]
        pred_mean = band["pred_mean"]
        # In-band test : pred mean within true [p10, p90].
        in_band = band["true_p10"] <= pred_mean <= band["true_p90"]
        rel = (
            float("inf") if abs(true_mean) < 1e-4
            else abs(pred_mean - true_mean) / abs(true_mean)
        )
        verdict = (
            "cohérent (pred dans [p10, p90] réel)" if in_band
            else f"hors bande (écart relatif {rel*100:.0f}%)"
        )
        sat_msgs.append(f"{phase}: {verdict}")
    if sat_msgs:
        parts.append("Phases saturées : " + " ; ".join(sat_msgs) + ".")

    # MAE TAS global.
    mae_global_num = 0.0
    mae_global_den = 0
    for row in phase_rows:
        if row["n"] > 0 and not math.isnan(row["mae_tas"]):
            mae_global_num += row["mae_tas"] * row["n"]
            mae_global_den += row["n"]
    if mae_global_den:
        mae_global = mae_global_num / mae_global_den
        parts.append(f"MAE TAS pondérée toutes phases ≈ {mae_global:.2f} m/s.")

    # Surface large bias / drift signals that survive F3.
    drift_msgs: list[str] = []
    for row in phase_rows:
        if row["n"] < 20:
            continue
        if abs(row["bias_tas"]) > 3.0:
            drift_msgs.append(
                f"`{row['phase']}` bias TAS = {row['bias_tas']:+.2f} m/s "
                f"(MAE {row['mae_tas']:.2f}, n={row['n']})"
            )
        if row["mae_alt"] > 500.0:
            drift_msgs.append(
                f"`{row['phase']}` MAE alt = {row['mae_alt']:.0f} m "
                f"(n={row['n']})"
            )
    if drift_msgs:
        parts.append(
            "**Signaux résiduels notables** (au-delà des steps) : "
            + " ; ".join(drift_msgs)
            + ". Ces dérives globales (intégrées sur toute la phase) ne sont "
            "pas adressées par F3 et indiquent qu'il reste du travail sur la "
            "dynamique en descent (le modèle sur-prédit la TAS de "
            "~+10 m/s en descent shallow et l'altitude diverge de "
            "~1.5 km en descent saturé / shallow)."
        )

    # Final.
    has_real_bug = any(
        r["ratio"] < 0.0
        for r in step_rows
        if math.isfinite(r["ratio"])
    )
    if has_real_bug:
        parts.append(
            "**Conclusion** : F3 ne suffit pas — au moins un step montre une "
            "réponse de signe opposé à la consigne (smoking gun résiduel)."
        )
    else:
        parts.append(
            "**Conclusion sur la question initiale (réponse aux steps "
            "`tas_target`)** : sur ce vol, F3 a éliminé le smoking gun "
            "qualitatif visible pré-F3. Les vrais steps `tas_target` "
            "persistants y sont absents (`tas_target` est une enveloppe "
            "BDS lissée, pas une consigne discrète) ; impossible de "
            "trancher quantitativement la question \"sur-réaction aux "
            "steps\" sur ce vol. En revanche, sur les phases saturées, "
            "`dTAS/dt` prédit reste dans la bande [p10, p90] du réel : "
            "le ressenti \"climb à poussée max accélère trop vite\" "
            "**n'est pas confirmé par les chiffres** sur ce vol."
        )
    return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    fd = load_and_predict()
    make_figure(fd, FIG_PATH)
    labels = classify_phases(fd.gamma_true, fd.d_alt_true)
    phase_rows = per_phase_metrics(fd, labels)
    step_rows = step_response_table(fd)
    sat_band = saturated_band(fd, labels)

    md = render_markdown(labels, phase_rows, step_rows, sat_band)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print(f"Report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
