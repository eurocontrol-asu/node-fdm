"""Round 2C falsifier — gamma as primary confounder of TAS bias.

Hypothesis (H_2C) under test
----------------------------
Le bias TAS de +19.73 m/s sur descent_shallow `tas_known=0` est un effet
dérivé d'une erreur primaire sur γ (gamma_pred − gamma_true) qui se
propage via la dynamique couplée du Neural ODE. Les samples `known=0`
ont un MAE_γ et un bias_γ significativement plus mauvais que `known=1`,
et l'erreur TAS est corrélée à l'erreur γ sample-par-sample.

Falsifier
---------
- Statistic primaire : ratio MAE_γ(known=0) / MAE_γ(known=1) sur
  descent_shallow agrégé 5 vols, ET corrélation Pearson entre bias_TAS
  et bias_gamma sur les samples `known=0` descent_shallow.
- Threshold          : H_2C survives si ratio MAE_γ > 1.5 ET |Pearson| > 0.5.
                       Dies si ratio < 1.2 ET |Pearson| < 0.2. Indéterminé sinon.
- Null world         : si H_2C est fausse, MAE_γ similaire entre stratums et
                       bias_TAS / bias_γ indépendants.
- Data slice         : val seed 0, A320, 5 vols, descent_shallow, rollout
                       d'inférence baseline (pas de masquage).
- Strata             : par stratum tas_known. Aussi reporter par-vol.

Method
------
For each of the 5 flights:
  1. Load trajectory from `data/flights.delta` (val split, A320).
  2. Run `NodeFDMPredictor.predict_flight` once (baseline rollout, no
     masking — exactly like Round 1 baseline).
  3. Classify each step into a regime via the same scheme as
     `dataset_regime_stats.py`.
  4. Restrict to descent_shallow samples; stratify by `tas_known` (0/1).
  5. Compute bias_TAS = pred − true (m/s), bias_gamma = pred − true (deg),
     bias_alt = pred − true (m); aggregate MAE / mean per stratum.
  6. Compute ratio MAE_γ(0)/MAE_γ(1), ratio MAE_alt(0)/MAE_alt(1).
  7. Pearson correlations bias_TAS↔bias_γ and bias_TAS↔bias_alt on
     known=0 (and same on known=1 for contrast).

Output
------
- Markdown report : ``data/mardown/round2c_gamma_confounder.md``

Usage
-----
    uv run python scripts/debug/diag_round2c_gamma_confounder.py
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

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
STEP_S = 4.0
REPORT_PATH = Path("data/mardown/round2c_gamma_confounder.md")

# The 5 flights aggregated in `data/mardown/target_known_diagnosis.md`.
TARGET_FLIGHTS: list[str] = [
    "3c6634_DLH4PV_s0",
    "0aca66_AVA8443_s0",
    "0aca66_AVA8447_s0",
    "0aca66_AVA8548_s0",
    "0aca66_AVA8558_s0",
]

# Phase thresholds — mirror `dataset_regime_stats.py`.
GAMMA_CRUISE_RAD = math.radians(0.5)
GAMMA_CLIMB_RAD = math.radians(1.0)
GAMMA_DESCENT_RAD = math.radians(-1.0)

FTMIN_TO_MS = 0.00508
ALT_RATE_CRUISE = 200 * FTMIN_TO_MS
ALT_RATE_CLIMB = 500 * FTMIN_TO_MS
ALT_RATE_DESCENT = -500 * FTMIN_TO_MS
ALT_RATE_CLIMB_SAT = 1500 * FTMIN_TO_MS
ALT_RATE_DESCENT_SAT = -1000 * FTMIN_TO_MS


# ----------------------------------------------------------------------
# Falsifier echo
# ----------------------------------------------------------------------
def _print_falsifier_header() -> None:
    print("=" * 72)
    print("FALSIFIER HEADER (round 2C — gamma confounder)")
    print("=" * 72)
    print(
        "H_2C: le bias TAS de +19.73 m/s sur descent_shallow tas_known=0"
        " est un effet derive d'une erreur primaire sur gamma"
        " (gamma_pred - gamma_true) propagee via la dynamique couplee"
        " du Neural ODE."
    )
    print(
        "STATISTIC: ratio MAE_gamma(known=0) / MAE_gamma(known=1) sur"
        " descent_shallow agrege 5 vols, ET Pearson(bias_TAS, bias_gamma)"
        " sur samples known=0."
    )
    print(
        "THRESHOLD: H_2C survives si ratio MAE_gamma > 1.5 ET |Pearson|"
        " > 0.5. Dies si ratio < 1.2 ET |Pearson| < 0.2. Indetermine"
        " sinon."
    )
    print(
        "NULL WORLD: si H_2C fausse, MAE_gamma similaire entre stratums"
        " et bias_TAS / bias_gamma independants."
    )
    print(
        "DATA SLICE: val seed 0, A320, descent_shallow, rollout baseline"
        f" sans masquage. Flights: {TARGET_FLIGHTS}."
    )
    print("STRATA: tas_known in {0, 1}; reporte aussi par-vol.")
    print("=" * 72)


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class FlightArrays:
    flight_id: str
    x_arr: np.ndarray  # (n, |X|)
    u_arr: np.ndarray  # (n, |U|)
    e_arr: np.ndarray  # (n, |E0|)
    tas_known_orig: np.ndarray  # (n,) 0/1


@dataclass
class FlightSamples:
    """Aligned samples for one flight in descent_shallow regime."""

    flight_id: str
    # All arrays aligned on descent_shallow mask
    bias_tas: np.ndarray
    bias_gamma_deg: np.ndarray
    bias_alt: np.ndarray
    tas_known: np.ndarray  # 0/1 on aligned grid


# ----------------------------------------------------------------------
# Phase classification (mirror dataset_regime_stats.py)
# ----------------------------------------------------------------------
def _classify_phases_np(gamma: np.ndarray, d_alt: np.ndarray) -> np.ndarray:
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
# Per-flight loading + inference
# ----------------------------------------------------------------------
def _build_flight_arrays(
    info,
    val_df: pl.DataFrame,
    flight_id: str,
) -> FlightArrays | None:
    flight_df = val_df.filter(pl.col("meta_flight_id") == flight_id)
    sel_cols = [c for c in flight_df.columns if c.startswith("fdm_") and "_sel" in c]
    if sel_cols:
        flight_df = flight_df.with_columns(
            [pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols]
        )
    flight_df = flight_df.sort("raw_timestamp")

    x_arr = flight_df.select(info.x_cols).to_numpy().astype(np.float32)
    u_arr = flight_df.select(info.u_cols).to_numpy().astype(np.float32)
    e_arr = flight_df.select(info.e0_cols).to_numpy().astype(np.float32)

    finite_mask = (
        np.isfinite(x_arr).all(axis=1)
        & np.isfinite(e_arr).all(axis=1)
    )
    x_arr = x_arr[finite_mask]
    u_arr = u_arr[finite_mask]
    e_arr = e_arr[finite_mask]

    if x_arr.shape[0] < 5:
        return None

    tas_known_idx = info.u_cols.index("fdm_tas_target_known")
    u_arr = np.where(np.isnan(u_arr), 0.0, u_arr)
    tas_known_orig = u_arr[:, tas_known_idx].copy()

    return FlightArrays(
        flight_id=flight_id,
        x_arr=x_arr,
        u_arr=u_arr,
        e_arr=e_arr,
        tas_known_orig=tas_known_orig,
    )


def _run_baseline_inference(
    predictor: NodeFDMPredictor,
    info,
    fa: FlightArrays,
) -> FlightSamples | None:
    x_arr = fa.x_arr
    n_true = x_arr.shape[0]
    x0 = x_arr[0]

    pred = predictor.predict_flight(x0, fa.u_arr, fa.e_arr)

    n_pred = len(pred["era_tas_ms"])
    end = min(n_true - 1, n_pred)
    if end <= 1:
        return None
    pred_slice = slice(0, end)
    true_slice = slice(1, end + 1)

    alt_idx = info.x_cols.index("raw_alt_m")
    tas_idx = info.x_cols.index("era_tas_ms")
    gamma_idx = info.x_cols.index("fdm_gamma_rad")

    alt_full = x_arr[:, alt_idx]
    d_alt_full = np.gradient(alt_full, STEP_S)

    tas_true = x_arr[true_slice, tas_idx]
    alt_true = x_arr[true_slice, alt_idx]
    gamma_true = x_arr[true_slice, gamma_idx]
    d_alt_true = d_alt_full[true_slice]

    tas_pred = pred["era_tas_ms"][pred_slice]
    alt_pred = pred["raw_alt_m"][pred_slice]
    gamma_pred = pred["fdm_gamma_rad"][pred_slice]

    tas_known_aligned = fa.tas_known_orig[true_slice]

    # Classify regimes using the *true* gamma + d_alt (consistent with
    # dataset_regime_stats.py and the Round 1 protocol).
    labels = _classify_phases_np(gamma_true, d_alt_true)
    desc_mask = labels == "descent_shallow"

    if not desc_mask.any():
        return FlightSamples(
            flight_id=fa.flight_id,
            bias_tas=np.array([], dtype=np.float64),
            bias_gamma_deg=np.array([], dtype=np.float64),
            bias_alt=np.array([], dtype=np.float64),
            tas_known=np.array([], dtype=np.float64),
        )

    bias_tas = (tas_pred[desc_mask] - tas_true[desc_mask]).astype(np.float64)
    # gamma in degrees for readability
    bias_gamma_deg = (
        np.degrees(gamma_pred[desc_mask] - gamma_true[desc_mask])
    ).astype(np.float64)
    bias_alt = (alt_pred[desc_mask] - alt_true[desc_mask]).astype(np.float64)
    known = tas_known_aligned[desc_mask].astype(np.float64)

    return FlightSamples(
        flight_id=fa.flight_id,
        bias_tas=bias_tas,
        bias_gamma_deg=bias_gamma_deg,
        bias_alt=bias_alt,
        tas_known=known,
    )


# ----------------------------------------------------------------------
# Stats helpers
# ----------------------------------------------------------------------
def _safe_mean(a: np.ndarray) -> float:
    return float(np.mean(a)) if a.size > 0 else float("nan")


def _safe_mae(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a))) if a.size > 0 else float("nan")


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_ratio(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b) or abs(b) < 1e-12:
        return float("nan")
    return a / b


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------
def _f(v: float, fmt: str = "+.4f") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return format(v, fmt)


def render_markdown(
    per_flight_samples: list[FlightSamples],
    used_flights: list[str],
    skipped: list[str],
) -> str:
    # Aggregate concat
    all_bias_tas = (
        np.concatenate([s.bias_tas for s in per_flight_samples])
        if per_flight_samples else np.array([])
    )
    all_bias_gamma = (
        np.concatenate([s.bias_gamma_deg for s in per_flight_samples])
        if per_flight_samples else np.array([])
    )
    all_bias_alt = (
        np.concatenate([s.bias_alt for s in per_flight_samples])
        if per_flight_samples else np.array([])
    )
    all_known = (
        np.concatenate([s.tas_known for s in per_flight_samples])
        if per_flight_samples else np.array([])
    )

    mask0 = all_known < 0.5
    mask1 = all_known > 0.5

    n0 = int(mask0.sum())
    n1 = int(mask1.sum())

    bias_tas_0 = all_bias_tas[mask0]
    bias_tas_1 = all_bias_tas[mask1]
    bias_g_0 = all_bias_gamma[mask0]
    bias_g_1 = all_bias_gamma[mask1]
    bias_a_0 = all_bias_alt[mask0]
    bias_a_1 = all_bias_alt[mask1]

    mean_tas_0 = _safe_mean(bias_tas_0)
    mae_tas_0 = _safe_mae(bias_tas_0)
    mean_tas_1 = _safe_mean(bias_tas_1)
    mae_tas_1 = _safe_mae(bias_tas_1)
    mean_g_0 = _safe_mean(bias_g_0)
    mae_g_0 = _safe_mae(bias_g_0)
    mean_g_1 = _safe_mean(bias_g_1)
    mae_g_1 = _safe_mae(bias_g_1)
    mean_a_0 = _safe_mean(bias_a_0)
    mae_a_0 = _safe_mae(bias_a_0)
    mean_a_1 = _safe_mean(bias_a_1)
    mae_a_1 = _safe_mae(bias_a_1)

    ratio_tas = _safe_ratio(mae_tas_0, mae_tas_1)
    ratio_g = _safe_ratio(mae_g_0, mae_g_1)
    ratio_a = _safe_ratio(mae_a_0, mae_a_1)

    # Correlations on known=0
    corr_tas_g_0 = _safe_pearson(bias_tas_0, bias_g_0)
    corr_tas_a_0 = _safe_pearson(bias_tas_0, bias_a_0)
    # Correlations on known=1
    corr_tas_g_1 = _safe_pearson(bias_tas_1, bias_g_1)
    corr_tas_a_1 = _safe_pearson(bias_tas_1, bias_a_1)

    lines: list[str] = []
    lines.append("# Round 2C — Gamma confounder")
    lines.append("")
    lines.append("## Hypothesis under test")
    lines.append("")
    lines.append(
        "Le bias TAS de +19.73 m/s sur descent_shallow `tas_known=0` est un"
        " effet dérivé d'une erreur primaire sur γ (gamma_pred − gamma_true)"
        " qui se propage via la dynamique couplée du Neural ODE. Les samples"
        " `known=0` ont un MAE_γ et un bias_γ significativement plus mauvais"
        " que `known=1`, et l'erreur TAS est corrélée à l'erreur γ"
        " sample-par-sample."
    )
    lines.append("")
    lines.append("## Falsifier")
    lines.append("")
    lines.append(
        "- **Statistic primaire** : ratio `MAE_γ(known=0) / MAE_γ(known=1)`"
        " sur descent_shallow agrégé 5 vols, ET corrélation de Pearson entre"
        " `bias_TAS` et `bias_gamma` sur les samples `known=0`"
        " descent_shallow."
    )
    lines.append(
        "- **Threshold** : H_2C survives si ratio MAE_γ > 1.5 ET |Pearson|"
        " > 0.5. Dies si ratio < 1.2 ET |Pearson| < 0.2. Indéterminé sinon."
    )
    lines.append(
        "- **Null world** : si H_2C était fausse (γ n'est pas la source"
        " primaire), MAE_γ serait similaire entre stratums et bias_TAS /"
        " bias_γ seraient indépendants."
    )
    lines.append(
        "- **Data slice** : val seed 0, A320, 5 vols, descent_shallow, sur"
        " les samples du rollout d'inférence baseline (pas de masquage)."
    )
    lines.append(
        "- **Strata** : par stratum `tas_known`. Aussi reporter par-vol."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Pour chaque vol, on charge la trajectoire depuis"
        " `data/flights.delta` (val seed 0, A320), puis on lance"
        " `NodeFDMPredictor.predict_flight` une fois (rollout baseline,"
        " aucun masquage). On classifie chaque step via le même schéma"
        " que `dataset_regime_stats.py` (γ + alt rate basés sur la"
        " *vérité*), on restreint à descent_shallow, et on stratifie par"
        " `tas_known`. Les biais sont `pred − true` ; γ est exprimé en"
        " degrés. Les corrélations Pearson sont calculées sample-par-sample"
        " sur l'agrégat des 5 vols."
    )
    if skipped:
        lines.append("")
        lines.append(
            "Vols skipés (introuvables dans le val split ou trop courts) :"
            f" `{', '.join(skipped)}`."
        )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"### Aggregate ({len(used_flights)} flights, descent_shallow)")
    lines.append("")
    lines.append(
        "| Stratum | n | mean_bias_TAS | MAE_TAS | mean_bias_γ (deg) |"
        " MAE_γ (deg) | mean_bias_alt (m) | MAE_alt (m) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| known=0 | {n0} | {_f(mean_tas_0, '+.3f')} |"
        f" {_f(mae_tas_0, '.3f')} | {_f(mean_g_0, '+.4f')} |"
        f" {_f(mae_g_0, '.4f')} | {_f(mean_a_0, '+.2f')} |"
        f" {_f(mae_a_0, '.2f')} |"
    )
    lines.append(
        f"| known=1 | {n1} | {_f(mean_tas_1, '+.3f')} |"
        f" {_f(mae_tas_1, '.3f')} | {_f(mean_g_1, '+.4f')} |"
        f" {_f(mae_g_1, '.4f')} | {_f(mean_a_1, '+.2f')} |"
        f" {_f(mae_a_1, '.2f')} |"
    )
    lines.append(
        f"| ratio (0/1) | — | — | {_f(ratio_tas, '.3f')} | — |"
        f" {_f(ratio_g, '.3f')} | — | {_f(ratio_a, '.3f')} |"
    )
    lines.append("")
    lines.append("### Correlations on known=0 descent_shallow (aggregate)")
    lines.append("")
    lines.append("| Pair | Pearson r | n |")
    lines.append("|---|---:|---:|")
    lines.append(f"| bias_TAS vs bias_γ | {_f(corr_tas_g_0, '+.4f')} | {n0} |")
    lines.append(f"| bias_TAS vs bias_alt | {_f(corr_tas_a_0, '+.4f')} | {n0} |")
    lines.append("")
    lines.append("### Same on known=1 (contrast)")
    lines.append("")
    lines.append("| Pair | Pearson r | n |")
    lines.append("|---|---:|---:|")
    lines.append(f"| bias_TAS vs bias_γ | {_f(corr_tas_g_1, '+.4f')} | {n1} |")
    lines.append(f"| bias_TAS vs bias_alt | {_f(corr_tas_a_1, '+.4f')} | {n1} |")
    lines.append("")
    lines.append("### Per-flight (descent_shallow)")
    lines.append("")
    lines.append(
        "| Flight | n_0 | n_1 | MAE_γ_0 (deg) | MAE_γ_1 (deg) | ratio_γ |"
        " corr_TAS_γ (known=0) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in per_flight_samples:
        m0 = s.tas_known < 0.5
        m1 = s.tas_known > 0.5
        nf0 = int(m0.sum())
        nf1 = int(m1.sum())
        mae_g0 = _safe_mae(s.bias_gamma_deg[m0])
        mae_g1 = _safe_mae(s.bias_gamma_deg[m1])
        rg = _safe_ratio(mae_g0, mae_g1) if nf1 > 0 else float("nan")
        c = _safe_pearson(s.bias_tas[m0], s.bias_gamma_deg[m0])
        rg_str = _f(rg, '.3f') if nf1 > 0 else "n/a"
        lines.append(
            f"| {s.flight_id} | {nf0} | {nf1} | {_f(mae_g0, '.4f')} |"
            f" {_f(mae_g1, '.4f')} | {rg_str} | {_f(c, '+.4f')} |"
        )
    lines.append("")

    # --- Verdict ---
    lines.append("## Verdict")
    lines.append("")
    abs_corr = abs(corr_tas_g_0) if not math.isnan(corr_tas_g_0) else float("nan")
    if math.isnan(ratio_g) or math.isnan(abs_corr):
        verdict = "indeterminate"
        sentence = (
            f"ratio MAE_γ = {_f(ratio_g, '.3f')}, |Pearson| ="
            f" {_f(abs_corr, '.3f')} — un des deux est non calculable."
        )
    elif ratio_g > 1.5 and abs_corr > 0.5:
        verdict = "survives"
        sentence = (
            f"ratio MAE_γ = {ratio_g:.3f} > 1.5 ET |Pearson| ="
            f" {abs_corr:.3f} > 0.5."
        )
    elif ratio_g < 1.2 and abs_corr < 0.2:
        verdict = "dies"
        sentence = (
            f"ratio MAE_γ = {ratio_g:.3f} < 1.2 ET |Pearson| ="
            f" {abs_corr:.3f} < 0.2."
        )
    else:
        verdict = "indeterminate"
        sentence = (
            f"ratio MAE_γ = {ratio_g:.3f}, |Pearson| = {abs_corr:.3f} — un"
            " seul des deux critères est satisfait."
        )
    lines.append(f"H_2C **{verdict}** : {sentence}")
    lines.append("")

    # --- Caveats ---
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **gamma_pred est sur float, true gamma vient de"
        " `fdm_gamma_rad` qui est lui-même dérivé du baromètre et"
        " lissé** (cf. `gamma_snr_analysis.md`). MAE_γ mesure donc"
        " \"modèle vs gamma lissé\", pas \"modèle vs vérité\". Le SNR"
        " γ_diff est faible (0.34 cruise, 1.08 descent_shallow), ce qui"
        " borne par le bas tout signal de corrélation détectable."
    )
    lines.append(
        "- **Régimes classifiés sur la vérité** : on utilise"
        " `gamma_true` et `d_alt_true` pour étiqueter les samples (même"
        " convention que `dataset_regime_stats.py` et le protocole"
        " Round 1)."
    )
    lines.append(
        "- **Vols sans known=1 sur descent_shallow** : ratio par-vol"
        " marqué `n/a`, exclu de l'agrégat ratio."
    )
    lines.append(
        "- **Effet propagation ODE** : l'inférence est un rollout"
        " complet ; biais_TAS et biais_γ à un sample reflètent l'erreur"
        " accumulée depuis le début du vol. La corrélation Pearson"
        " sample-par-sample mélange donc influence locale et"
        " contamination amont — H_2C reste valide même sous ce mélange,"
        " mais la *direction de causalité* (γ → TAS vs TAS → γ) ne peut"
        " pas être tranchée par cette mesure seule."
    )
    lines.append(
        "- **Stratum tas_known est défini sur le sample courant**, pas"
        " sur la fenêtre amont — un sample known=0 peut hériter d'une"
        " trajectoire propre si tous ses prédécesseurs étaient known=1,"
        " et inversement."
    )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    _print_falsifier_header()

    info = resolve_architecture(ARCH)
    model_path = MODEL_DIR / f"{info.name}_A320"
    if not model_path.exists():
        print(f"[blocker] Model not found at {model_path}", file=sys.stderr)
        return 2

    predictor = NodeFDMPredictor(model_path=model_path, device="cpu")

    df = pl.read_delta(str(DELTA_PATH))
    df = df.filter(pl.col("fdm_flag_valid"))
    df_a320 = df.filter(pl.col("meta_aircraft_type") == "A320")
    val_df = df_a320.filter(pl.col("meta_split") == "val")
    available_ids = set(val_df["meta_flight_id"].unique().to_list())

    per_flight_samples: list[FlightSamples] = []
    used_flights: list[str] = []
    skipped: list[str] = []
    for fid in TARGET_FLIGHTS:
        if fid not in available_ids:
            print(f"[skip] {fid}: not in val split")
            skipped.append(fid)
            continue
        fa = _build_flight_arrays(info, val_df, fid)
        if fa is None:
            print(f"[skip] {fid}: too short / no finite rows")
            skipped.append(fid)
            continue
        fs = _run_baseline_inference(predictor, info, fa)
        if fs is None:
            print(f"[skip] {fid}: prediction alignment failed")
            skipped.append(fid)
            continue
        per_flight_samples.append(fs)
        used_flights.append(fid)
        n_total = fs.bias_tas.size
        n_k0 = int((fs.tas_known < 0.5).sum())
        n_k1 = int((fs.tas_known > 0.5).sum())
        print(
            f"[ok] {fid}: descent_shallow samples = {n_total}"
            f" (known=0: {n_k0}, known=1: {n_k1})"
        )

    if not used_flights:
        print(
            "[blocker] none of the 5 target flights produced predictions",
            file=sys.stderr,
        )
        return 3

    md = render_markdown(per_flight_samples, used_flights, skipped)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
