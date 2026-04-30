"""Round 1 falsifier — TAS target masking ablation on `node_adsb_v1_A320`.

Hypothesis (H_1) under test
---------------------------
Si on prive le backbone du checkpoint `node_adsb_v1_A320` de la consigne
`tas_target` sur les samples `tas_known=1` à l'inférence (en forçant
`tas_diff=0` ET `tas_known=0`), la dégradation produite atteint au moins
80% du gap observé entre `known=1` et `known=0` réel sur les mêmes vols
et le même régime descent_shallow.

Falsifier
---------
- Statistic   : degradation_ratio = (bias_masked - bias_baseline) / Δ_ref
- Threshold   : H_1 survives if ratio ≥ 0.80, dies if ≤ 0.30, else indeterminate.
- Null world  : if H_1 false, masking produces minimal degradation
                (ratio < 0.30) and the +19.73 m/s bias on `known=0`
                comes from a confounder (regime difficulty, selection bias).
- Data slice  : val seed 0, the same 5 flights aggregated in
                `target_known_diagnosis.md`
                (DLH4PV + 4 AVA flights).
- Strata      : per regime (cruise, climb_shallow, climb_saturated,
                descent_shallow, descent_saturated, transition).
                Primary verdict on descent_shallow.

Method (script-level)
---------------------
For each of the 5 flights:
  1. Load the trajectory from `data/flights.delta` (val split, A320).
  2. Run inference TWICE with `NodeFDMPredictor`:
     - Baseline : real `u_seq` (tas_known untouched).
     - Masked   : copy of `u_seq` with `fdm_tas_target_known` set to 0.0
                  on every sample (the TrajectoryLayer then computes
                  `tas_diff = 0` regardless of `tas_target`, and the
                  StructuredLayer sees the `tas_known=0` flag directly
                  via its input cols — both NN feature paths blinded
                  simultaneously, see architectures/adsb.py:65-72 and
                  layers/trajectory.py:199-218).
     - gamma_known and gamma_target are left at their real values.
  3. Classify each step into a regime (mirrors `dataset_regime_stats.py`).
  4. Restrict the comparison to samples that were originally
     `tas_known=1` (we cannot re-blind already blind samples).
  5. Compute mean TAS bias and MAE per (flight, regime, condition).
  6. Aggregate across the 5 flights and compute degradation_ratio
     against Δ_ref = +16.67 m/s (descent_shallow, known=0 minus known=1
     real, from `target_known_diagnosis.md`).

Output
------
- Markdown report : ``data/mardown/target_masking_diag.md``

Usage
-----
    uv run python scripts/debug/diag_target_masking.py
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
REPORT_PATH = Path("data/mardown/target_masking_diag.md")

# The 5 flights aggregated in `data/mardown/target_known_diagnosis.md`
# (val split, seed 0, A320). Verbatim copy from that report's section 4.
TARGET_FLIGHTS: list[str] = [
    "3c6634_DLH4PV_s0",
    "0aca66_AVA8443_s0",
    "0aca66_AVA8447_s0",
    "0aca66_AVA8548_s0",
    "0aca66_AVA8558_s0",
]

# Reference Δ from `target_known_diagnosis.md` § 5 (aggregate, descent_shallow,
# known=0 minus known=1 real bias).
DELTA_REF_DESCENT_SHALLOW = 16.67  # +23.35 - +6.68 m/s

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

PHASES = [
    "cruise",
    "climb_shallow",
    "climb_saturated",
    "descent_shallow",
    "descent_saturated",
    "transition",
]


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class FlightArrays:
    """Arrays needed to run paired inferences and stratify the output."""

    flight_id: str
    x_arr: np.ndarray  # (n, |X|) state truth
    u_arr_baseline: np.ndarray  # (n, |U|) original control
    u_arr_masked: np.ndarray  # (n, |U|) tas_known forced to 0
    e_arr: np.ndarray  # (n, |E0|)
    tas_known_orig: np.ndarray  # (n,) original tas_known values (0/1)
    gamma_known_orig: np.ndarray  # (n,) original gamma_known values (0/1)


@dataclass
class PairedPrediction:
    """Aligned truth / two-condition predictions for one flight."""

    flight_id: str
    n_aligned: int
    tas_true: np.ndarray
    alt_true: np.ndarray
    gamma_true: np.ndarray
    d_alt_true: np.ndarray
    tas_pred_baseline: np.ndarray
    tas_pred_masked: np.ndarray
    tas_known_orig_aligned: np.ndarray  # 0/1 on aligned grid


# ----------------------------------------------------------------------
# Falsifier echo
# ----------------------------------------------------------------------
def _print_falsifier_header() -> None:
    print("=" * 72)
    print("FALSIFIER HEADER (round 1 — target masking ablation)")
    print("=" * 72)
    print(
        "H_1: priver l'inference du checkpoint `node_adsb_v1_A320` de la"
        " consigne `tas_target` sur les samples `tas_known=1` produit une"
        " degradation atteignant >= 80% du gap (known=1 vs known=0 reel)"
        " mesure sur le meme regime descent_shallow."
    )
    print(
        "STATISTIC: degradation_ratio = (bias_masked - bias_baseline) / Delta_ref"
    )
    print(
        "THRESHOLD: H_1 survives ratio >= 0.80, dies ratio <= 0.30, else"
        " indeterminate."
    )
    print(
        "NULL WORLD: masking produces minimal degradation (ratio < 0.30);"
        " the +19.73 m/s bias on known=0 comes from a confounder."
    )
    print(
        "STRATA: cruise, climb_shallow, climb_saturated, descent_shallow,"
        " descent_saturated, transition."
    )
    print(
        "DATA SLICE: val seed 0, A320, 5 flights from"
        f" target_known_diagnosis.md: {TARGET_FLIGHTS}."
    )
    print("=" * 72)


# ----------------------------------------------------------------------
# Phase classification
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
# Per-flight loading + paired inference
# ----------------------------------------------------------------------
def _build_flight_arrays(
    info,
    val_df: pl.DataFrame,
    flight_id: str,
) -> FlightArrays | None:
    """Materialise the arrays for one flight (baseline + masked U)."""
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
    gamma_known_idx = info.u_cols.index("fdm_gamma_target_known")

    # Replace NaN in u_arr (especially tas_target) with finite zeros so
    # that the trajectory layer's nan_to_num behaviour is exercised
    # consistently between baseline and masked. Baseline path keeps the
    # original tas_known mask, so tas_diff = 0 is still produced where
    # the target was originally NaN (matches inference today).
    u_arr = np.where(np.isnan(u_arr), 0.0, u_arr)

    tas_known_orig = u_arr[:, tas_known_idx].copy()
    gamma_known_orig = u_arr[:, gamma_known_idx].copy()

    u_arr_masked = u_arr.copy()
    # Force tas_known = 0 everywhere -> TrajectoryLayer computes
    # tas_diff = known_tas * (target - tas) = 0, AND the StructuredLayer
    # receives the tas_known=0 flag directly via its input_cols.
    # gamma_known is left untouched per protocol.
    u_arr_masked[:, tas_known_idx] = 0.0

    return FlightArrays(
        flight_id=flight_id,
        x_arr=x_arr,
        u_arr_baseline=u_arr,
        u_arr_masked=u_arr_masked,
        e_arr=e_arr,
        tas_known_orig=tas_known_orig,
        gamma_known_orig=gamma_known_orig,
    )


def _run_paired_inference(
    predictor: NodeFDMPredictor,
    info,
    fa: FlightArrays,
) -> PairedPrediction | None:
    """Run baseline + masked inference, align truth/predictions."""
    x_arr = fa.x_arr
    n_true = x_arr.shape[0]
    x0 = x_arr[0]

    pred_baseline = predictor.predict_flight(x0, fa.u_arr_baseline, fa.e_arr)
    pred_masked = predictor.predict_flight(x0, fa.u_arr_masked, fa.e_arr)

    n_pred = len(pred_baseline["era_tas_ms"])
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

    return PairedPrediction(
        flight_id=fa.flight_id,
        n_aligned=end,
        tas_true=x_arr[true_slice, tas_idx],
        alt_true=x_arr[true_slice, alt_idx],
        gamma_true=x_arr[true_slice, gamma_idx],
        d_alt_true=d_alt_full[true_slice],
        tas_pred_baseline=pred_baseline["era_tas_ms"][pred_slice],
        tas_pred_masked=pred_masked["era_tas_ms"][pred_slice],
        tas_known_orig_aligned=fa.tas_known_orig[true_slice],
    )


# ----------------------------------------------------------------------
# Stratification + metrics
# ----------------------------------------------------------------------
def _stratified_rows(pp: PairedPrediction) -> list[dict]:
    """Per (flight, regime), restrict to originally tas_known=1 samples,
    then compute MAE/bias for baseline vs masked TAS.
    """
    labels = _classify_phases_np(pp.gamma_true, pp.d_alt_true)
    known_mask = pp.tas_known_orig_aligned > 0.5
    rows: list[dict] = []
    for phase in PHASES:
        mask = (labels == phase) & known_mask
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "flight": pp.flight_id, "phase": phase, "n": 0,
                "bias_baseline": float("nan"), "mae_baseline": float("nan"),
                "bias_masked": float("nan"), "mae_masked": float("nan"),
                "delta": float("nan"),
            })
            continue
        diff_b = pp.tas_pred_baseline[mask] - pp.tas_true[mask]
        diff_m = pp.tas_pred_masked[mask] - pp.tas_true[mask]
        bias_b = float(np.mean(diff_b))
        bias_m = float(np.mean(diff_m))
        rows.append({
            "flight": pp.flight_id, "phase": phase, "n": n,
            "bias_baseline": bias_b,
            "mae_baseline": float(np.mean(np.abs(diff_b))),
            "bias_masked": bias_m,
            "mae_masked": float(np.mean(np.abs(diff_m))),
            "delta": bias_m - bias_b,
        })
    return rows


def _aggregate_per_phase(per_flight_rows: list[dict]) -> list[dict]:
    """Aggregate baseline/masked bias across all flights per phase
    (sample-weighted).
    """
    agg: list[dict] = []
    for phase in PHASES:
        n_total = 0
        sum_b = 0.0
        sum_m = 0.0
        sum_abs_b = 0.0
        sum_abs_m = 0.0
        for r in per_flight_rows:
            if r["phase"] != phase or r["n"] == 0:
                continue
            n = r["n"]
            n_total += n
            sum_b += r["bias_baseline"] * n
            sum_m += r["bias_masked"] * n
            sum_abs_b += r["mae_baseline"] * n
            sum_abs_m += r["mae_masked"] * n
        if n_total == 0:
            agg.append({
                "phase": phase, "n": 0,
                "bias_baseline": float("nan"), "mae_baseline": float("nan"),
                "bias_masked": float("nan"), "mae_masked": float("nan"),
                "delta": float("nan"),
            })
            continue
        bias_b = sum_b / n_total
        bias_m = sum_m / n_total
        agg.append({
            "phase": phase, "n": n_total,
            "bias_baseline": bias_b,
            "mae_baseline": sum_abs_b / n_total,
            "bias_masked": bias_m,
            "mae_masked": sum_abs_m / n_total,
            "delta": bias_m - bias_b,
        })
    return agg


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------
def _f(v: float, fmt: str = "+.3f") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return format(v, fmt)


def _ratio(delta: float, ref: float = DELTA_REF_DESCENT_SHALLOW) -> float:
    if math.isnan(delta) or abs(ref) < 1e-9:
        return float("nan")
    return delta / ref


def render_markdown(
    per_flight_rows: list[dict],
    agg_rows: list[dict],
    used_flights: list[str],
    skipped_flights: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Round 1 — Target masking ablation")
    lines.append("")

    # --- Hypothesis verbatim ---
    lines.append("## Hypothesis under test")
    lines.append("")
    lines.append(
        "Si on prive le backbone du checkpoint `node_adsb_v1_A320` de la"
        " consigne `tas_target` sur les samples `tas_known=1` à l'inférence"
        " (en forçant `tas_diff=0` ET `tas_known=0`), la dégradation"
        " produite atteint au moins 80% du gap observé entre `known=1` et"
        " `known=0` réel sur les mêmes vols et le même régime"
        " descent_shallow."
    )
    lines.append("")

    # --- Falsifier verbatim ---
    lines.append("## Falsifier")
    lines.append("")
    lines.append(
        "- **Statistic** : `degradation_ratio = (mean_bias_TAS_masked −"
        " mean_bias_TAS_baseline) / (mean_bias_TAS_known0_real −"
        " mean_bias_TAS_known1_real)`. Computed on samples that are"
        " originally `tas_known=1` only (we can't mask what's already"
        " masked). Reported per flight AND aggregated."
    )
    lines.append(
        "- **Threshold** : H_1 survives if `degradation_ratio ≥ 0.80`."
        " Dies if `≤ 0.30`. Indeterminate between."
    )
    lines.append(
        "- **Null world** : if H_1 were false (backbone doesn't use"
        " `tas_target` meaningfully), masking would produce minimal"
        " degradation (ratio < 0.30) — the +19.73 m/s bias on `known=0`"
        " would come from another confounder (regime difficulty,"
        " selection bias)."
    )
    lines.append(
        "- **Data slice** : val seed 0, the SAME 5 flights aggregated in"
        " `target_known_diagnosis.md` ("
        + ", ".join(f"`{f}`" for f in TARGET_FLIGHTS)
        + ")."
    )
    lines.append(
        "- **Strata** : by regime (cruise, climb_shallow, climb_saturated,"
        " descent_shallow, descent_saturated, transition). Primary"
        " verdict on descent_shallow; report all strata."
    )
    lines.append(
        "- **Reference** : `Δ_ref = +16.67 m/s` (descent_shallow,"
        " agrégat 5 vols, `known=0 − known=1` réel, source"
        " `target_known_diagnosis.md` § 5)."
    )
    lines.append("")

    # --- Method ---
    lines.append("## Method")
    lines.append("")
    lines.append(
        "Pour chaque vol des 5 ciblés (val seed 0, A320), on charge la"
        " trajectoire depuis `data/flights.delta`, on lance"
        " `NodeFDMPredictor.predict_flight` deux fois sur le même `x0`"
        " et le même `e_seq` :"
    )
    lines.append(
        "1. **Baseline** : `u_seq` original (le `fdm_tas_target_known` réel"
        " gouverne le masquage natif côté `TrajectoryLayer`)."
    )
    lines.append(
        "2. **Masked**  : copie de `u_seq` avec `fdm_tas_target_known`"
        " forcé à 0 sur tous les samples — `TrajectoryLayer` (l. 199-218)"
        " calcule alors `tas_diff = 0` partout, et la `StructuredLayer`"
        " reçoit le drapeau `tas_known=0` directement via ses `input_cols`"
        " (les deux chemins-feature du NN sont aveuglés ensemble)."
        " `gamma_known`/`gamma_target` sont laissés intacts."
    )
    lines.append(
        "Le test n'est calculé que sur les samples originellement"
        " `tas_known=1` (impossible de re-masquer ce qui est déjà masqué)."
        " Bias = `pred − true`. Régimes via la même classification que"
        " `dataset_regime_stats.py` (γ + alt rate)."
    )
    if skipped_flights:
        lines.append("")
        lines.append(
            "Vols skipés (introuvables dans le val split ou trajectoire"
            f" trop courte) : `{', '.join(skipped_flights)}`."
        )
    lines.append("")

    # --- Aggregate table ---
    lines.append("## Result")
    lines.append("")
    lines.append(f"### Aggregate ({len(used_flights)} flights)")
    lines.append("")
    lines.append(
        "| Regime | n | bias_baseline | bias_masked | Δ (masked − baseline)"
        " | Δ_ref (known0 − known1) | degradation_ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in agg_rows:
        if r["n"] == 0:
            lines.append(
                f"| {r['phase']} | 0 | n/a | n/a | n/a | — | n/a |"
            )
            continue
        ref = (
            DELTA_REF_DESCENT_SHALLOW if r["phase"] == "descent_shallow"
            else float("nan")
        )
        ref_str = f"{ref:+.2f}" if not math.isnan(ref) else "—"
        ratio = _ratio(r["delta"], ref) if not math.isnan(ref) else float("nan")
        lines.append(
            f"| {r['phase']} | {r['n']} | {_f(r['bias_baseline'])} |"
            f" {_f(r['bias_masked'])} | {_f(r['delta'])} | {ref_str}"
            f" | {_f(ratio, '.3f') if not math.isnan(ratio) else 'n/a'} |"
        )
    lines.append("")

    # --- Per-flight (descent_shallow only) ---
    lines.append("### Per-flight (descent_shallow only)")
    lines.append("")
    lines.append(
        "| Flight | n | bias_baseline | bias_masked | Δ |"
        " degradation_ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in per_flight_rows:
        if r["phase"] != "descent_shallow":
            continue
        if r["n"] == 0:
            lines.append(f"| {r['flight']} | 0 | n/a | n/a | n/a | n/a |")
            continue
        ratio = _ratio(r["delta"])
        lines.append(
            f"| {r['flight']} | {r['n']} | {_f(r['bias_baseline'])} |"
            f" {_f(r['bias_masked'])} | {_f(r['delta'])} |"
            f" {_f(ratio, '.3f')} |"
        )
    lines.append("")

    # --- Per-flight, all regimes (audit-friendly) ---
    lines.append("### Per-flight, all regimes (audit)")
    lines.append("")
    lines.append(
        "| Flight | Regime | n | bias_baseline | bias_masked | Δ |"
        " mae_baseline | mae_masked |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in per_flight_rows:
        if r["n"] == 0:
            continue
        lines.append(
            f"| {r['flight']} | {r['phase']} | {r['n']} |"
            f" {_f(r['bias_baseline'])} | {_f(r['bias_masked'])} |"
            f" {_f(r['delta'])} | {_f(r['mae_baseline'], '.3f')} |"
            f" {_f(r['mae_masked'], '.3f')} |"
        )
    lines.append("")

    # --- Verdict ---
    lines.append("## Verdict")
    lines.append("")
    desc = next(
        (r for r in agg_rows if r["phase"] == "descent_shallow"), None
    )
    if desc is None or desc["n"] == 0:
        lines.append(
            "H_1 indeterminate: aucune donnée descent_shallow sur les"
            " samples `tas_known=1` originaux dans les 5 vols."
        )
    else:
        ratio = _ratio(desc["delta"])
        if math.isnan(ratio):
            verdict = "indeterminate"
            justification = (
                "ratio non calculable (Δ ou Δ_ref dégénéré)."
            )
        elif ratio >= 0.80:
            verdict = "survives"
            justification = (
                f"degradation_ratio = {ratio:.3f} ≥ 0.80 — le masquage"
                " reproduit ≥ 80% du gap réel `known=0 vs known=1`."
            )
        elif ratio <= 0.30:
            verdict = "dies"
            justification = (
                f"degradation_ratio = {ratio:.3f} ≤ 0.30 — le masquage"
                " produit une dégradation marginale, le bias `known=0`"
                " réel est donc dû à un autre confondeur."
            )
        else:
            verdict = "indeterminate"
            justification = (
                f"degradation_ratio = {ratio:.3f} ∈ ]0.30, 0.80[ — le"
                " masquage explique une part substantielle mais"
                " incomplète du gap; la cause est partiellement le"
                " manque de target."
            )
        lines.append(f"H_1 **{verdict}** : {justification}")
    lines.append("")

    # --- Caveats ---
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **Asymétrie du test** : seuls les samples originellement"
        " `tas_known=1` peuvent être masqués; les `known=0` natifs sont"
        " déjà aveugles, on ne peut pas les *unmask*. Le test mesure"
        " donc \"perte d'information sur les samples qui en avaient\","
        " pas \"gain d'information sur ceux qui n'en avaient pas\"."
    )
    lines.append(
        "- **Sanity check baseline = `known=1` réel** : par construction"
        " la baseline (samples `known=1` originaux, sans modification"
        " de `u_seq`) doit reproduire les biais `known=1` du rapport"
        " `target_known_diagnosis.md` § 4.1 (DLH4PV descent_shallow"
        " known=1 : bias = +3.45 m/s, n=58). Toute divergence > 0.5 m/s"
        " indique un drift d'inférence non lié à F3 / au seed."
    )
    lines.append(
        "- **gamma_known laissé réel** : le test isole l'effet TAS;"
        " le bias d'altitude/γ peut bouger comme effet collatéral via"
        " la dynamique couplée du Neural ODE, mais cela n'invalide pas"
        " la statistique principale qui porte sur la TAS."
    )
    lines.append(
        "- **Effet propagation ODE** : l'inférence est un rollout complet"
        " — masquer `tas_diff` à un sample décale la trajectoire jusqu'à"
        " la fin du vol. Le bias mesuré sur les samples `known=1`"
        " inclut donc un effet \"contamination\" par les samples masqués"
        " environnants. Ce n'est PAS un bug du protocole : c'est"
        " exactement ce que subit un vol où `tas_known=0` apparaît"
        " dans 45% des samples descent_shallow."
    )
    lines.append(
        "- **Tailles d'échantillons** : reportées par cellule. Δ_ref"
        " agrégé est issu d'un `n=94 (known=0) + n=64 (known=1)` (cf."
        " rapport référent) — toute conclusion par-vol avec n<10 est"
        " indicative."
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

    per_flight_rows: list[dict] = []
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
        pp = _run_paired_inference(predictor, info, fa)
        if pp is None:
            print(f"[skip] {fid}: prediction alignment failed")
            skipped.append(fid)
            continue
        rows = _stratified_rows(pp)
        per_flight_rows.extend(rows)
        used_flights.append(fid)
        n_known1 = int((pp.tas_known_orig_aligned > 0.5).sum())
        print(
            f"[ok] {fid}: aligned n={pp.n_aligned}, samples"
            f" tas_known=1 (eligible) = {n_known1}"
        )

    if not used_flights:
        print(
            "[blocker] none of the 5 target flights produced predictions",
            file=sys.stderr,
        )
        return 3

    agg_rows = _aggregate_per_phase(per_flight_rows)
    md = render_markdown(per_flight_rows, agg_rows, used_flights, skipped)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print(f"\nReport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
