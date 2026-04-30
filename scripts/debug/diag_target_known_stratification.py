"""Diagnostic: is the descent-shallow TAS bias driven by missing target?

Hypothesis under test
---------------------
Post-F3 analysis (`data/mardown/dlh4pv_postF3_analysis.md`) showed a TAS
bias of +10.73 m/s in descent-shallow on flight DLH4PV. The smoking gun
of F1 (per-head input routing) was invalidated by 4 independent diagnostics.

Hypothesis: the descent-shallow bias is caused by **missing FMS targets**
(`fdm_tas_target_known == 0`), not bad routing. When the target is unknown,
the trajectory layer forces `tas_diff = 0` (no pilot signal), and the NN
drifts.

Method
------
Three quantitative questions:

Q1. Per-phase fraction of samples with target unknown on the A320 train
    split. Phases mirror `dataset_regime_stats.py` (cruise / climb shallow
    / climb saturated / descent shallow / descent saturated / transition).

Q2. Documented in the markdown report from the source code (this script
    cites the exact lines).

Q3. Re-run inference on DLH4PV (val split) and stratify the descent-shallow
    metrics by `tas_known` (and γ metrics by `gamma_known`).

Outputs
-------
- ``data/mardown/target_known_diagnosis.md`` : full report.

Usage
-----
    uv run python scripts/debug/diag_target_known_stratification.py
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
TARGET_FLIGHT_ID = "3c6634_DLH4PV_s0"
REPORT_PATH = Path("data/mardown/target_known_diagnosis.md")

# Phase thresholds (mirror scripts/debug/dataset_regime_stats.py and
# scripts/debug/diag_dlh4pv_postF3.py)
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
# Q1: per-phase fraction without target on the A320 train split
# ----------------------------------------------------------------------
def _classify_expr_v2() -> pl.Expr:
    """Map each row to one of PHASES."""
    gamma = pl.col("fdm_gamma_rad")
    alt_rate = pl.col("fdm_d_alt_ms")

    is_cruise = (gamma.abs() < GAMMA_CRUISE_RAD) & (alt_rate.abs() < ALT_RATE_CRUISE)
    is_climb = (gamma > GAMMA_CLIMB_RAD) | (alt_rate > ALT_RATE_CLIMB)
    is_descent = (gamma < GAMMA_DESCENT_RAD) | (alt_rate < ALT_RATE_DESCENT)

    is_climb_sat = is_climb & (alt_rate > ALT_RATE_CLIMB_SAT)
    is_descent_sat = is_descent & (alt_rate < ALT_RATE_DESCENT_SAT)

    return (
        pl.when(is_cruise)
        .then(pl.lit("cruise"))
        .when(is_climb_sat)
        .then(pl.lit("climb_saturated"))
        .when(is_climb)
        .then(pl.lit("climb_shallow"))
        .when(is_descent_sat)
        .then(pl.lit("descent_saturated"))
        .when(is_descent)
        .then(pl.lit("descent_shallow"))
        .otherwise(pl.lit("transition"))
        .alias("phase")
    )


def compute_phase_known_table(
    delta_path: Path,
    *,
    typecode: str = "A320",
    split: str = "train",
) -> list[dict]:
    """Per phase, return counts and fractions of samples missing targets."""
    lf = (
        pl.scan_delta(str(delta_path))
        .filter(pl.col("fdm_flag_valid"))
        .filter(pl.col("meta_aircraft_type") == typecode)
        .filter(pl.col("meta_split") == split)
        .select(
            [
                "fdm_gamma_rad",
                "fdm_d_alt_ms",
                "fdm_tas_target_known",
                "fdm_gamma_target_known",
            ]
        )
        .drop_nulls(["fdm_gamma_rad", "fdm_d_alt_ms"])
        .filter(
            ~pl.col("fdm_gamma_rad").is_nan() & ~pl.col("fdm_d_alt_ms").is_nan()
        )
        .with_columns(_classify_expr_v2())
        # Boolean to float for uniform aggregation.
        .with_columns(
            pl.col("fdm_tas_target_known").cast(pl.Float64).alias("tas_known_f"),
            pl.col("fdm_gamma_target_known").cast(pl.Float64).alias("gamma_known_f"),
        )
    )
    df = lf.collect()
    n_total = df.height

    rows: list[dict] = []
    for phase in PHASES:
        sub = df.filter(pl.col("phase") == phase)
        n = sub.height
        if n == 0:
            rows.append({"phase": phase, "n": 0, "frac_phase": 0.0,
                         "tas_unknown": 0.0, "gamma_unknown": 0.0,
                         "both_unknown": 0.0, "any_unknown": 0.0})
            continue
        tas_unk = (sub["tas_known_f"] < 0.5).sum()
        gam_unk = (sub["gamma_known_f"] < 0.5).sum()
        both_unk = ((sub["tas_known_f"] < 0.5) & (sub["gamma_known_f"] < 0.5)).sum()
        any_unk = ((sub["tas_known_f"] < 0.5) | (sub["gamma_known_f"] < 0.5)).sum()
        rows.append({
            "phase": phase,
            "n": n,
            "frac_phase": n / n_total,
            "tas_unknown": tas_unk / n,
            "gamma_unknown": gam_unk / n,
            "both_unknown": both_unk / n,
            "any_unknown": any_unk / n,
        })

    rows.append({
        "phase": "ALL",
        "n": n_total,
        "frac_phase": 1.0,
        "tas_unknown": float((df["tas_known_f"] < 0.5).sum()) / max(n_total, 1),
        "gamma_unknown": float((df["gamma_known_f"] < 0.5).sum()) / max(n_total, 1),
        "both_unknown": float(((df["tas_known_f"] < 0.5) & (df["gamma_known_f"] < 0.5)).sum()) / max(n_total, 1),
        "any_unknown": float(((df["tas_known_f"] < 0.5) | (df["gamma_known_f"] < 0.5)).sum()) / max(n_total, 1),
    })
    return rows


# ----------------------------------------------------------------------
# Q3: stratified MAE/bias on DLH4PV
# ----------------------------------------------------------------------
@dataclass
class FlightInference:
    time_min: np.ndarray
    alt_true: np.ndarray
    tas_true: np.ndarray
    gamma_true: np.ndarray
    alt_pred: np.ndarray
    tas_pred: np.ndarray
    gamma_pred: np.ndarray
    tas_known: np.ndarray  # 0/1
    gamma_known: np.ndarray  # 0/1
    d_alt_true: np.ndarray
    flight_id: str


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


def _run_inference_for_flight(
    predictor: NodeFDMPredictor,
    info,
    flight_df: pl.DataFrame,
    flight_id: str,
) -> FlightInference | None:
    """Run inference + align truth/pred for a single flight."""
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

    x0 = x_arr[0]
    predictions = predictor.predict_flight(x0, u_arr, e_arr)
    n_pred = len(predictions["era_tas_ms"])
    n_true = x_arr.shape[0]
    end = min(n_true - 1, n_pred)
    if end <= 1:
        return None
    pred_slice = slice(0, end)
    true_slice = slice(1, end + 1)

    alt_idx = info.x_cols.index("raw_alt_m")
    tas_idx = info.x_cols.index("era_tas_ms")
    gamma_idx = info.x_cols.index("fdm_gamma_rad")
    tas_known_idx = info.u_cols.index("fdm_tas_target_known")
    gamma_known_idx = info.u_cols.index("fdm_gamma_target_known")

    alt_full = x_arr[:, alt_idx]
    d_alt_full = np.gradient(alt_full, STEP_S)

    return FlightInference(
        time_min=(np.arange(end) + 1) * STEP_S / 60.0,
        alt_true=x_arr[true_slice, alt_idx],
        tas_true=x_arr[true_slice, tas_idx],
        gamma_true=x_arr[true_slice, gamma_idx],
        alt_pred=predictions["raw_alt_m"][pred_slice],
        tas_pred=predictions["era_tas_ms"][pred_slice],
        gamma_pred=predictions["fdm_gamma_rad"][pred_slice],
        tas_known=u_arr[true_slice, tas_known_idx],
        gamma_known=u_arr[true_slice, gamma_known_idx],
        d_alt_true=d_alt_full[true_slice],
        flight_id=flight_id,
    )


def stratified_metrics(
    fi: FlightInference, labels: np.ndarray
) -> list[dict]:
    """For each (phase, known_strat), compute MAE/bias."""
    rows: list[dict] = []
    for phase in PHASES:
        phase_mask = labels == phase
        if phase_mask.sum() == 0:
            continue
        for known_kind, known_arr in (
            ("tas", fi.tas_known),
            ("gamma", fi.gamma_known),
        ):
            for known_val in (0, 1):
                bool_mask = (known_arr > 0.5) if known_val == 1 else (known_arr <= 0.5)
                mask = phase_mask & bool_mask
                n = int(mask.sum())
                if n == 0:
                    rows.append({
                        "flight": fi.flight_id, "phase": phase,
                        "known_kind": known_kind, "known": known_val,
                        "n": 0, "mae_tas": float("nan"),
                        "bias_tas": float("nan"), "mae_alt": float("nan"),
                        "mae_gamma_deg": float("nan"),
                    })
                    continue
                rows.append({
                    "flight": fi.flight_id, "phase": phase,
                    "known_kind": known_kind, "known": known_val,
                    "n": n,
                    "mae_tas": float(np.mean(np.abs(fi.tas_pred[mask] - fi.tas_true[mask]))),
                    "bias_tas": float(np.mean(fi.tas_pred[mask] - fi.tas_true[mask])),
                    "mae_alt": float(np.mean(np.abs(fi.alt_pred[mask] - fi.alt_true[mask]))),
                    "mae_gamma_deg": float(
                        np.degrees(np.mean(np.abs(fi.gamma_pred[mask] - fi.gamma_true[mask])))
                    ),
                })
    return rows


def run_q3(
    extra_flights: int = 4,
) -> tuple[list[dict], list[str]]:
    """Run inference + stratification on DLH4PV plus a few more val flights."""
    info = resolve_architecture(ARCH)
    model_path = MODEL_DIR / f"{info.name}_A320"
    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}")

    predictor = NodeFDMPredictor(model_path=model_path, device="cpu")

    df = pl.read_delta(str(DELTA_PATH))
    df = df.filter(pl.col("fdm_flag_valid"))
    df_a320 = df.filter(pl.col("meta_aircraft_type") == "A320")
    val_df = df_a320.filter(pl.col("meta_split") == "val")
    flight_ids = val_df["meta_flight_id"].unique().sort().to_list()

    if TARGET_FLIGHT_ID not in flight_ids:
        raise SystemExit(
            f"Flight {TARGET_FLIGHT_ID!r} not in val split. "
            f"Available[:20]={flight_ids[:20]}"
        )

    selected: list[str] = [TARGET_FLIGHT_ID]
    for fid in flight_ids:
        if fid == TARGET_FLIGHT_ID:
            continue
        selected.append(fid)
        if len(selected) >= 1 + extra_flights:
            break

    rows: list[dict] = []
    used: list[str] = []
    for fid in selected:
        flight_df = val_df.filter(pl.col("meta_flight_id") == fid)
        fi = _run_inference_for_flight(predictor, info, flight_df, fid)
        if fi is None:
            print(f"[skip] {fid}: too short for inference")
            continue
        labels = _classify_phases_np(fi.gamma_true, fi.d_alt_true)
        rows.extend(stratified_metrics(fi, labels))
        used.append(fid)
        print(f"[ok] {fid}: n={len(fi.tas_true)} aligned points")

    return rows, used


# ----------------------------------------------------------------------
# Markdown rendering
# ----------------------------------------------------------------------
def _f(v: float, fmt: str = ".3f") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return format(v, fmt)


def _pct(v: float) -> str:
    if v is None or math.isnan(v):
        return "n/a"
    return f"{100.0 * v:.2f}%"


def render_markdown(
    q1_rows: list[dict],
    q3_rows: list[dict],
    flights_used: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Diagnostic — bias TAS descent shallow lié à l'absence de target ?")
    lines.append("")
    lines.append("Hypothèse testée : le bias TAS de **+10.73 m/s** observé en descent")
    lines.append("shallow sur DLH4PV (cf. `data/mardown/dlh4pv_postF3_analysis.md`)")
    lines.append("vient de **l'absence de consigne FMS** (`fdm_tas_target_known == 0`)")
    lines.append("plutôt que d'un mauvais routage de features (F1 / per-head, déjà")
    lines.append("invalidé par 4 diagnostics).")
    lines.append("")

    # 1. Schema
    lines.append("## 1. Schéma colonnes target / known")
    lines.append("")
    lines.append("Noms exacts trouvés dans `data/flights.delta` :")
    lines.append("")
    lines.append("| Colonne                       | Dtype    | Sémantique                                                        |")
    lines.append("|---|---|---|")
    lines.append("| `fdm_tas_target_ms`           | Float64  | Target TAS issu de l'enveloppe Mach/CAS (NaN si non couvert)      |")
    lines.append("| `fdm_tas_target_known`        | Boolean  | True ⟺ `tas_target_ms` non-NaN                                   |")
    lines.append("| `fdm_gamma_target_rad`        | Float64  | Target γ issu de `fdm_gamma_from_alt`/`gamma_sel`/`vz_sel` (NaN→0) |")
    lines.append("| `fdm_gamma_target_known`      | Float64  | 1.0 ⟺ source γ disponible, 0.0 sinon                             |")
    lines.append("| `fdm_tas_diff_ms`             | Float64  | `target − tas` (m/s), forcé à 0 quand target NaN (convert.py:120) |")
    lines.append("| `fdm_gamma_diff_rad`          | Float64  | `target − gamma` (rad), forcé à 0 quand target NaN                |")
    lines.append("")
    lines.append(
        "Source des emissions : `packages/node-fdm-data/src/node_fdm_data/segments.py`"
        " — `_build_tas_target` (l. 379-434) et `_build_gamma_target` (l. 465-477)."
    )
    lines.append("")

    # 2. Q1
    lines.append("## 2. Q1 — Fraction de samples sans target par phase (A320 train)")
    lines.append("")
    lines.append(
        "Filtre : `fdm_flag_valid AND meta_aircraft_type == 'A320' AND meta_split == 'train'`."
    )
    lines.append("Phases : seuils alignés sur `dataset_regime_stats.py`.")
    lines.append("")
    lines.append("| Phase             | n           | % phase | tas_known=0 | gamma_known=0 | both=0  | any=0   |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in q1_rows:
        if r["phase"] == "ALL":
            continue
        lines.append(
            f"| {r['phase']:<17} | {r['n']:>11,} | {_pct(r['frac_phase']):>7} | "
            f"{_pct(r['tas_unknown']):>11} | {_pct(r['gamma_unknown']):>13} | "
            f"{_pct(r['both_unknown']):>7} | {_pct(r['any_unknown']):>7} |"
        )
    all_row = next((r for r in q1_rows if r["phase"] == "ALL"), None)
    if all_row is not None:
        lines.append(
            f"| **{all_row['phase']:<13}** | **{all_row['n']:>7,}** | "
            f"**{_pct(all_row['frac_phase']):>7}** | "
            f"**{_pct(all_row['tas_unknown']):>11}** | "
            f"**{_pct(all_row['gamma_unknown']):>13}** | "
            f"**{_pct(all_row['both_unknown']):>7}** | "
            f"**{_pct(all_row['any_unknown']):>7}** |"
        )
    lines.append("")

    # Verdict on Q1
    desc_shallow = next((r for r in q1_rows if r["phase"] == "descent_shallow"), None)
    if desc_shallow and desc_shallow["n"] > 0:
        crit_msg = (
            "**Seuil critique** : si descent_shallow > 50 % `tas_known=0`, hypothèse forte. "
            f"Mesuré : `tas_known=0` = {_pct(desc_shallow['tas_unknown'])}, "
            f"`gamma_known=0` = {_pct(desc_shallow['gamma_unknown'])}."
        )
        lines.append(crit_msg)
        lines.append("")

    # 3. Q2
    lines.append("## 3. Q2 — Stratégie d'imputation actuelle")
    lines.append("")
    lines.append("### 3.1. Génération du target")
    lines.append("")
    lines.append(
        "`fdm_tas_target_ms` (et son booléen `_known`) est construit dans"
        " `packages/node-fdm-data/src/node_fdm_data/segments.py:379-434` à partir de"
        " l'enveloppe `min(Mach→TAS, CAS→TAS)` issue des sélections FMS détectées par"
        " `_detect_*_segments`. Lignes 421-423 : si une seule source est disponible,"
        " `target = tas_mach OU tas_cas` ; ligne 425-428 : fallback sur `fdm_tas_sel_kt`"
        " si rien n'est couvert. Ligne 430 : `known = ~np.isnan(target)`. **Aucun"
        " backward-fill global** : les rows non couvertes restent NaN."
    )
    lines.append("")
    lines.append(
        "`fdm_gamma_target_rad` (`segments.py:465-477`) construit le target γ depuis"
        " `fdm_gamma_from_alt_rad` / `fdm_gamma_sel_rad` / `fdm_vz_sel_ftmin`. Ligne 472 :"
        " `gamma_known = (~np.isnan(gamma_target)).astype(float64)`. Ligne 473 :"
        " **`gamma_filled = np.where(np.isnan, 0.0, target)`** — donc ici le target γ"
        " est imputé à **0.0** (NaN écrasé) avant d'être stocké, mais le booléen"
        " `gamma_known` permet à l'aval de savoir que c'était inconnu."
    )
    lines.append("")
    lines.append("### 3.2. Construction des diff (convert.py)")
    lines.append("")
    lines.append(
        "`packages/node-fdm-data/src/node_fdm_data/preprocessing/convert.py:117-128` :"
    )
    lines.append("")
    lines.append("```python")
    lines.append("DELTA_DIFFS = [")
    lines.append('    ("fdm_alt_target_m",    "raw_alt_m",     "fdm_alt_diff_m"),')
    lines.append('    ("fdm_tas_target_ms",   "era_tas_ms",    "fdm_tas_diff_ms"),')
    lines.append('    ("fdm_gamma_target_rad","fdm_gamma_rad", "fdm_gamma_diff_rad"),')
    lines.append("]")
    lines.append("# Where target is NaN, diff is 0 (NaN-preserving gamma target, AXM-809).")
    lines.append("diff_exprs = [")
    lines.append("    pl.when(pl.col(tgt).is_nan() | pl.col(tgt).is_null())")
    lines.append("    .then(pl.lit(0.0))")
    lines.append("    .otherwise(pl.col(tgt) - pl.col(src))")
    lines.append("    .alias(out)")
    lines.append("    for tgt, src, out in DELTA_DIFFS")
    lines.append("]")
    lines.append("```")
    lines.append("")
    lines.append(
        "**Conclusion** : `fdm_tas_diff_ms` est forcé à `0.0` quand `fdm_tas_target_ms`"
        " est NaN. Idem `fdm_gamma_diff_rad`. Le diff est précomputé dans le Delta."
    )
    lines.append("")
    lines.append("### 3.3. Re-calcul à l'inférence (trajectory layer)")
    lines.append("")
    lines.append(
        "`packages/node-fdm/src/node_fdm/layers/trajectory.py:199-218` recalcule"
        " `tas_diff` à chaque pas du Neural ODE :"
    )
    lines.append("")
    lines.append("```python")
    lines.append("# When known=1: tas_diff = target - tas")
    lines.append("# When known=0: tas_diff = 0 (no target, avoid the false -tas signal")
    lines.append("#   that nan_to_num(target, 0) would produce)")
    lines.append("tas_target_raw = x[tas_sel_col]                       # fdm_tas_target_ms")
    lines.append("known_tas = x[tas_known_col]                          # fdm_tas_target_known {0,1}")
    lines.append("tas_target = torch.nan_to_num(tas_target_raw, nan=0.0)")
    lines.append("output[c['tas_diff']] = known_tas * (tas_target - tas)  # masquage explicite")
    lines.append("")
    lines.append("# Pass through tas_known flag for the StructuredLayer")
    lines.append("output[tas_known_col] = x[tas_known_col]")
    lines.append("```")
    lines.append("")
    lines.append("Idem pour γ (lignes 220-238).")
    lines.append("")
    lines.append("### 3.4. Features alimentant le NN")
    lines.append("")
    lines.append(
        "Architecture `node_adsb_v1` (`packages/node-fdm/src/node_fdm/architectures/adsb.py`)"
        " : la `StructuredLayer` (data_ode, trainable) reçoit en entrée"
        " `X_COLS + U_ODE_COLS + E0_COLS + E1_COLS + ['fdm_gamma_target_known',"
        " 'fdm_tas_target_known']` (l. 65-72). `U_ODE_COLS` est **vide** (l. 54 du schéma)"
        " : aucune cible n'entre directement dans le NN, mais via E1 le NN voit"
        " `fdm_tas_diff_ms`, `fdm_gamma_diff_rad` (calculés par TrajectoryLayer)"
        " **et** les flags `_known`."
    )
    lines.append("")
    lines.append(
        "**Conséquence opérationnelle** : quand `tas_known == 0`, `tas_diff` vu par le NN"
        " est strictement `0`. Le NN n'a plus que `tas_known=0` comme drapeau pour savoir"
        " qu'il doit \"piloter à l'aveugle\" la dynamique TAS — pas de signal d'erreur,"
        " pas d'estimation de la consigne. La TAS prédite dérive selon l'apprentissage"
        " moyen sur ces régions."
    )
    lines.append("")

    # 4. Q3
    lines.append("## 4. Q3 — Stratification du bias DLH4PV par `tas_known` / `gamma_known`")
    lines.append("")
    lines.append(f"Vols utilisés (val split, A320, seed 0) : `{', '.join(flights_used)}`.")
    lines.append("")

    # Filter q3 rows for descent_shallow first.
    lines.append("### 4.1. Phase **descent_shallow** stratifiée par `tas_known` (TAS metrics)")
    lines.append("")
    lines.append("| Vol                | known | n     | MAE TAS [m/s] | bias TAS [m/s] | MAE alt [m] |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in q3_rows:
        if r["phase"] != "descent_shallow" or r["known_kind"] != "tas":
            continue
        lines.append(
            f"| {r['flight']:<18} | {r['known']:>5} | {r['n']:>5} | "
            f"{_f(r['mae_tas'])} | {_f(r['bias_tas'], '+.3f')} | {_f(r['mae_alt'], '.1f')} |"
        )
    lines.append("")

    lines.append("### 4.2. Phase **descent_shallow** stratifiée par `gamma_known` (γ metrics)")
    lines.append("")
    lines.append("| Vol                | known | n     | MAE γ [deg] | MAE alt [m] | MAE TAS [m/s] |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in q3_rows:
        if r["phase"] != "descent_shallow" or r["known_kind"] != "gamma":
            continue
        lines.append(
            f"| {r['flight']:<18} | {r['known']:>5} | {r['n']:>5} | "
            f"{_f(r['mae_gamma_deg'])} | {_f(r['mae_alt'], '.1f')} | {_f(r['mae_tas'])} |"
        )
    lines.append("")

    lines.append("### 4.3. Toutes phases — DLH4PV uniquement, stratification par `tas_known`")
    lines.append("")
    lines.append("| Phase             | known | n     | MAE TAS [m/s] | bias TAS [m/s] |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in q3_rows:
        if r["flight"] != TARGET_FLIGHT_ID or r["known_kind"] != "tas":
            continue
        lines.append(
            f"| {r['phase']:<17} | {r['known']:>5} | {r['n']:>5} | "
            f"{_f(r['mae_tas'])} | {_f(r['bias_tas'], '+.3f')} |"
        )
    lines.append("")

    # 5. Verdict
    lines.append("## 5. Verdict global")
    lines.append("")
    lines.append(_q3_verdict(q3_rows, q1_rows))
    lines.append("")

    # 6. Implications
    lines.append("## 6. Implications pour GammaNet / TasNet")
    lines.append("")
    lines.append(_implications(q1_rows, q3_rows))
    lines.append("")

    # 7. Anomalies
    lines.append("## 7. Anomalies / blocages")
    lines.append("")
    lines.append(
        "- `fdm_tas_target_known` est de dtype `Boolean` côté Delta, converti à"
        " `Float64` dans le code (le NN reçoit `{0.0, 1.0}` après cast)."
        " `fdm_gamma_target_known` est déjà `Float64`. Pas d'incohérence sémantique"
        " mais asymétrie de dtype documentée."
    )
    lines.append(
        "- `_known` étant calculé en amont (pipeline data), il ne dépend pas du modèle"
        " : ces chiffres sont stables tant que `flights.delta` n'est pas régénéré."
    )
    lines.append(
        "- L'alignement pred[i] vs true[i+1] suit la convention du trainer ; on perd"
        " 1 sample (le tout dernier) par vol — négligeable pour la stratification."
    )
    lines.append("")
    return "\n".join(lines)


def _q3_verdict(q3_rows: list[dict], q1_rows: list[dict]) -> str:
    """Produce a textual verdict from Q3 (DLH4PV descent_shallow tas-stratified)."""
    msgs: list[str] = []

    target = [
        r for r in q3_rows
        if r["flight"] == TARGET_FLIGHT_ID
        and r["phase"] == "descent_shallow"
        and r["known_kind"] == "tas"
    ]
    by_known = {r["known"]: r for r in target}
    r0 = by_known.get(0)
    r1 = by_known.get(1)

    if not r0 or not r1:
        msgs.append(
            "Données insuffisantes pour conclure sur DLH4PV : un des"
            " strates `tas_known` est vide en descent shallow."
        )
        if r0:
            msgs.append(
                f"`tas_known=0` : MAE TAS = {_f(r0['mae_tas'])} m/s, "
                f"bias = {_f(r0['bias_tas'], '+.3f')} m/s, n={r0['n']}."
            )
        if r1:
            msgs.append(
                f"`tas_known=1` : MAE TAS = {_f(r1['mae_tas'])} m/s, "
                f"bias = {_f(r1['bias_tas'], '+.3f')} m/s, n={r1['n']}."
            )
    else:
        bias0 = r0["bias_tas"]
        bias1 = r1["bias_tas"]
        mae0 = r0["mae_tas"]
        mae1 = r1["mae_tas"]
        msgs.append(
            f"DLH4PV / descent_shallow : `tas_known=0` (n={r0['n']}) bias TAS"
            f" = {_f(bias0, '+.3f')} m/s, MAE = {_f(mae0)} m/s ;"
            f" `tas_known=1` (n={r1['n']}) bias TAS = {_f(bias1, '+.3f')} m/s,"
            f" MAE = {_f(mae1)} m/s."
        )
        if r1["n"] < 10:
            msgs.append(
                "Trop peu d'échantillons `tas_known=1` (n<10) pour conclure formellement"
                " sur ce vol — verdict provisoire, à étendre aux vols supplémentaires."
            )
        elif (
            not math.isnan(bias0) and not math.isnan(bias1)
            and abs(bias0) > 5.0 and abs(bias1) < 2.5
        ):
            msgs.append(
                "**Hypothèse confirmée** : le bias est concentré sur les rows sans"
                " target (`tas_known=0`). La présence d'une consigne FMS suffit à"
                " ramener le bias dans la bande [-2.5, +2.5] m/s."
            )
        elif (
            not math.isnan(bias0) and not math.isnan(bias1)
            and abs(bias0 - bias1) < 2.0
        ):
            msgs.append(
                "**Hypothèse rejetée** : le bias est similaire dans les deux strates."
                " La cause du bias descent_shallow est ailleurs (couverture de"
                " distribution training, dynamique idle non apprise, etc.)."
            )
        else:
            msgs.append(
                "**Hypothèse partielle** : bias différent entre strates mais ni"
                " franchement nul côté `known=1` ni intégralement concentré côté"
                " `known=0`. À combiner avec les autres vols et avec le diagnostic γ."
            )

    # Aggregate across all flights for descent_shallow tas stratification.
    agg0_n = agg0_b = agg1_n = agg1_b = 0
    sum0 = sum1 = 0.0
    for r in q3_rows:
        if r["phase"] == "descent_shallow" and r["known_kind"] == "tas" and not math.isnan(r["bias_tas"]):
            if r["known"] == 0:
                agg0_n += r["n"]
                sum0 += r["bias_tas"] * r["n"]
            else:
                agg1_n += r["n"]
                sum1 += r["bias_tas"] * r["n"]
    if agg0_n > 0 and agg1_n > 0:
        msgs.append(
            f"Agrégat tous vols (descent_shallow, tas-stratifié) : "
            f"`known=0` n={agg0_n}, bias pondéré = {_f(sum0/agg0_n, '+.3f')} m/s ; "
            f"`known=1` n={agg1_n}, bias pondéré = {_f(sum1/agg1_n, '+.3f')} m/s."
        )

    return "\n\n".join(msgs)


def _implications(q1_rows: list[dict], q3_rows: list[dict]) -> str:
    """Sketch implications for GammaNet / TasNet design."""
    desc_shallow = next((r for r in q1_rows if r["phase"] == "descent_shallow"), None)
    all_row = next((r for r in q1_rows if r["phase"] == "ALL"), None)
    parts: list[str] = []

    if desc_shallow and desc_shallow["n"] > 0:
        parts.append(
            "**Population concernée** : en descent_shallow, "
            f"{_pct(desc_shallow['tas_unknown'])} des samples ont `tas_known=0` "
            f"(n={int(desc_shallow['n'] * desc_shallow['tas_unknown']):,} sur"
            f" {desc_shallow['n']:,}). C'est exactement la zone où le bias post-F3 a"
            " été mesuré à +10.73 m/s : un TasNet capable de **prédire ou inférer**"
            " le `tas_target` quand il est manquant adresse cette population à 100 %."
        )
    if all_row:
        parts.append(
            f"**Champ d'application global** : sur l'ensemble du dataset A320 train,"
            f" {_pct(all_row['tas_unknown'])} des rows sont `tas_known=0`,"
            f" {_pct(all_row['gamma_unknown'])} sont `gamma_known=0`,"
            f" {_pct(all_row['any_unknown'])} ont au moins une cible manquante."
            " Un TasNet/GammaNet adresserait ces rows ; les autres restent pilotées"
            " par les consignes existantes."
        )

    parts.append(
        "**Conditionnellement à la confirmation Q3** : si l'hypothèse \"bias = absence"
        " de target\" est validée, l'architecture pertinente est un *consigne synthétique*"
        " activée par `1 - tas_known` :"
    )
    parts.append(
        "- TasNet : prédire `tas_target_synth` à partir de `(alt, gamma, alt_diff,"
        " mach, cas, alt_target, regime, era_temp_K, wind, masse_proxy)` ;"
        " injecter dans `tas_diff = (1 - known) * (target_synth - tas) + known *"
        " (target_real - tas)`."
    )
    parts.append(
        "- GammaNet : symétrique pour γ, sortant un `gamma_target_synth` activé par"
        " `1 - gamma_known`. Particulièrement utile si la table Q1 montre que"
        " `gamma_known=0` corrèle aussi avec un MAE γ dégradé en descent shallow."
    )
    parts.append(
        "**Si l'hypothèse est rejetée**, il faut chercher la cause du bias ailleurs : "
        " (i) couverture en distribution du training set en descent shallow réelle"
        " (cf. `dataset_regime_stats.md` § saturation breakdown), (ii) régularisation"
        " du `n_z_residual` cap qui borne `dTAS/dt` à `g·sin(γ)` plus la résidue NN —"
        " un cap trop bas empêche la décélération naturelle d'un descent idle, (iii)"
        " mismatch ERA5 vent / température sur ce vol."
    )
    return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main() -> int:
    print("[Q1] Computing per-phase target-known table on A320 train split...")
    q1_rows = compute_phase_known_table(DELTA_PATH)
    print("[Q1] done — phases:", [(r["phase"], r["n"]) for r in q1_rows])

    print("[Q3] Running stratified inference on DLH4PV + extras...")
    q3_rows, flights_used = run_q3(extra_flights=4)
    print(f"[Q3] done — {len(q3_rows)} stratified rows across {len(flights_used)} flights")

    md = render_markdown(q1_rows, q3_rows, flights_used)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print(f"Report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
