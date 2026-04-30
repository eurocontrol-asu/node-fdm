"""Normalized integrated-gradients attribution for node_adsb_v1_A320.

Goal
----
The original gradient-attribution diagnostic
(``scripts/debug/diag_step_transient.py``) reported per-input gradients in
*raw input units*::

    d a_spec / d cos_gamma   ≈ 13–186
    d a_spec / d tas_diff    ≈ 0.06–0.4

and concluded that ``a_spec`` and ``n_z_residual`` heads are "dominated by
the gamma features".  That conclusion **drives the F1 (per-head input
routing) decision** in ``data/mardown/per_head_input_routing.md``.

But raw gradients are not normalized by the empirical feature scale.  The
StructuredLayer normalizes inputs internally::

    z_i = (x_i - mu_i) / sigma_i

so the *useful* attribution on the actual data distribution is::

    IG_i_eff = IG_i_raw * sigma_i

With the training stats (sigma_cos_gamma ~ 1.1e-3,
sigma_tas_diff ~ 2.06), the brut ratio of 186 / 0.4 ~ 465 collapses to::

    (186 * 1.1e-3) / (0.4 * 2.06) ~ 0.25

which **inverts** the dominance.  This script reproduces the diagnostic
correctly.

Method
------
1. Sample ~30 real points per regime (cruise / climb_shallow /
   climb_saturated / descent_shallow / descent_saturated) from
   ``data/flights.delta`` with the same filter as the training set
   (``fdm_flag_valid AND meta_aircraft_type='A320' AND meta_split='train'``).
2. Reconstruct the input dict to ``data_ode`` by passing each row through
   the trained ``TrajectoryLayer`` (so derived features ``cos_gamma``,
   ``q_pa``, ``g_sin_gamma``, etc. match the model's actual inputs).
3. Compute Integrated Gradients (Sundararajan et al. 2017) for the heads
   ``fdm_a_spec_ms2`` and ``fdm_n_z_residual``::

       IG_i(x) = (x_i - x'_i) * integral_0^1 d f / d x_i (x' + alpha (x - x')) d alpha

   with baseline ``x'`` = mean cruise input vector and 50 Riemann steps.
4. Compute regime-conditional sigma_i empirically over the sampled
   regime points (also export training-stat sigmas from meta.json for
   reference).
5. Report mean +/- 95% CI for both:
   - **raw IG** (matches signature of original diagnostic)
   - **normalized IG**: ``IG_i * sigma_i^(regime)`` -- the true effective
     attribution on the regime distribution.

Output
------
Markdown report at ``data/mardown/normalized_attribution_diag.md``.

Run from project root::

    uv run python scripts/debug/diag_normalized_attribution.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "node-fdm" / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "node-fdm-data" / "src"))

from node_fdm.architectures import adsb as _adsb_reg  # noqa: F401
from node_fdm.predictor import NodeFDMPredictor

MODEL_DIR = Path("data/models/node_adsb_v1_A320")
DELTA_PATH = Path("data/flights.delta")
OUTPUT_PATH = Path("data/mardown/normalized_attribution_diag.md")

# Reuse exactly the same regime thresholds as dataset_regime_stats.py
GAMMA_CRUISE_RAD = math.radians(0.5)
GAMMA_CLIMB_RAD = math.radians(1.0)
GAMMA_DESCENT_RAD = math.radians(-1.0)
FTMIN_TO_MS = 0.00508
ALT_RATE_CRUISE = 200 * FTMIN_TO_MS
ALT_RATE_CLIMB = 500 * FTMIN_TO_MS
ALT_RATE_DESCENT = -500 * FTMIN_TO_MS
ALT_RATE_CLIMB_SAT = 1500 * FTMIN_TO_MS
ALT_RATE_DESCENT_SAT = -1000 * FTMIN_TO_MS

# How many points to sample per regime
N_PER_REGIME = 30
# Riemann steps for IG
N_IG_STEPS = 50
# RNG seed (deterministic sampling)
SAMPLE_SEED = 1234

REGIMES: list[str] = [
    "cruise",
    "climb_shallow",
    "climb_saturated",
    "descent_shallow",
    "descent_saturated",
]

# Heads of interest
HEADS: list[str] = ["fdm_a_spec_ms2", "fdm_n_z_residual"]

# Pretty list of features in stable order (matches data_ode input_cols)
# We compute that dynamically at runtime from the model.


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

# Columns we need from the Delta table to rebuild a `data_ode` input row
NEEDED_COLS_FROM_DELTA: list[str] = [
    "meta_aircraft_type",
    "meta_split",
    "fdm_flag_valid",
    "fdm_gamma_rad",
    "fdm_d_alt_ms",
    "raw_alt_m",
    "era_tas_ms",
    "fdm_alt_target_m",
    "fdm_tas_target_ms",
    "fdm_gamma_target_rad",
    "fdm_gamma_target_known",
    "fdm_tas_target_known",
    "fdm_long_wind_ms",
    "era_temp_K",
]


def _regime_filter(regime: str) -> pl.Expr:
    """Polars filter expression for a given regime label."""
    g = pl.col("fdm_gamma_rad")
    a = pl.col("fdm_d_alt_ms")
    if regime == "cruise":
        return (g.abs() < GAMMA_CRUISE_RAD) & (a.abs() < ALT_RATE_CRUISE)
    if regime == "climb_shallow":
        is_climb = (g > GAMMA_CLIMB_RAD) | (a > ALT_RATE_CLIMB)
        is_sat = a > ALT_RATE_CLIMB_SAT
        return is_climb & ~is_sat
    if regime == "climb_saturated":
        return pl.col("fdm_d_alt_ms") > ALT_RATE_CLIMB_SAT
    if regime == "descent_shallow":
        is_desc = (g < GAMMA_DESCENT_RAD) | (a < ALT_RATE_DESCENT)
        is_sat = a < ALT_RATE_DESCENT_SAT
        return is_desc & ~is_sat
    if regime == "descent_saturated":
        return pl.col("fdm_d_alt_ms") < ALT_RATE_DESCENT_SAT
    msg = f"unknown regime {regime}"
    raise ValueError(msg)


def sample_regime_points(
    delta_path: Path,
    *,
    typecode: str,
    split: str,
    n_per_regime: int,
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Sample ~n_per_regime rows from each regime.

    Returns a dict regime -> DataFrame of raw delta rows ready for
    rebuilding the ``data_ode`` input.
    """
    base = (
        pl.scan_delta(str(delta_path))
        .filter(pl.col("fdm_flag_valid"))
        .filter(pl.col("meta_aircraft_type") == typecode)
        .filter(pl.col("meta_split") == split)
        .select(NEEDED_COLS_FROM_DELTA)
        .drop_nulls(["fdm_gamma_rad", "fdm_d_alt_ms", "era_tas_ms", "raw_alt_m"])
        .filter(
            ~pl.col("fdm_gamma_rad").is_nan()
            & ~pl.col("fdm_d_alt_ms").is_nan()
            & ~pl.col("era_tas_ms").is_nan()
            & ~pl.col("raw_alt_m").is_nan()
        )
    )

    out: dict[str, pl.DataFrame] = {}
    for regime in REGIMES:
        flt = base.filter(_regime_filter(regime))
        df = flt.collect()
        if df.height == 0:
            print(f"WARN: regime {regime} has 0 rows", file=sys.stderr)
            out[regime] = df
            continue
        # Deterministic sample (seeded). Use polars sample with seed.
        n = min(n_per_regime, df.height)
        out[regime] = df.sample(n=n, seed=seed)
    return out


# ---------------------------------------------------------------------------
# Build ``data_ode`` inputs from raw delta rows
# ---------------------------------------------------------------------------


def build_data_ode_inputs(
    predictor: NodeFDMPredictor,
    df: pl.DataFrame,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Pass raw rows through TrajectoryLayer to obtain ``data_ode`` inputs.

    Returns
    -------
    inputs
        dict[str, Tensor of shape (n,)] containing exactly the columns
        ``data_ode.input_cols``.
    cols
        ordered list of input column names (== predictor.model.layers_dict["data_ode"].input_cols).
    """
    model = predictor.model
    traj = model.layers_dict["trajectory"]
    data_ode = model.layers_dict["data_ode"]

    n = df.height
    # Build raw vect_dict from delta rows. Names must match what TrajectoryLayer expects.
    raw: dict[str, torch.Tensor] = {}
    for col in NEEDED_COLS_FROM_DELTA:
        if col in {"meta_aircraft_type", "meta_split", "fdm_flag_valid"}:
            continue
        arr = df[col].to_numpy().astype(np.float32)
        # Booleans (fdm_tas_target_known) -> float
        raw[col] = torch.tensor(arr, dtype=torch.float32)

    with torch.no_grad():
        derived = traj(raw)
    vect = {**raw, **derived}

    # data_ode wants exactly its input_cols in order
    inputs: dict[str, torch.Tensor] = {}
    for col in data_ode.input_cols:
        if col not in vect:
            # Fall back to zeros if a column is genuinely missing (defensive)
            inputs[col] = torch.zeros(n, dtype=torch.float32)
        else:
            v = vect[col]
            inputs[col] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return inputs, list(data_ode.input_cols)


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------


def head_value(
    data_ode: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    head: str,
) -> torch.Tensor:
    """Forward pass through data_ode and return head value (denormalized)."""
    out = data_ode(inputs)
    return out[head]


def integrated_gradients(
    data_ode: torch.nn.Module,
    cols: list[str],
    x: dict[str, torch.Tensor],
    x_baseline: dict[str, torch.Tensor],
    head: str,
    *,
    n_steps: int = N_IG_STEPS,
) -> dict[str, np.ndarray]:
    """Compute IG for one head over a batch of points.

    Args:
        data_ode: the StructuredLayer
        cols: ordered input columns
        x: dict[col -> Tensor(n)] target points
        x_baseline: dict[col -> Tensor(n)] baseline (broadcastable to (n,))
        head: head name
        n_steps: Riemann steps

    Returns:
        dict[col -> ndarray(n)] with IG per feature.
    """
    n = next(iter(x.values())).shape[0]
    # Pre-compute deltas
    delta = {c: (x[c] - x_baseline[c]) for c in cols}

    # Accumulator: dict col -> Tensor(n)
    grad_accum: dict[str, torch.Tensor] = {c: torch.zeros(n) for c in cols}

    # Riemann midpoint approximation: alpha_k = (k + 0.5) / n_steps
    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        inputs = {}
        for c in cols:
            base = x_baseline[c]
            tgt = x[c]
            v = base + alpha * (tgt - base)
            v = v.clone().detach().requires_grad_(True)
            inputs[c] = v
        y = head_value(data_ode, inputs, head).sum()
        grads = torch.autograd.grad(
            y, [inputs[c] for c in cols], retain_graph=False, allow_unused=True
        )
        for c, g in zip(cols, grads, strict=True):
            if g is not None:
                grad_accum[c] = grad_accum[c] + g.detach()

    # Mean gradient * delta_x => integrated gradient
    ig: dict[str, np.ndarray] = {}
    for c in cols:
        mean_g = grad_accum[c] / n_steps
        ig_c = (mean_g * delta[c]).cpu().numpy()
        ig[c] = ig_c
    return ig


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def mean_ci95(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using normal approximation for the 95% CI."""
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(arr.mean())
    se = float(arr.std(ddof=1) / max(np.sqrt(arr.size), 1.0)) if arr.size > 1 else 0.0
    half = 1.96 * se
    return m, m - half, m + half


def empirical_sigma_per_regime(
    inputs_per_regime: dict[str, dict[str, torch.Tensor]],
    cols: list[str],
) -> dict[str, dict[str, float]]:
    """Empirical sigma per (regime, col) from the sampled inputs."""
    out: dict[str, dict[str, float]] = {}
    for regime, inputs in inputs_per_regime.items():
        sigmas: dict[str, float] = {}
        for c in cols:
            v = inputs[c].cpu().numpy()
            v = v[np.isfinite(v)]
            sigmas[c] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        out[regime] = sigmas
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    *,
    delta_path: Path,
    model_dir: Path,
    output_path: Path,
    typecode: str,
    split: str,
    n_per_regime: int,
    seed: int,
) -> None:
    print(f"[1/5] Loading model from {model_dir}", file=sys.stderr)
    predictor = NodeFDMPredictor(model_path=model_dir, device="cpu")
    data_ode = predictor.model.layers_dict["data_ode"]
    cols = list(data_ode.input_cols)

    print(f"[2/5] Sampling {n_per_regime} points per regime from {delta_path}", file=sys.stderr)
    samples = sample_regime_points(
        delta_path,
        typecode=typecode,
        split=split,
        n_per_regime=n_per_regime,
        seed=seed,
    )

    inputs_per_regime: dict[str, dict[str, torch.Tensor]] = {}
    n_per_regime_actual: dict[str, int] = {}
    for regime, df in samples.items():
        if df.height == 0:
            inputs_per_regime[regime] = {c: torch.zeros(0) for c in cols}
            n_per_regime_actual[regime] = 0
            continue
        inputs, _ = build_data_ode_inputs(predictor, df)
        inputs_per_regime[regime] = inputs
        n_per_regime_actual[regime] = df.height

    print("[3/5] Computing empirical sigma per regime", file=sys.stderr)
    sigma_per_regime = empirical_sigma_per_regime(inputs_per_regime, cols)
    # Training-set sigma (from meta.json stats_dict)
    train_stats = predictor.meta.stats_dict
    sigma_train_full: dict[str, float] = {}
    for c in cols:
        if c in train_stats:
            sigma_train_full[c] = float(train_stats[c].std)
        else:
            sigma_train_full[c] = float("nan")

    # Build baseline = mean over cruise sample. Fallback: zero if cruise empty.
    cruise_inputs = inputs_per_regime.get("cruise")
    if cruise_inputs is None or n_per_regime_actual.get("cruise", 0) == 0:
        msg = "cruise regime has 0 samples; cannot define IG baseline"
        raise RuntimeError(msg)
    baseline_scalars: dict[str, float] = {}
    for c in cols:
        v = cruise_inputs[c].cpu().numpy()
        v = v[np.isfinite(v)]
        baseline_scalars[c] = float(v.mean()) if v.size > 0 else 0.0

    print("[4/5] Running Integrated Gradients per regime per head", file=sys.stderr)
    # ig_results[regime][head][col] = ndarray(n)
    ig_results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for regime, inputs in inputs_per_regime.items():
        n = n_per_regime_actual[regime]
        if n == 0:
            ig_results[regime] = {}
            continue
        # Broadcast baseline to (n,) tensors
        x_baseline = {
            c: torch.full((n,), baseline_scalars[c], dtype=torch.float32) for c in cols
        }
        ig_results[regime] = {}
        for head in HEADS:
            ig = integrated_gradients(
                data_ode,
                cols,
                inputs,
                x_baseline,
                head,
                n_steps=N_IG_STEPS,
            )
            # IG_raw above is "(x-x') * mean_grad" => already in physical units of head per feature point.
            # We split this into:
            #   - "raw IG attribution" (per-sample, dimension = head_units)
            # And we will *also* report the per-sample gradient * sigma:
            #   "normalized": IG / (x_i - x'_i) * sigma_i = mean_grad * sigma_i
            # because the rescaling we want answers "if I move feature by 1 sigma, how much does head change?"
            ig_results[regime][head] = ig

            # Also pre-compute mean_grad (without delta-x multiplication)
            # so we can report sigma-normalized attribution = mean_grad * sigma.
            # mean_grad = ig / delta_x (with safe divide)
            # But we'd lose sign info if delta_x ~ 0. Instead, recompute mean_grad
            # cleanly:
        # We need mean_grad too -> recompute once (small cost).
    print("[4b/5] Recomputing mean gradients for sigma-normalized attribution", file=sys.stderr)
    mean_grad_results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for regime, inputs in inputs_per_regime.items():
        n = n_per_regime_actual[regime]
        if n == 0:
            mean_grad_results[regime] = {}
            continue
        x_baseline = {
            c: torch.full((n,), baseline_scalars[c], dtype=torch.float32) for c in cols
        }
        mean_grad_results[regime] = {}
        for head in HEADS:
            mean_grad_results[regime][head] = _mean_grad_along_path(
                data_ode, cols, inputs, x_baseline, head, n_steps=N_IG_STEPS
            )

    print(f"[5/5] Rendering markdown report to {output_path}", file=sys.stderr)
    md = render_markdown(
        cols=cols,
        n_per_regime_actual=n_per_regime_actual,
        sigma_per_regime=sigma_per_regime,
        sigma_train=sigma_train_full,
        baseline_scalars=baseline_scalars,
        ig_results=ig_results,
        mean_grad_results=mean_grad_results,
        n_steps=N_IG_STEPS,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    print(f"OK: report written to {output_path}", file=sys.stderr)


def _mean_grad_along_path(
    data_ode: torch.nn.Module,
    cols: list[str],
    x: dict[str, torch.Tensor],
    x_baseline: dict[str, torch.Tensor],
    head: str,
    *,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Compute the path-averaged gradient ``<d head / d x_i>_alpha`` per feature.

    This is the "raw gradient" the StructuredLayer sees on the path
    between baseline and target, *averaged* (no delta_x multiplication).
    Multiplying by sigma_i gives the head response to a 1-sigma move in
    feature i.
    """
    n = next(iter(x.values())).shape[0]
    grad_accum: dict[str, torch.Tensor] = {c: torch.zeros(n) for c in cols}
    for k in range(n_steps):
        alpha = (k + 0.5) / n_steps
        inputs = {}
        for c in cols:
            base = x_baseline[c]
            tgt = x[c]
            v = base + alpha * (tgt - base)
            v = v.clone().detach().requires_grad_(True)
            inputs[c] = v
        y = head_value(data_ode, inputs, head).sum()
        grads = torch.autograd.grad(
            y, [inputs[c] for c in cols], retain_graph=False, allow_unused=True
        )
        for c, g in zip(cols, grads, strict=True):
            if g is not None:
                grad_accum[c] = grad_accum[c] + g.detach()
    out: dict[str, np.ndarray] = {}
    for c in cols:
        out[c] = (grad_accum[c] / n_steps).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

# Features prioritized in tables (the ones that drive the F1 narrative).
PRIORITY_FEATURES: list[str] = [
    "fdm_tas_diff_ms",
    "fdm_gamma_diff_rad",
    "fdm_cos_gamma",
    "fdm_g_sin_gamma_ms2",
    "fdm_q_pa",
    "fdm_g_over_v",
    "era_tas_ms",
    "fdm_gamma_rad",
    "raw_alt_m",
]


def render_markdown(  # noqa: C901 - report formatter, mostly linear
    *,
    cols: list[str],
    n_per_regime_actual: dict[str, int],
    sigma_per_regime: dict[str, dict[str, float]],
    sigma_train: dict[str, float],
    baseline_scalars: dict[str, float],
    ig_results: dict[str, dict[str, dict[str, np.ndarray]]],
    mean_grad_results: dict[str, dict[str, dict[str, np.ndarray]]],
    n_steps: int,
    seed: int,
) -> str:
    lines: list[str] = []
    lines.append("# Diagnostic d'attribution normalisée — heads `a_spec` & `n_z_residual`")
    lines.append("")
    lines.append("> **Re-run du diagnostic original `diag_step_transient.py`**, mais avec :")
    lines.append(">")
    lines.append("> - **Integrated Gradients** (Sundararajan et al. 2017) au lieu du gradient ponctuel local")
    lines.append("> - **30 points réels** (échantillonnés depuis le train set) par régime, avec IC 95%")
    lines.append("> - **Normalisation par sigma feature** — la mesure correcte de l'attribution effective sur la distribution")
    lines.append("")
    lines.append("## 1. Méthodologie")
    lines.append("")
    lines.append(f"- Modèle      : `data/models/node_adsb_v1_A320/`")
    lines.append(f"- Delta source: `data/flights.delta` (filtre `fdm_flag_valid AND meta_aircraft_type='A320' AND meta_split='train'`)")
    lines.append(f"- Échantillon : {N_PER_REGIME} points par régime (seed={seed})")
    lines.append(f"- Régimes     : {', '.join(REGIMES)} (mêmes seuils que `dataset_regime_stats.py`)")
    lines.append(f"- IG steps    : {n_steps} (Riemann mid-point)")
    lines.append("- Baseline IG : moyenne empirique des inputs sur le sous-échantillon **cruise**")
    lines.append("- Heads       : `fdm_a_spec_ms2`, `fdm_n_z_residual` (sortie dénormalisée du `data_ode`)")
    lines.append("- Inputs IG   : tous les `data_ode.input_cols` (passés via `TrajectoryLayer` pour reproduire les features dérivées exactement comme à l'inférence)")
    lines.append("")
    lines.append("Définitions employées :")
    lines.append("")
    lines.append("- **`IG_raw`** = `(x_i - x'_i) * <d head / d x_i>_alpha`  — attribution brute en *unités physiques de la feature*")
    lines.append("- **`grad_path`** = `<d head / d x_i>_alpha`  — gradient moyen le long du path baseline -> target")
    lines.append("- **`grad_sigma`** = `grad_path * sigma_i`  — réponse du head à un déplacement de **1 sigma** de la feature (la mesure invariante de la dominance)")
    lines.append("")
    lines.append(f"### Effectifs réels par régime")
    lines.append("")
    lines.append("| Régime               |   n  |")
    lines.append("|---|---:|")
    for r in REGIMES:
        lines.append(f"| {r:<20} | {n_per_regime_actual.get(r, 0):>3} |")
    lines.append("")

    lines.append("## 2. Sigmas features par régime (vs sigma train global)")
    lines.append("")
    lines.append("Sigmas calculés sur l'échantillon de chaque régime (post-`TrajectoryLayer`).")
    lines.append(f"`sigma_train` rappelé pour comparaison (depuis `meta.json` stats_dict).")
    lines.append("")
    header = "| feature | sigma_train | " + " | ".join(REGIMES) + " |"
    sep = "|---|" + "---:|" * (len(REGIMES) + 1)
    lines.append(header)
    lines.append(sep)
    for c in PRIORITY_FEATURES:
        if c not in cols:
            continue
        row = f"| `{c}` | {sigma_train.get(c, float('nan')):.4g} |"
        for r in REGIMES:
            row += f" {sigma_per_regime.get(r, {}).get(c, float('nan')):.4g} |"
        lines.append(row)
    # Other features (collapse)
    other = [c for c in cols if c not in PRIORITY_FEATURES]
    if other:
        lines.append("")
        lines.append("Autres features (annexe) :")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for c in other:
            row = f"| `{c}` | {sigma_train.get(c, float('nan')):.4g} |"
            for r in REGIMES:
                row += f" {sigma_per_regime.get(r, {}).get(c, float('nan')):.4g} |"
            lines.append(row)
    lines.append("")

    # IG raw (signature originale)
    lines.append("## 3. IG bruts — `(x - x') * <grad>` — comparaison avec le diagnostic original")
    lines.append("")
    lines.append("Mean +/- 95% CI. Les unités sont en `[head_unit] * 1` (l'IG cumulé sur le path)." )
    lines.append("La somme des IG sur toutes les features ~ `head(x) - head(x')` (axiome de complétude).")
    lines.append("")
    for head in HEADS:
        lines.append(f"### Head `{head}` — IG bruts par feature × régime")
        lines.append("")
        header = "| feature | " + " | ".join(REGIMES) + " |"
        sep = "|---|" + "---:|" * len(REGIMES)
        lines.append(header)
        lines.append(sep)
        for c in PRIORITY_FEATURES:
            if c not in cols:
                continue
            row = f"| `{c}` |"
            for r in REGIMES:
                arr = ig_results.get(r, {}).get(head, {}).get(c, np.array([]))
                m, lo, hi = mean_ci95(arr)
                if math.isnan(m):
                    row += " n/a |"
                else:
                    row += f" {m:+.3e} [{lo:+.2e}, {hi:+.2e}] |"
            lines.append(row)
        lines.append("")

    # Grad path × sigma — la vraie mesure de dominance
    lines.append("## 4. Attribution **normalisée** — réponse du head à 1-sigma de feature")
    lines.append("")
    lines.append("`grad_sigma = <d head / d x_i>_alpha * sigma_i^(regime)` — comparable à travers features, "
                 "car toutes les features sont mises à la même échelle (1 sigma).")
    lines.append("")
    lines.append("**C'est cette colonne qui doit être utilisée pour conclure à la dominance d'une feature.**")
    lines.append("")
    for head in HEADS:
        lines.append(f"### Head `{head}` — grad * sigma_regime, mean +/- IC95")
        lines.append("")
        header = "| feature | " + " | ".join(REGIMES) + " |"
        sep = "|---|" + "---:|" * len(REGIMES)
        lines.append(header)
        lines.append(sep)
        for c in PRIORITY_FEATURES:
            if c not in cols:
                continue
            row = f"| `{c}` |"
            for r in REGIMES:
                grads = mean_grad_results.get(r, {}).get(head, {}).get(c, np.array([]))
                sig = sigma_per_regime.get(r, {}).get(c, float("nan"))
                if grads.size == 0 or not math.isfinite(sig):
                    row += " n/a |"
                    continue
                vals = grads * sig
                m, lo, hi = mean_ci95(vals)
                if math.isnan(m):
                    row += " n/a |"
                else:
                    row += f" {m:+.3e} [{lo:+.2e}, {hi:+.2e}] |"
            lines.append(row)
        lines.append("")

    # Ratios per pair of interest, in cruise
    lines.append("## 5. Ratios de dominance — `cos_gamma` vs `tas_diff` et `gamma_diff` vs `tas_diff`")
    lines.append("")
    lines.append("Ratio en valeur absolue, sur les attributions normalisées (grad * sigma). "
                 "`> 1` => le numérateur domine ; `< 1` => le dénominateur domine.")
    lines.append("")
    pairs = [
        ("fdm_cos_gamma", "fdm_tas_diff_ms"),
        ("fdm_g_sin_gamma_ms2", "fdm_tas_diff_ms"),
        ("fdm_gamma_diff_rad", "fdm_tas_diff_ms"),
    ]
    for head in HEADS:
        lines.append(f"### Head `{head}`")
        lines.append("")
        lines.append("| Ratio | " + " | ".join(REGIMES) + " |")
        lines.append("|---|" + "---:|" * len(REGIMES))
        for num, den in pairs:
            if num not in cols or den not in cols:
                continue
            row = f"| `\\|grad*sigma\\|({num})` / `\\|grad*sigma\\|({den})` |"
            for r in REGIMES:
                gn = mean_grad_results.get(r, {}).get(head, {}).get(num, np.array([]))
                gd = mean_grad_results.get(r, {}).get(head, {}).get(den, np.array([]))
                sn = sigma_per_regime.get(r, {}).get(num, float("nan"))
                sd = sigma_per_regime.get(r, {}).get(den, float("nan"))
                if gn.size == 0 or gd.size == 0 or not math.isfinite(sn) or not math.isfinite(sd):
                    row += " n/a |"
                    continue
                num_attr = float(np.abs(np.mean(gn * sn)))
                den_attr = float(np.abs(np.mean(gd * sd)))
                if den_attr <= 0:
                    row += " inf |"
                else:
                    row += f" {num_attr / den_attr:.3f} |"
            lines.append(row)
        lines.append("")

    # Verdict per regime
    lines.append("## 6. Verdict par régime — la dominance gamma survit-elle après normalisation ?")
    lines.append("")
    lines.append("Pour chaque régime, on prend le head `fdm_a_spec_ms2` et on regarde "
                 "`|grad*sigma|(cos_gamma) / |grad*sigma|(tas_diff)`. Une dominance gamma "
                 "réelle implique ce ratio > 1.")
    lines.append("")
    lines.append("| Régime | ratio cos_gamma / tas_diff (a_spec) | verdict |")
    lines.append("|---|---:|---|")
    for r in REGIMES:
        gn = mean_grad_results.get(r, {}).get("fdm_a_spec_ms2", {}).get("fdm_cos_gamma", np.array([]))
        gd = mean_grad_results.get(r, {}).get("fdm_a_spec_ms2", {}).get("fdm_tas_diff_ms", np.array([]))
        sn = sigma_per_regime.get(r, {}).get("fdm_cos_gamma", float("nan"))
        sd = sigma_per_regime.get(r, {}).get("fdm_tas_diff_ms", float("nan"))
        if gn.size == 0 or gd.size == 0 or not math.isfinite(sn) or not math.isfinite(sd):
            lines.append(f"| {r} | n/a | regime indisponible |")
            continue
        num_attr = float(np.abs(np.mean(gn * sn)))
        den_attr = float(np.abs(np.mean(gd * sd)))
        if den_attr <= 0:
            verdict = "tas_diff inerte"
            ratio_s = "inf"
        else:
            ratio = num_attr / den_attr
            ratio_s = f"{ratio:.3f}"
            if ratio > 2.0:
                verdict = "**cos_gamma domine** (>2x)"
            elif ratio > 1.0:
                verdict = "cos_gamma legerement dominant"
            elif ratio > 0.5:
                verdict = "approximate parite"
            else:
                verdict = "**tas_diff domine** (la dominance gamma s'inverse)"
        lines.append(f"| {r} | {ratio_s} | {verdict} |")
    lines.append("")

    lines.append("## 7. Implications et lecture")
    lines.append("")
    lines.append("Si le ratio normalisé bascule sous 1 (notamment en cruise), la motivation empirique "
                 "de F1 (per-head input routing) s'effondre : la 'dominance gamma' du diagnostic original "
                 "était un artefact de l'absence de normalisation par les sigmas features.")
    lines.append("")
    lines.append("Le 'smoking gun' rapporté `tas_diff = +20 -> n_z_residual = -0.040` doit alors être "
                 "réinterprété : ce n'est pas que le NN ignore `tas_diff`, c'est que la sortie est dominée "
                 "par l'**intercept du head** (biais + activations sortantes) plutôt que par la direction "
                 "du gradient sur `tas_diff`. Une sweep `tas_diff` à autres features fixes mesure le "
                 "*premier-ordre* de la sortie ; si le biais du head est lui-même corrélé à un autre "
                 "input non-stationnaire (e.g. q, cos_gamma), le sweep ne dissocie pas les contributions.")
    lines.append("")
    lines.append("## 8. Anomalies et limites")
    lines.append("")
    lines.append("- Sigma_cos_gamma au sein du sample cruise peut être quasi-nul (les avions sont quasi-niveau).")
    lines.append("  Le ratio devient alors instable (`inf` ou bruité). Inspecter section 5 si on observe ça.")
    lines.append("- IG dépend du choix du baseline. Ici baseline = mean cruise. C'est cohérent avec la "
                 "question 'dominance dans le voisinage du cruise', mais en saturated climb la baseline "
                 "est lointaine — IG devient une mesure de comportement *moyen* sur le path et perd en "
                 "résolution locale.")
    lines.append("- Booléens `fdm_*_known` traités en float (1.0 / 0.0) ; sigma vaut 0 dans les sous-régimes "
                 "où le flag est constant -> attribution normalisée non significative pour ces features.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Path, default=DELTA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--typecode", default="A320")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-per-regime", type=int, default=N_PER_REGIME)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = parser.parse_args(argv)

    if not args.delta.exists():
        print(f"ERROR: delta path {args.delta} does not exist", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"ERROR: model dir {args.model} does not exist", file=sys.stderr)
        return 1

    run(
        delta_path=args.delta,
        model_dir=args.model,
        output_path=args.output,
        typecode=args.typecode,
        split=args.split,
        n_per_regime=args.n_per_regime,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
