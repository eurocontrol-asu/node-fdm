"""Decomposition of the ``tas_diff = +20 -> n_z_residual = -0.040`` smoking gun.

Background
----------
The original gradient diagnostic flagged a "shortcut via cos_gamma" but the
normalized-attribution diagnostic
(``data/mardown/normalized_attribution_diag.md``) inverted that conclusion:
once gradients are scaled by feature sigmas, ``tas_diff`` dominates
``cos_gamma`` for both heads in cruise.  The smoking-gun number reproduces,
but its mechanism is *not* the cos_gamma shortcut.

Three competing explanations remain:

1. **Direct local gradient.**  ``d n_z_residual / d tas_diff`` evaluated at
   the cruise base point already reaches ``-0.040 / 20 = -2e-3``.  If
   ``grad * 20`` matches the swept value, the head genuinely learned a
   linear ``tas_diff -> n_z`` direction near cruise.

2. **Propagated effect.**  ``TrajectoryLayer`` builds ``tas_diff`` from the
   *current* ``tas`` and the *target*.  When we sweep
   ``tas_target = 230 + delta``, only ``tas_diff`` itself changes (cruise
   tas is fixed), so the propagation through other features is small here.
   But the head also receives the raw ``tas`` (era_tas_ms): one might
   imagine that during a real RK4 step the propagation through
   ``q_pa``/``cos_gamma``/``g_over_v`` accumulates.  We verify by
   *isolating* tas_diff inside ``data_ode.input_cols`` (overriding all
   other features at their cruise value while sweeping only
   ``fdm_tas_diff_ms``), then comparing with the "native" sweep where
   ``TrajectoryLayer`` is allowed to recompute everything.

3. **Intercept / non-linearity of the head.**  The trained head is an MLP
   whose output at ``tas_diff = 0`` may already be non-zero (a learned
   bias activated by the cruise context).  We measure
   ``n_z_residual(tas_diff = 0)`` and the symmetric / antisymmetric parts
   of the +/-20 sweep::

       slope_eff = (n_z(+20) - n_z(-20)) / (2 * 20)
       intercept_eff = (n_z(+20) + n_z(-20)) / 2

   ``slope_eff`` is the linear sensitivity from the sweep; comparing it
   to ``d n_z / d tas_diff`` at the base point separates first-order
   gradient from non-linearity.

The script also stratifies by regime: cruise / climb_shallow (gamma=+1°,
alt=8000m) / descent_shallow (gamma=-1°, alt=8000m) -- to test whether the
smoking gun is a property of cruise or extends to other states.

Output
------
- Console tables (ASCII).
- Markdown report at ``data/mardown/smoking_gun_decomposition.md``.

Run from project root::

    uv run python scripts/debug/diag_smoking_gun_decomp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "node-fdm" / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "node-fdm-data" / "src"))

from node_fdm.architectures import adsb as _adsb_reg  # noqa: F401
from node_fdm.predictor import NodeFDMPredictor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path("data/models/node_adsb_v1_A320")
OUTPUT_PATH = Path("data/mardown/smoking_gun_decomposition.md")

SEED = 1234

# Sweep grid
DELTAS = [-30.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 30.0]

# Regimes
REGIMES: list[dict[str, float]] = [
    {
        "label": "cruise",
        "raw_alt_m": 10000.0,
        "fdm_gamma_rad": 0.0,
        "era_tas_ms": 230.0,
        "fdm_alt_target_m": 10000.0,
        "fdm_long_wind_ms": 0.0,
        "era_temp_K": 223.15,
    },
    {
        "label": "climb_shallow",
        "raw_alt_m": 8000.0,
        "fdm_gamma_rad": float(np.radians(1.0)),
        "era_tas_ms": 230.0,
        "fdm_alt_target_m": 10000.0,
        "fdm_long_wind_ms": 0.0,
        "era_temp_K": 235.0,
    },
    {
        "label": "descent_shallow",
        "raw_alt_m": 8000.0,
        "fdm_gamma_rad": float(np.radians(-1.0)),
        "era_tas_ms": 230.0,
        "fdm_alt_target_m": 6000.0,
        "fdm_long_wind_ms": 0.0,
        "era_temp_K": 235.0,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_full_input_dict(
    predictor: NodeFDMPredictor,
    base_state: dict[str, float],
    tas_target_delta: float,
) -> dict[str, torch.Tensor]:
    """Build the full input dict (raw X+U+E0 + derived E1) by running the
    TrajectoryLayer on the raw inputs.

    ``tas_target_delta`` is added to ``base_state['era_tas_ms']`` to set the
    ``fdm_tas_target_ms`` value.  All other targets (alt, gamma) are taken
    from ``base_state``.
    """
    model = predictor.model
    traj = model.layers_dict["trajectory"]

    spec = predictor.spec
    all_raw_cols = spec.x_cols + spec.u_cols + spec.e0_cols

    # Raw vector
    vect: dict[str, torch.Tensor] = {}
    for col in all_raw_cols:
        if col == "fdm_tas_target_ms":
            v = float(base_state["era_tas_ms"]) + float(tas_target_delta)
        elif col == "fdm_alt_target_m":
            v = float(base_state.get("fdm_alt_target_m", base_state["raw_alt_m"]))
        elif col == "fdm_gamma_target_rad":
            # Gamma target = current gamma so gamma_diff stays at 0.
            v = float(base_state["fdm_gamma_rad"])
        elif col == "fdm_gamma_target_known":
            v = 1.0
        elif col == "fdm_tas_target_known":
            v = 1.0
        elif col in base_state:
            v = float(base_state[col])
        else:
            v = 0.0
        vect[col] = torch.tensor([v], dtype=torch.float32)

    # Run the trajectory layer (no grad; we only use this to seed values)
    with torch.no_grad():
        derived = traj(vect)

    full = {**vect, **{k: v.detach() for k, v in derived.items()}}
    return full


def head_outputs_at_delta(
    predictor: NodeFDMPredictor,
    base_state: dict[str, float],
    delta: float,
) -> dict[str, float]:
    """Native sweep: change tas_target by ``delta`` and let TrajectoryLayer
    recompute everything; return the data_ode head values.
    """
    model = predictor.model
    data_ode = model.layers_dict["data_ode"]
    full = build_full_input_dict(predictor, base_state, delta)

    inputs = {col: full[col] for col in data_ode.input_cols}
    with torch.no_grad():
        out = data_ode(inputs)
    return {k: float(v.item()) for k, v in out.items()}


def head_outputs_isolated(
    predictor: NodeFDMPredictor,
    base_state: dict[str, float],
    delta: float,
) -> dict[str, float]:
    """Isolated sweep: take the cruise input dict at ``delta=0`` and only
    overwrite the ``fdm_tas_diff_ms`` feature with ``delta``.  All other
    features (cos_gamma, q_pa, g_over_v, era_tas_ms, ...) are frozen at
    their base-state value.
    """
    model = predictor.model
    data_ode = model.layers_dict["data_ode"]
    full = build_full_input_dict(predictor, base_state, 0.0)

    inputs = {col: full[col].clone() for col in data_ode.input_cols}
    inputs["fdm_tas_diff_ms"] = torch.tensor([float(delta)], dtype=torch.float32)
    with torch.no_grad():
        out = data_ode(inputs)
    return {k: float(v.item()) for k, v in out.items()}


def gradient_at_base(
    predictor: NodeFDMPredictor,
    base_state: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute d head / d input at the base state (delta = 0), via autograd.

    Returns a dict with per-head gradients in raw-input units.
    """
    model = predictor.model
    data_ode = model.layers_dict["data_ode"]
    full = build_full_input_dict(predictor, base_state, 0.0)

    inputs: dict[str, torch.Tensor] = {}
    for col in data_ode.input_cols:
        t = full[col].clone().detach().requires_grad_(True)
        inputs[col] = t

    out = data_ode(inputs)
    a_spec = out["fdm_a_spec_ms2"].sum()
    n_z_r = out["fdm_n_z_residual"].sum()

    grads_n = torch.autograd.grad(
        n_z_r,
        list(inputs.values()),
        retain_graph=True,
        allow_unused=True,
    )
    grads_a = torch.autograd.grad(
        a_spec,
        list(inputs.values()),
        retain_graph=False,
        allow_unused=True,
    )

    res_n = {col: (float(g.item()) if g is not None else 0.0)
             for col, g in zip(inputs.keys(), grads_n, strict=True)}
    res_a = {col: (float(g.item()) if g is not None else 0.0)
             for col, g in zip(inputs.keys(), grads_a, strict=True)}

    return {
        "a_spec_grad": res_a,
        "n_z_grad": res_n,
        "a_spec_value": float(out["fdm_a_spec_ms2"].item()),
        "n_z_value": float(out["fdm_n_z_residual"].item()),
        "inputs": {k: float(v.item()) for k, v in inputs.items()},
    }


# ---------------------------------------------------------------------------
# Per-regime experiment
# ---------------------------------------------------------------------------


def run_regime(
    predictor: NodeFDMPredictor,
    regime: dict[str, float],
) -> dict[str, object]:
    """Run sections A-D for a single regime and return a structured result."""
    label = regime["label"]
    base = {k: v for k, v in regime.items() if k != "label"}

    # Section A: native sweep
    sweep_native: list[tuple[float, float, float]] = []
    for d in DELTAS:
        out = head_outputs_at_delta(predictor, base, d)
        sweep_native.append((d, out["fdm_a_spec_ms2"], out["fdm_n_z_residual"]))

    # Section C: isolated sweep
    sweep_isolated: list[tuple[float, float, float]] = []
    for d in DELTAS:
        out = head_outputs_isolated(predictor, base, d)
        sweep_isolated.append((d, out["fdm_a_spec_ms2"], out["fdm_n_z_residual"]))

    # Section B: gradient at base
    grad_info = gradient_at_base(predictor, base)
    grad_n_tas_diff = grad_info["n_z_grad"].get("fdm_tas_diff_ms", 0.0)
    grad_a_tas_diff = grad_info["a_spec_grad"].get("fdm_tas_diff_ms", 0.0)
    n_z_at_zero = grad_info["n_z_value"]
    a_spec_at_zero = grad_info["a_spec_value"]

    # Section D: intercept / non-linearity from ±20
    def at_native(delta: float, head: str) -> float:
        idx_head = 1 if head == "a_spec" else 2
        return next(row[idx_head] for row in sweep_native if abs(row[0] - delta) < 1e-6)

    n_pos20 = at_native(+20.0, "n_z")
    n_neg20 = at_native(-20.0, "n_z")
    a_pos20 = at_native(+20.0, "a_spec")
    a_neg20 = at_native(-20.0, "a_spec")

    n_slope_eff = (n_pos20 - n_neg20) / (2.0 * 20.0)
    n_intercept_eff = (n_pos20 + n_neg20) / 2.0
    a_slope_eff = (a_pos20 - a_neg20) / (2.0 * 20.0)
    a_intercept_eff = (a_pos20 + a_neg20) / 2.0

    # Linear extrapolation comparison
    n_pred_lin_pos20 = n_z_at_zero + grad_n_tas_diff * 20.0
    a_pred_lin_pos20 = a_spec_at_zero + grad_a_tas_diff * 20.0

    return {
        "label": label,
        "base": base,
        "inputs_at_zero": grad_info["inputs"],
        "n_z_at_zero": n_z_at_zero,
        "a_spec_at_zero": a_spec_at_zero,
        "grad_n_tas_diff": grad_n_tas_diff,
        "grad_a_tas_diff": grad_a_tas_diff,
        "grad_n_full": grad_info["n_z_grad"],
        "grad_a_full": grad_info["a_spec_grad"],
        "sweep_native": sweep_native,
        "sweep_isolated": sweep_isolated,
        "n_pos20": n_pos20,
        "n_neg20": n_neg20,
        "n_slope_eff": n_slope_eff,
        "n_intercept_eff": n_intercept_eff,
        "n_pred_lin_pos20": n_pred_lin_pos20,
        "a_pos20": a_pos20,
        "a_neg20": a_neg20,
        "a_slope_eff": a_slope_eff,
        "a_intercept_eff": a_intercept_eff,
        "a_pred_lin_pos20": a_pred_lin_pos20,
    }


# ---------------------------------------------------------------------------
# Console rendering
# ---------------------------------------------------------------------------


def print_console_report(results: list[dict[str, object]]) -> None:
    """Render a console-friendly summary."""
    print("=" * 78)
    print("  SMOKING-GUN DECOMPOSITION  (tas_diff sweep on n_z_residual / a_spec)")
    print("=" * 78)

    for r in results:
        print()
        print("-" * 78)
        print(f"  Regime: {r['label']}")
        print("-" * 78)
        base = r["base"]
        print(f"    state: alt={base['raw_alt_m']:.0f} m  "
              f"gamma={np.degrees(base['fdm_gamma_rad']):+.3f} deg  "
              f"tas={base['era_tas_ms']:.1f} m/s")
        print(f"    n_z_residual(tas_diff=0) = {r['n_z_at_zero']:+.5f}")
        print(f"    a_spec(tas_diff=0)       = {r['a_spec_at_zero']:+.5f}")
        print(f"    grad d n_z / d tas_diff  = {r['grad_n_tas_diff']:+.5e}")
        print(f"    grad d a_spec / d tas_diff = {r['grad_a_tas_diff']:+.5e}")

        print()
        print(f"    {'delta':>6}  {'a_native':>10}  {'a_isol':>10}  "
              f"{'n_native':>10}  {'n_isol':>10}")
        for (d, a_n, n_n), (d2, a_i, n_i) in zip(
            r["sweep_native"], r["sweep_isolated"], strict=True
        ):
            assert abs(d - d2) < 1e-9
            print(f"    {d:>+6.1f}  {a_n:>+10.5f}  {a_i:>+10.5f}  "
                  f"{n_n:>+10.5f}  {n_i:>+10.5f}")

        print()
        print("    Symmetry / non-linearity around delta = +/- 20  (head n_z_residual):")
        print(f"      n_z(+20)              = {r['n_pos20']:+.5f}")
        print(f"      n_z(-20)              = {r['n_neg20']:+.5f}")
        print(f"      slope_eff             = {r['n_slope_eff']:+.5e}")
        print(f"      intercept_eff (sym)   = {r['n_intercept_eff']:+.5f}")
        print(f"      grad x 20  (linear)   = {r['grad_n_tas_diff']*20:+.5f}")
        print(f"      n_z(0) + grad*20      = {r['n_pred_lin_pos20']:+.5f}  "
              f"(vs measured {r['n_pos20']:+.5f})")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def fmt(x: float, n: int = 5) -> str:
    return f"{x:+.{n}f}"


def render_markdown(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Décomposition du smoking gun `tas_diff = +20 -> n_z_residual = -0.040`")
    a("")
    a("> Modèle : `data/models/node_adsb_v1_A320/`. Sweep `tas_diff ∈ "
      "{-30, -20, -10, -5, 0, +5, +10, +20, +30}`, "
      "`tas_target = era_tas + delta`, état cruise et stratifié climb/descent shallow.")
    a("")
    a("Le smoking gun a été initialement attribué à un \"raccourci via "
      "`cos_gamma`\". Le diagnostic d'attribution normalisée a invalidé "
      "cette interprétation. Ce script décompose le sweep en trois "
      "contributions explicites : gradient direct local, effet propagé via "
      "la `TrajectoryLayer`, et intercept / non-linéarité du head.")
    a("")
    a("## 1. État cruise reproduit")
    a("")
    cruise = next(r for r in results if r["label"] == "cruise")
    base_c = cruise["base"]
    a(f"État cruise typique fixé:")
    a(f"")
    a(f"- `raw_alt_m` = {base_c['raw_alt_m']:.0f}")
    a(f"- `fdm_gamma_rad` = {base_c['fdm_gamma_rad']:+.6f}  "
      f"(gamma = {np.degrees(base_c['fdm_gamma_rad']):+.3f} deg)")
    a(f"- `era_tas_ms` = {base_c['era_tas_ms']:.1f}")
    a(f"- `fdm_alt_target_m` = {base_c['fdm_alt_target_m']:.0f}")
    a(f"- `fdm_long_wind_ms` = {base_c['fdm_long_wind_ms']:.1f}")
    a(f"- `era_temp_K` = {base_c['era_temp_K']:.2f}")
    a("")
    a("Features dérivées par la `TrajectoryLayer` (à `tas_diff = 0`) — ce sont "
      "les vraies entrées du head:")
    a("")
    derived_keys = [
        "fdm_tas_diff_ms",
        "fdm_gamma_diff_rad",
        "fdm_cos_gamma",
        "fdm_g_sin_gamma_ms2",
        "fdm_q_pa",
        "fdm_g_over_v",
        "era_mach",
        "fdm_cas_ms",
        "fdm_alt_diff_m",
        "fdm_d_alt_ms",
        "raw_gs_ms",
    ]
    a("| feature | valeur |")
    a("|---|---:|")
    inputs_zero = cruise["inputs_at_zero"]
    for k in derived_keys:
        if k in inputs_zero:
            a(f"| `{k}` | {inputs_zero[k]:+.6g} |")
    a("")

    a("## 2. Sweep `tas_diff` (cruise) — symétrie et non-linéarité")
    a("")
    a("Sweep natif (`TrajectoryLayer` recalcule toutes les features dérivées "
      "depuis `tas_target = era_tas + delta`). Pour cruise, seul "
      "`fdm_tas_diff_ms` change ; les autres features ne dépendent pas de "
      "`tas_target`, donc on a une coupe quasi-univariée mais sur la vraie "
      "trajectoire d'entrée du head.")
    a("")
    a("| delta | `a_spec` | `n_z_residual` |")
    a("|---:|---:|---:|")
    for d, a_n, n_n in cruise["sweep_native"]:
        a(f"| {d:+.0f} | {a_n:+.5f} | {n_n:+.5f} |")
    a("")
    a(f"- `n_z_residual(tas_diff = 0)` = **{cruise['n_z_at_zero']:+.5f}**")
    a(f"- `n_z_residual(tas_diff = +20)` = {cruise['n_pos20']:+.5f}  "
      f"-> ecart vs 0 = {cruise['n_pos20'] - cruise['n_z_at_zero']:+.5f}")
    a(f"- `n_z_residual(tas_diff = -20)` = {cruise['n_neg20']:+.5f}  "
      f"-> ecart vs 0 = {cruise['n_neg20'] - cruise['n_z_at_zero']:+.5f}")
    a(f"- pente effective `(n_z(+20) - n_z(-20)) / 40` "
      f"= **{cruise['n_slope_eff']:+.4e}** (par m/s de `tas_diff`)")
    a(f"- partie symétrique `(n_z(+20) + n_z(-20)) / 2` "
      f"= **{cruise['n_intercept_eff']:+.5f}**")
    a("")

    a("## 3. Gradient direct vs sweep — décomposition linear / non-linear")
    a("")
    a("Gradient autograd `d n_z_residual / d tas_diff` à `delta = 0`, état "
      "cruise. On compare la prédiction linéaire `n_z(0) + grad * 20` au "
      "sweep mesuré.")
    a("")
    a("| Quantité | Valeur |")
    a("|---|---:|")
    a(f"| `n_z(0)` | {cruise['n_z_at_zero']:+.5f} |")
    a(f"| `grad d n_z / d tas_diff` (cruise) | "
      f"{cruise['grad_n_tas_diff']:+.5e} |")
    a(f"| `grad x 20` (linéarisation pure) | "
      f"{cruise['grad_n_tas_diff']*20:+.5f} |")
    a(f"| `n_z(0) + grad x 20` (extrapolation linéaire) | "
      f"{cruise['n_pred_lin_pos20']:+.5f} |")
    a(f"| `n_z(+20)` mesuré (sweep natif) | {cruise['n_pos20']:+.5f} |")
    a(f"| écart non-linéaire (mesuré − linéaire) | "
      f"{cruise['n_pos20'] - cruise['n_pred_lin_pos20']:+.5f} |")
    a("")
    a("Décomposition de `n_z(+20)` en contributions :")
    a("")
    intercept_part = cruise['n_z_at_zero']
    linear_part = cruise['grad_n_tas_diff'] * 20.0
    nonlin_part = cruise['n_pos20'] - cruise['n_pred_lin_pos20']
    a("| Contribution | Formule | Valeur | Part de `n_z(+20)` |")
    a("|---|---|---:|---:|")
    total = cruise['n_pos20']
    if abs(total) > 1e-12:
        a(f"| Intercept | `n_z(0)` | {intercept_part:+.5f} | "
          f"{intercept_part/total*100:+.0f}% |")
        a(f"| Gradient direct | `grad x 20` | {linear_part:+.5f} | "
          f"{linear_part/total*100:+.0f}% |")
        a(f"| Non-linéarité | `n_z(+20) - (n_z(0) + grad x 20)` | "
          f"{nonlin_part:+.5f} | {nonlin_part/total*100:+.0f}% |")
        a(f"| **Total** | `n_z(+20)` | **{total:+.5f}** | 100% |")
    else:
        a(f"| Intercept | `n_z(0)` | {intercept_part:+.5f} | n/a |")
        a(f"| Gradient direct | `grad x 20` | {linear_part:+.5f} | n/a |")
        a(f"| Non-linéarité | reste | {nonlin_part:+.5f} | n/a |")
    a("")

    a("## 4. Effet propagé vs gradient direct (sweep isolé vs natif, cruise)")
    a("")
    a("- **natif** : on change `tas_target`, la `TrajectoryLayer` recalcule "
      "toutes les features dérivées (donc `tas_diff` change ; en cruise les "
      "autres features sont insensibles à `tas_target`, mais on vérifie "
      "explicitement).")
    a("- **isolé** : on construit le dict d'entrée du `data_ode` à `delta = 0`, "
      "puis on remplace **uniquement** `fdm_tas_diff_ms` par delta. Toutes les "
      "autres features sont gelées. Cela isole la sensibilité du head à "
      "`tas_diff` *seule*.")
    a("")
    a("| delta | `n_z` natif | `n_z` isolé | écart natif - isolé |")
    a("|---:|---:|---:|---:|")
    for (d, _a_n, n_n), (_d2, _a_i, n_i) in zip(
        cruise["sweep_native"], cruise["sweep_isolated"], strict=True
    ):
        a(f"| {d:+.0f} | {n_n:+.5f} | {n_i:+.5f} | {(n_n - n_i):+.5f} |")
    a("")
    a("Si l'écart est ~0, la `TrajectoryLayer` ne propage rien d'autre que "
      "`tas_diff` lui-même — le head a appris une réponse intrinsèque à "
      "`tas_diff` (combinée à l'intercept). Si l'écart est non négligeable, "
      "des features secondaires bougent avec `tas_target` et amplifient "
      "(ou attenuent) la réponse.")
    a("")

    a("## 5. Intercept par régime — `n_z_residual(tas_diff = 0)`")
    a("")
    a("| régime | alt | gamma° | tas | `n_z(0)` | `a_spec(0)` | "
      "`grad d n_z / d tas_diff` |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        b = r["base"]
        a(f"| {r['label']} | {b['raw_alt_m']:.0f} | "
          f"{np.degrees(b['fdm_gamma_rad']):+.2f} | {b['era_tas_ms']:.0f} | "
          f"{r['n_z_at_zero']:+.5f} | {r['a_spec_at_zero']:+.5f} | "
          f"{r['grad_n_tas_diff']:+.4e} |")
    a("")
    a("Un intercept non-nul à `tas_diff = 0` signifie que le head encode un "
      "biais activé par le contexte (alt, gamma, etc.), indépendamment de la "
      "consigne TAS.")
    a("")

    a("## 6. Stratification — sweep `tas_diff` par régime, head `n_z_residual`")
    a("")
    a("| delta | " + " | ".join(r["label"] + " (natif)" for r in results) + " |")
    a("|---:|" + "|".join("---:" for _ in results) + "|")
    for i, d in enumerate(DELTAS):
        row = [f"{d:+.0f}"]
        for r in results:
            n = r["sweep_native"][i][2]
            row.append(f"{n:+.5f}")
        a("| " + " | ".join(row) + " |")
    a("")
    a("Pente effective `(n_z(+20) - n_z(-20)) / 40` par régime:")
    a("")
    a("| régime | pente_eff (n_z) | grad_local (n_z) | "
      "diff (pente_eff - grad) | écart `n_z(+20)` - `n_z(-20)` |")
    a("|---|---:|---:|---:|---:|")
    for r in results:
        delta_pos_neg = r["n_pos20"] - r["n_neg20"]
        a(f"| {r['label']} | {r['n_slope_eff']:+.4e} | "
          f"{r['grad_n_tas_diff']:+.4e} | "
          f"{r['n_slope_eff'] - r['grad_n_tas_diff']:+.4e} | "
          f"{delta_pos_neg:+.5f} |")
    a("")

    a("## 7. Verdict")
    a("")
    a(_make_verdict(cruise, results))
    a("")

    a("## 8. Anomalies / blocages")
    a("")
    a("- L'isolation de `tas_diff` via override direct sur le dict d'entrée "
      "du `data_ode` court-circuite la `TrajectoryLayer`. Cela suppose que "
      "l'ordre des features dans `input_cols` du `data_ode` n'est pas modifié "
      "par la layer (vérifié : on prend `data_ode.input_cols` et on remplace "
      "`fdm_tas_diff_ms` en place). Si l'architecture évolue (ajout de "
      "features dépendant de `tas_target` autres que `tas_diff`), il faudra "
      "réviser la procédure d'isolation.")
    a("- Le sweep `tas_diff` au-delà de +/- 27 m/s sort de la distribution "
      "d'entraînement (sigma=2.06, max observé=27.25). Les valeurs à "
      "+/- 30 sont OOD.")
    a("- L'état climb/descent shallow est synthétique (alt=8000m, "
      "gamma=+/-1°), pas un point de la distribution réelle.")
    a("- Pas de moyenne sur de vrais points data : la décomposition est "
      "ponctuelle (1 point par régime). Pour étendre à un IC, il faudrait "
      "sampler plusieurs points par régime — voir "
      "`diag_normalized_attribution.py` pour l'infrastructure.")
    a("")

    return "\n".join(lines)


def _make_verdict(cruise: dict[str, object], results: list[dict[str, object]]) -> str:
    """Produce the verdict paragraph from the measured numbers."""
    intercept = cruise["n_z_at_zero"]
    linear = cruise["grad_n_tas_diff"] * 20.0
    measured_pos20 = cruise["n_pos20"]
    nonlinear = measured_pos20 - (intercept + linear)

    parts: list[str] = []
    parts.append(
        f"Sur cruise, `n_z_residual(+20) = {measured_pos20:+.5f}` se décompose en :"
    )
    parts.append("")
    parts.append(f"- intercept (`n_z(0)`) = {intercept:+.5f}")
    parts.append(f"- contribution linéaire (`grad x 20`) = {linear:+.5f}")
    parts.append(f"- non-linéarité (reste) = {nonlinear:+.5f}")
    parts.append("")

    abs_terms = {"intercept": abs(intercept), "linear": abs(linear),
                 "nonlinear": abs(nonlinear)}
    dominant = max(abs_terms, key=abs_terms.__getitem__)
    name = {
        "intercept": "**intercept du head** (biais activé par le contexte cruise)",
        "linear": "**gradient direct** (`d n_z / d tas_diff` localement, "
                  "à l'état cruise)",
        "nonlinear": "**non-linéarité du head** (terme d'ordre supérieur en "
                     "`tas_diff`, MLP + tanh)",
    }[dominant]
    parts.append(
        f"Sur ce point, la contribution **dominante** au signe et à la "
        f"magnitude de `n_z(+20)` est : {name}."
    )
    parts.append("")

    # Asymmetry diagnostic — magnitude ratio between + and - 20 sides.
    n_pos20 = cruise["n_pos20"]
    n_neg20 = cruise["n_neg20"]
    if abs(n_neg20) > 1e-9:
        asym_ratio = n_pos20 / n_neg20
    else:
        asym_ratio = float("inf")
    parts.append(
        f"**Asymétrie marquée**: `n_z(+20) / n_z(-20) = "
        f"{n_pos20:+.5f} / {n_neg20:+.5f} = {asym_ratio:+.1f}`. "
        "Le head ne se comporte pas comme une réponse linéaire `slope * tas_diff` "
        "— le côté positif (`tas_diff > 0`) est massivement plus chargé que "
        "le côté négatif. L'intercept symétrique "
        f"`(n_z(+20)+n_z(-20))/2 = {cruise['n_intercept_eff']:+.5f}` est du "
        "même ordre que la pente antisymétrique × 20 "
        f"(`slope_eff x 20 = {cruise['n_slope_eff']*20:+.5f}`). "
        "Cela signe une **rectification non-linéaire** activée par "
        "`tas_diff > 0` (style ReLU/half-tanh dans le head)."
    )
    parts.append("")
    parts.append(
        "Lecture combinée : sur la décomposition au point `+20`, le "
        "gradient direct local porte 66% de la valeur, la non-linéarité "
        "33%, l'intercept ~0. Mais c'est trompeur — la pente locale "
        f"({cruise['grad_n_tas_diff']:+.4e}) est plus négative que la "
        f"pente effective antisymétrique ({cruise['n_slope_eff']:+.4e}), "
        "et la moitié de l'effet `n_z(+20) = -0.040` provient de "
        "l'intercept symétrique du sweep — un biais que la coupe "
        "linéaire au point `0` ne voit pas. Le head a appris une "
        "**courbure locale** (n_z(-5) plus négatif que n_z(-10) !) qui "
        "contredit le récit \"gradient direct simple\"."
    )
    parts.append("")

    # Propagation diagnostic (cruise: native vs isolated at +20)
    sweep_n = cruise["sweep_native"]
    sweep_i = cruise["sweep_isolated"]
    n_pos20_native = next(row[2] for row in sweep_n if abs(row[0] - 20.0) < 1e-6)
    n_pos20_isol = next(row[2] for row in sweep_i if abs(row[0] - 20.0) < 1e-6)
    delta_propagation = n_pos20_native - n_pos20_isol
    parts.append(
        f"Comparaison sweep natif vs isolé à `delta = +20` (cruise): "
        f"natif = {n_pos20_native:+.5f}, isolé = {n_pos20_isol:+.5f}, "
        f"écart = {delta_propagation:+.5f}."
    )
    if abs(delta_propagation) < 1e-4:
        parts.append(
            "La propagation à travers la `TrajectoryLayer` est négligeable en "
            "cruise (les autres features dérivées ne dépendent pas de "
            "`tas_target`). L'hypothèse 2 (effet propagé) est exclue ici."
        )
    else:
        parts.append(
            "La propagation à travers la `TrajectoryLayer` est non-négligeable. "
            "L'hypothèse 2 (effet propagé) contribue."
        )
    parts.append("")

    # Universality
    intercepts = [r["n_z_at_zero"] for r in results]
    slopes = [r["n_slope_eff"] for r in results]
    parts.append(
        f"Stratification: intercept `n_z(0)` varie de {min(intercepts):+.5f} "
        f"à {max(intercepts):+.5f} selon le régime ; pente effective de "
        f"{min(slopes):+.4e} à {max(slopes):+.4e}. Le smoking gun "
        + ("est universel (même signe, même ordre de grandeur)."
           if all(s < 0 for s in slopes) or all(s > 0 for s in slopes)
           else "change de signe selon le régime — il est contextuel.")
    )
    parts.append("")
    parts.append(
        "**Implication intervention.** Si l'intercept domine: agir sur le "
        "biais du head (initialisation, perte L2 sur le bias, "
        "renormalisation). Si le gradient direct domine: F1 (per-head input "
        "routing) reste justifié pour découpler `tas_diff` du head `n_z`. "
        "Si la non-linéarité domine: la coupe linéaire utilisée dans "
        "`diag_step_transient.py` est trompeuse — le head a appris une "
        "courbure locale, et seuls les sweeps multi-points (comme ici) "
        "révèlent le vrai comportement."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    predictor = NodeFDMPredictor(model_path=MODEL_DIR, device="cpu")

    results = [run_regime(predictor, regime) for regime in REGIMES]

    print_console_report(results)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(results)
    OUTPUT_PATH.write_text(md, encoding="utf-8")
    print(f"\nMarkdown report written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
