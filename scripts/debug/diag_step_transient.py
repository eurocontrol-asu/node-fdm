"""Substep-level diagnostic for the TAS-step transient bug.

Goal: explain why the model produces a *non-monotonic, wrong-direction* TAS
response immediately after a TAS-target step.

This script:

1. Loads the trained ``node_adsb_v1_A320`` model.
2. Runs a synthetic cruise scenario with a ``tas_target = 230 -> 250`` step.
3. Captures inputs and outputs of the StructuredLayer ("data_ode") at EVERY
   RK4 sub-step (4 per integration window) — not just averaged.
4. Calls the StructuredLayer manually around the discontinuity to compute
   ``d a_spec / d input`` for each input feature (gradient attribution).
5. Runs the same scenario with a *smoothed* target (no step) to test whether
   the BatchNeuralODE linear interpolation is the culprit.

Run from project root::

    uv run python scripts/debug/diag_step_transient.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "node-fdm" / "src"))

from node_fdm.architectures import adsb as _adsb_reg  # noqa: F401
from node_fdm.predictor import NodeFDMPredictor

MODEL_DIR = Path("data/models/node_adsb_v1_A320")
STEP_S = 4.0
N_STEPS = 16
CHANGE_STEP = 8  # t = 32 s

X_COLS = ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms"]
U_COLS = [
    "fdm_alt_target_m",
    "fdm_tas_target_ms",
    "fdm_gamma_target_rad",
    "fdm_gamma_target_known",
    "fdm_tas_target_known",
]
E0_COLS = ["fdm_long_wind_ms", "era_temp_K"]


# ---------------------------------------------------------------------------
# Substep capture: hook the StructuredLayer and PhysicsLayer
# ---------------------------------------------------------------------------


def install_substep_hooks(predictor: NodeFDMPredictor) -> dict[str, list]:
    store: dict[str, list] = {"data_ode_in": [], "data_ode_out": [], "phys_out": []}
    model = predictor.model
    data_ode = model.layers_dict["data_ode"]
    phys = model.layers_dict["physics"]

    def _pre(_m, args, kwargs):
        x_dict = args[0] if args else kwargs.get("x")
        if isinstance(x_dict, dict):
            row = {}
            for k in data_ode.input_cols:
                v = x_dict.get(k)
                if torch.is_tensor(v):
                    row[k] = float(v.detach().mean().item())
            # Also capture target/known for diagnostic
            for k in (
                "fdm_tas_target_ms",
                "fdm_tas_target_known",
                "fdm_gamma_target_known",
            ):
                v = x_dict.get(k)
                if torch.is_tensor(v):
                    row[k] = float(v.detach().mean().item())
            store["data_ode_in"].append(row)

    def _post_data(_m, _i, out):
        if isinstance(out, dict):
            store["data_ode_out"].append(
                {k: float(v.detach().mean().item()) for k, v in out.items() if torch.is_tensor(v)}
            )

    def _post_phys(_m, _i, out):
        if isinstance(out, dict):
            store["phys_out"].append(
                {k: float(v.detach().mean().item()) for k, v in out.items() if torch.is_tensor(v)}
            )

    data_ode.register_forward_pre_hook(_pre, with_kwargs=True)
    data_ode.register_forward_hook(_post_data)
    phys.register_forward_hook(_post_phys)
    return store


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def make_scenario(target_after: float, ramp_steps: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cruise scenario; tas target steps from 230 to ``target_after`` at idx CHANGE_STEP.

    If ``ramp_steps > 0``, ramp the target linearly over that many steps.
    """
    x0 = np.array([10000.0, 0.0, 230.0], dtype=np.float32)
    u = np.zeros((N_STEPS, len(U_COLS)), dtype=np.float32)
    e = np.zeros((N_STEPS, len(E0_COLS)), dtype=np.float32)
    for i in range(N_STEPS):
        u[i, 0] = 10000.0  # alt_target
        if i < CHANGE_STEP:
            u[i, 1] = 230.0
        elif ramp_steps > 0 and i < CHANGE_STEP + ramp_steps:
            frac = (i - CHANGE_STEP + 1) / ramp_steps
            u[i, 1] = 230.0 + frac * (target_after - 230.0)
        else:
            u[i, 1] = target_after
        u[i, 2] = 0.0  # gamma_target
        u[i, 3] = 1.0  # gamma_known
        u[i, 4] = 1.0  # tas_known
        e[i, 0] = 0.0  # wind
        e[i, 1] = 223.15  # temp
    return x0, u, e


# ---------------------------------------------------------------------------
# Gradient attribution at a given physical state
# ---------------------------------------------------------------------------


def gradient_attribution(
    predictor: NodeFDMPredictor,
    state: dict[str, float],
    targets: dict[str, float],
    env: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute d a_spec / d input and d n_z_residual / d input at one point."""
    model = predictor.model
    traj = model.layers_dict["trajectory"]
    data_ode = model.layers_dict["data_ode"]

    # Build a single-row vect_dict
    vect: dict[str, torch.Tensor] = {}
    all_cols = predictor.spec.x_cols + predictor.spec.u_cols + predictor.spec.e0_cols
    for col in all_cols:
        v = state.get(col, targets.get(col, env.get(col, 0.0)))
        vect[col] = torch.tensor([v], dtype=torch.float32, requires_grad=False)

    # Build derived features via TrajectoryLayer (no grad needed for upstream)
    with torch.no_grad():
        derived = traj(vect)
    vect = {**vect, **derived}

    # Now build the inputs to data_ode with grad enabled
    inputs = {}
    for col in data_ode.input_cols:
        t = vect[col].clone().detach().requires_grad_(True)
        inputs[col] = t

    # Forward
    out = data_ode(inputs)
    a_spec = out["fdm_a_spec_ms2"].sum()
    n_z_r = out["fdm_n_z_residual"].sum()

    # Compute gradients
    grads_a = torch.autograd.grad(a_spec, list(inputs.values()), retain_graph=True, allow_unused=True)
    grads_n = torch.autograd.grad(n_z_r, list(inputs.values()), retain_graph=False, allow_unused=True)

    result_a = {}
    result_n = {}
    for col, ga, gn in zip(inputs.keys(), grads_a, grads_n, strict=True):
        result_a[col] = float(ga.item()) if ga is not None else 0.0
        result_n[col] = float(gn.item()) if gn is not None else 0.0

    return {
        "a_spec_grad": result_a,
        "n_z_grad": result_n,
        "a_spec_value": float(out["fdm_a_spec_ms2"].item()),
        "n_z_value": float(out["fdm_n_z_residual"].item()),
        "inputs": {k: float(v.item()) for k, v in inputs.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    predictor = NodeFDMPredictor(model_path=MODEL_DIR, device="cpu")

    print("=" * 78)
    print("  SUBSTEP-LEVEL DIAGNOSTIC OF TAS-STEP TRANSIENT")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Pass 1: hard step 230 -> 250
    # ------------------------------------------------------------------
    store = install_substep_hooks(predictor)
    x0, u, e = make_scenario(target_after=250.0, ramp_steps=0)
    state_traj = predictor.predict_flight(x0, u, e)

    n_calls = len(store["data_ode_in"])
    calls_per_step = n_calls // N_STEPS if N_STEPS else 1
    print(f"\nIntegrator calls: {n_calls} ({calls_per_step} per step)")

    # Print substep-level around the step
    print("\n--- INPUTS to data_ode at every RK4 substep around CHANGE_STEP=8 ---")
    print(
        f"{'win':>4}{'sub':>4} | {'t≈':>6} | {'tas':>7} {'tas_tgt':>8} {'tas_diff':>9} "
        f"{'gamma°':>7} {'g·sin γ':>8} {'q':>8}"
    )
    print("-" * 78)
    for window in range(CHANGE_STEP - 2, min(CHANGE_STEP + 4, N_STEPS)):
        for sub in range(calls_per_step):
            idx = window * calls_per_step + sub
            if idx >= n_calls:
                continue
            r_in = store["data_ode_in"][idx]
            t_approx = window * STEP_S + (sub / calls_per_step) * STEP_S
            print(
                f"{window:>4}{sub:>4} | {t_approx:>6.1f} | "
                f"{r_in.get('era_tas_ms', 0):>7.2f} "
                f"{r_in.get('fdm_tas_target_ms', 0):>8.2f} "
                f"{r_in.get('fdm_tas_diff_ms', 0):>9.3f} "
                f"{np.degrees(r_in.get('fdm_gamma_rad', 0)):>7.3f} "
                f"{r_in.get('fdm_g_sin_gamma_ms2', 0):>8.4f} "
                f"{r_in.get('fdm_q_pa', 0):>8.0f}"
            )

    print("\n--- OUTPUTS of data_ode at same substeps ---")
    print(f"{'win':>4}{'sub':>4} | {'a_spec':>9} {'n_z_resid':>10}")
    print("-" * 40)
    for window in range(CHANGE_STEP - 2, min(CHANGE_STEP + 4, N_STEPS)):
        for sub in range(calls_per_step):
            idx = window * calls_per_step + sub
            if idx >= n_calls:
                continue
            r_out = store["data_ode_out"][idx]
            print(
                f"{window:>4}{sub:>4} | "
                f"{r_out.get('fdm_a_spec_ms2', 0):>9.4f} "
                f"{r_out.get('fdm_n_z_residual', 0):>10.5f}"
            )

    # ------------------------------------------------------------------
    # Pass 2: gradient attribution at three points
    # ------------------------------------------------------------------
    # Use the first-substep input snapshots as "physical state" probes.
    def get_substep_state(window: int, sub: int = 0) -> dict[str, float]:
        idx = window * calls_per_step + sub
        return dict(store["data_ode_in"][idx]) if idx < n_calls else {}

    print("\n" + "=" * 78)
    print("  GRADIENT ATTRIBUTION  d{a_spec, n_z_resid} / d{input}")
    print("=" * 78)

    probes = [
        ("CRUISE pre-step (window 6)", get_substep_state(6)),
        ("AT-step (window 8 sub 0)", get_substep_state(CHANGE_STEP, 0)),
        ("AT-step (window 8 sub 2)", get_substep_state(CHANGE_STEP, 2)),
        ("POST-step (window 10)", get_substep_state(CHANGE_STEP + 2)),
    ]

    # Build a fake state/target/env mapping for each probe.
    for label, snap in probes:
        if not snap:
            print(f"\n{label}: no data")
            continue
        # Inputs to data_ode are X+U_ODE+E0+E1+known flags
        # For gradient_attribution we need the raw (state, targets, env) so traj
        # layer can reproduce E1. Reconstruct from snapshot.
        state = {
            "raw_alt_m": snap.get("raw_alt_m", 10000.0),
            "fdm_gamma_rad": snap.get("fdm_gamma_rad", 0.0),
            "era_tas_ms": snap.get("era_tas_ms", 230.0),
        }
        targets = {
            "fdm_alt_target_m": 10000.0,
            "fdm_tas_target_ms": snap.get("fdm_tas_target_ms", 230.0),
            "fdm_gamma_target_rad": 0.0,
            "fdm_gamma_target_known": 1.0,
            "fdm_tas_target_known": 1.0,
        }
        env = {"fdm_long_wind_ms": 0.0, "era_temp_K": 223.15}

        ga = gradient_attribution(predictor, state, targets, env)
        print(f"\n--- {label} ---")
        print(f"  state: tas={state['era_tas_ms']:.2f} gamma={np.degrees(state['fdm_gamma_rad']):.3f}° tas_tgt={targets['fdm_tas_target_ms']:.2f}")
        print(f"  a_spec value = {ga['a_spec_value']:+.4f}    n_z_residual value = {ga['n_z_value']:+.5f}")
        # Sort by absolute gradient magnitude for a_spec
        items = sorted(ga["a_spec_grad"].items(), key=lambda kv: -abs(kv[1]))
        print(f"  Top sensitivities ∂a_spec/∂input (in raw-input units):")
        for col, g in items[:8]:
            in_val = ga["inputs"].get(col, 0.0)
            print(f"     {col:<28s}  d/d input = {g:+.4e}   (input value = {in_val:+.4e})")

    # ------------------------------------------------------------------
    # Pass 3: smoothed-target scenario (ramp over 4 steps = 16 s)
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  COMPARISON: hard step vs 4-step ramped target")
    print("=" * 78)

    store2 = install_substep_hooks(predictor)
    x0r, ur, er = make_scenario(target_after=250.0, ramp_steps=4)
    traj_ramp = predictor.predict_flight(x0r, ur, er)

    print("\n  Scenario           |  step   tas(t=32)  tas(t=36)  tas(t=44)  tas(t=60) min(tas in [32,44])")
    print("-" * 100)
    for label, traj_, u_ in [("hard step (1×Δ=20)", state_traj, u), ("ramped 4×Δ=5", traj_ramp, ur)]:
        tas_arr = traj_["era_tas_ms"]

        def at(t: float) -> float:
            i = int(t / STEP_S)
            return float(tas_arr[i]) if 0 <= i < len(tas_arr) else float("nan")

        i_lo, i_hi = int(32 / STEP_S), int(44 / STEP_S)
        min_tas = float(np.min(tas_arr[i_lo : i_hi + 1])) if i_hi < len(tas_arr) else float("nan")
        print(
            f"  {label:<18s} |  step    {at(32):>7.2f}    {at(36):>7.2f}    {at(44):>7.2f}    {at(60):>7.2f}    {min_tas:>7.2f}"
        )

    # If hard-step has min_tas < 230 (initial) or shows non-monotonic dip
    # right after the step, the ramp version should be cleaner.

    # ------------------------------------------------------------------
    # Pass 4: synthetic OOD probe — sweep tas_diff through ±30 at fixed cruise
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  TAS_DIFF SWEEP at cruise state (alt=10000, gamma=0, tas=230)")
    print("=" * 78)
    print(f"  Training stats: tas_diff std=2.06, max=27.25 → step of +20 is ~10σ")
    print()
    print(f"  {'tas_diff':>9} | {'a_spec':>9} | {'n_z_resid':>10}")
    print("-" * 40)

    state = {"raw_alt_m": 10000.0, "fdm_gamma_rad": 0.0, "era_tas_ms": 230.0}
    env = {"fdm_long_wind_ms": 0.0, "era_temp_K": 223.15}
    for delta in [-30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30]:
        targets = {
            "fdm_alt_target_m": 10000.0,
            "fdm_tas_target_ms": 230.0 + delta,
            "fdm_gamma_target_rad": 0.0,
            "fdm_gamma_target_known": 1.0,
            "fdm_tas_target_known": 1.0,
        }
        ga = gradient_attribution(predictor, state, targets, env)
        print(f"  {delta:>+9} | {ga['a_spec_value']:>+9.4f} | {ga['n_z_value']:>+10.5f}")

    # ------------------------------------------------------------------
    # Pass 5: TAS-DOWN step rollout
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  TAS-DOWN step (230 -> 210)")
    print("=" * 78)
    store_d = install_substep_hooks(predictor)
    x0d, ud, ed = make_scenario(target_after=210.0, ramp_steps=0)
    traj_d = predictor.predict_flight(x0d, ud, ed)
    tas_d = traj_d["era_tas_ms"]
    gam_d = traj_d["fdm_gamma_rad"]
    print(f"  {'t':>4} | {'tas':>7} | {'gamma°':>8} | {'a_spec(avg)':>12} | {'n_z_res(avg)':>13}")
    print("-" * 70)
    n_calls_d = len(store_d["data_ode_in"])
    cps = max(1, n_calls_d // N_STEPS)
    for i in range(N_STEPS):
        a_avg = np.mean([store_d["data_ode_out"][i*cps + s].get("fdm_a_spec_ms2", 0) for s in range(cps) if i*cps+s < n_calls_d])
        n_avg = np.mean([store_d["data_ode_out"][i*cps + s].get("fdm_n_z_residual", 0) for s in range(cps) if i*cps+s < n_calls_d])
        t = (i+1) * STEP_S
        print(f"  {t:>4.0f} | {tas_d[i]:>7.2f} | {np.degrees(gam_d[i]):>8.4f} | {a_avg:>12.4f} | {n_avg:>13.5f}")
    tas_min_after_step = float(np.min(tas_d[CHANGE_STEP:]))
    print(f"  min TAS after step at t=32: {tas_min_after_step:.2f}  (target=210, initial=230)")
    if tas_min_after_step < 210.0 - 1.0:
        print(f"  *** UNDERSHOOT: TAS dipped to {tas_min_after_step:.2f}, below target {210}")
    # Check non-monotonicity in early transient (first 5 steps after step)
    early = tas_d[CHANGE_STEP:CHANGE_STEP+5]
    diffs = np.diff(early)
    if np.any(diffs > 0) and np.any(diffs < 0):
        print(f"  *** NON-MONOTONIC: early TAS diffs after step = {diffs}")


if __name__ == "__main__":
    main()
