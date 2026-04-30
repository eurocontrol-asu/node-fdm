"""Experiment: diagnose erratic model behavior at target speed changes.

Tests three hypotheses:
  H1: Mixed signal in a_spec head (must track both gravity compensation
      and tas_diff simultaneously).
  H2: Tanh saturation of a_spec at ±2.5 m/s² during large tas_diff steps.
  H3: Cross-coupling — n_z_residual moves when only tas_target changes.

Usage:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/debug/exp_speed_target.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Make sure packages are on the path when run directly
sys.path.insert(0, str(Path(__file__).parents[2] / "packages" / "node-fdm" / "src"))

from node_fdm.architectures import adsb as _adsb_reg  # noqa: F401 — triggers register()
from node_fdm.predictor import NodeFDMPredictor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path("data/models/node_adsb_v1_A320")
STEP_S = 4.0
G = 9.80665

# State / control / env column order (from adsb schema)
X_COLS = ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms"]
U_COLS = [
    "fdm_alt_target_m",
    "fdm_tas_target_ms",
    "fdm_gamma_target_rad",
    "fdm_gamma_target_known",
    "fdm_tas_target_known",
]
E0_COLS = ["fdm_long_wind_ms", "era_temp_K"]

# Key timesteps to report (in seconds)
REPORT_TIMES_S = [0, 28, 32, 36, 44, 60]

# ---------------------------------------------------------------------------
# Intermediate capture via forward hooks
# ---------------------------------------------------------------------------

_INTERMEDIATES: dict[str, list[dict[str, float]]] = {}
_HOOKS: list[torch.utils.hooks.RemovableHook] = []


def _install_hooks(predictor: NodeFDMPredictor) -> None:
    """Install forward hooks on data_ode and physics layers."""
    global _INTERMEDIATES, _HOOKS

    # Remove previous hooks
    for h in _HOOKS:
        h.remove()
    _HOOKS.clear()
    _INTERMEDIATES.clear()

    model = predictor.model

    def _hook_data_ode(_module, _input, output: dict[str, torch.Tensor]) -> None:
        """Capture a_spec and n_z_residual after data_ode layer."""
        row = {
            "fdm_a_spec_ms2": output["fdm_a_spec_ms2"].item(),
            "fdm_n_z_residual": output["fdm_n_z_residual"].item(),
        }
        _INTERMEDIATES.setdefault("data_ode", []).append(row)

    def _hook_physics(_module, _input, output: dict[str, torch.Tensor]) -> None:
        """Capture d_tas, d_gamma, n_z after physics layer."""
        row = {
            "fdm_d_tas_ms2": output["fdm_d_tas_ms2"].item(),
            "fdm_d_gamma_rads": output["fdm_d_gamma_rads"].item(),
            "fdm_n_z": output["fdm_n_z"].item(),
        }
        _INTERMEDIATES.setdefault("physics", []).append(row)

    ode_layer = model.layers_dict["data_ode"]
    phys_layer = model.layers_dict["physics"]
    _HOOKS.append(ode_layer.register_forward_hook(_hook_data_ode))
    _HOOKS.append(phys_layer.register_forward_hook(_hook_physics))


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_scenario(
    n_steps: int,
    alt: float = 10_000.0,
    gamma: float = 0.0,
    tas: float = 230.0,
    alt_target: float = 10_000.0,
    gamma_target: float = 0.0,
    gamma_known: float = 1.0,
    tas_target_before: float = 230.0,
    tas_target_after: float = 230.0,
    gamma_target_after: float | None = None,
    change_step: int | None = None,
    wind: float = 0.0,
    temp_K: float = 223.15,  # ~ISA at FL330, close to 10 000 m
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic (x0, u_arr, e_arr) arrays.

    Args:
        n_steps: Total number of timesteps.
        alt: Initial altitude [m].
        gamma: Initial flight-path angle [rad].
        tas: Initial true airspeed [m/s].
        alt_target: Target altitude [m].
        gamma_target: Target flight-path angle [rad] (before change).
        gamma_known: 1.0 if gamma target is active.
        tas_target_before: TAS target before change [m/s].
        tas_target_after: TAS target after change [m/s].
        gamma_target_after: Gamma target after change (None = no change).
        change_step: Timestep index where targets change.
        wind: Longitudinal wind [m/s].
        temp_K: Air temperature [K].

    Returns:
        Tuple of (x0, u_arr, e_arr).
    """
    x0 = np.array([alt, gamma, tas], dtype=np.float32)

    u_arr = np.zeros((n_steps, len(U_COLS)), dtype=np.float32)
    e_arr = np.zeros((n_steps, len(E0_COLS)), dtype=np.float32)

    for i in range(n_steps):
        after = change_step is not None and i >= change_step
        u_arr[i, 0] = alt_target                                          # fdm_alt_target_m
        u_arr[i, 1] = tas_target_after if after else tas_target_before    # fdm_tas_target_ms
        u_arr[i, 2] = (gamma_target_after if (after and gamma_target_after is not None)
                       else gamma_target)                                 # fdm_gamma_target_rad
        u_arr[i, 3] = gamma_known                                         # fdm_gamma_target_known
        u_arr[i, 4] = 1.0                                                 # fdm_tas_target_known

        e_arr[i, 0] = wind   # fdm_long_wind_ms
        e_arr[i, 1] = temp_K  # era_temp_K

    return x0, u_arr, e_arr


# ---------------------------------------------------------------------------
# Run a rollout and collect all intermediates
# ---------------------------------------------------------------------------

def run_rollout(
    predictor: NodeFDMPredictor,
    x0: np.ndarray,
    u_arr: np.ndarray,
    e_arr: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run a full rollout and return state + intermediates.

    Returns:
        Dict with keys: raw_alt_m, fdm_gamma_rad, era_tas_ms,
        fdm_a_spec_ms2, fdm_n_z_residual, fdm_d_tas_ms2,
        fdm_d_gamma_rads, fdm_n_z.
    """
    global _INTERMEDIATES

    _install_hooks(predictor)
    _INTERMEDIATES.clear()

    state = predictor.predict_flight(x0, u_arr, e_arr)

    n_steps = u_arr.shape[0]

    # data_ode and physics are called once per model.forward() call.
    # The RK4 integrator calls model.forward() 4× per step (sub-steps),
    # so we average over the 4 sub-calls to get one representative value per step.
    ode_rows = _INTERMEDIATES.get("data_ode", [])
    phys_rows = _INTERMEDIATES.get("physics", [])

    calls_per_step = len(ode_rows) // n_steps if n_steps > 0 else 1
    if calls_per_step == 0:
        calls_per_step = 1

    def _avg_rows(rows: list[dict[str, float]], key: str) -> np.ndarray:
        vals = np.array([r[key] for r in rows])
        # Reshape to (n_steps, calls_per_step) and average
        n = min(len(vals), n_steps * calls_per_step)
        vals = vals[:n]
        pad = n_steps * calls_per_step - n
        if pad > 0:
            vals = np.concatenate([vals, np.full(pad, np.nan)])
        return vals.reshape(n_steps, calls_per_step).mean(axis=1)

    result = dict(state)
    for key in ("fdm_a_spec_ms2", "fdm_n_z_residual"):
        result[key] = _avg_rows(ode_rows, key)
    for key in ("fdm_d_tas_ms2", "fdm_d_gamma_rads", "fdm_n_z"):
        result[key] = _avg_rows(phys_rows, key)

    return result


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

HEADER = (
    f"{'t(s)':>5} | {'tas':>7} | {'tas_tgt':>7} | {'gamma(°)':>8} | "
    f"{'a_spec':>7} | {'n_z_res':>8} | {'d_tas':>7} | {'d_gamma':>9}"
)
SEP = "-" * len(HEADER)


def _fmt_row(
    t_s: float,
    tas: float,
    tas_tgt: float,
    gamma: float,
    a_spec: float,
    n_z_res: float,
    d_tas: float,
    d_gamma: float,
) -> str:
    return (
        f"{t_s:>5.0f} | {tas:>7.2f} | {tas_tgt:>7.2f} | "
        f"{np.degrees(gamma):>8.4f} | "
        f"{a_spec:>7.4f} | {n_z_res:>8.5f} | "
        f"{d_tas:>7.4f} | {d_gamma:>9.6f}"
    )


def print_scenario(
    label: str,
    result: dict[str, np.ndarray],
    u_arr: np.ndarray,
    report_times_s: list[int],
) -> None:
    """Print a table of key timesteps for one scenario."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(HEADER)
    print(SEP)

    n = len(result["era_tas_ms"])
    for t_s in report_times_s:
        idx = int(t_s / STEP_S)
        if idx >= n:
            continue
        tas = result["era_tas_ms"][idx]
        tas_tgt = u_arr[idx, 1]
        gamma = result["fdm_gamma_rad"][idx]
        a_spec = result["fdm_a_spec_ms2"][idx]
        n_z_res = result["fdm_n_z_residual"][idx]
        d_tas = result["fdm_d_tas_ms2"][idx]
        d_gamma = result["fdm_d_gamma_rads"][idx]
        print(_fmt_row(t_s, tas, tas_tgt, gamma, a_spec, n_z_res, d_tas, d_gamma))
    print(SEP)

    # Overshoot detection
    tas_arr = result["era_tas_ms"]
    tgt_arr = u_arr[:, 1]
    if np.any(tgt_arr > tgt_arr[0]):
        final_tgt = tgt_arr[-1]
        max_tas = np.max(tas_arr)
        overshoot = max_tas - final_tgt
        if overshoot > 0.5:
            print(f"  *** TAS OVERSHOOT detected: max={max_tas:.2f} vs target={final_tgt:.2f}"
                  f" (+{overshoot:.2f} m/s)")
        else:
            print(f"  No significant TAS overshoot (max={max_tas:.2f}, target={final_tgt:.2f})")


def diagnose_hypotheses(
    result_step: dict[str, np.ndarray],
    result_baseline: dict[str, np.ndarray],
    u_step: np.ndarray,
    change_step: int,
    cap_a_spec: float = 2.5,
    cap_n_z_res: float = 0.13,
) -> None:
    """Print H1/H2/H3 triage."""
    print(f"\n{'='*70}")
    print("  HYPOTHESIS TRIAGE")
    print(f"{'='*70}")

    # Find the index right at the change
    idx_change = change_step

    # Steady-state a_spec before change (use step before)
    idx_pre = max(0, idx_change - 1)
    a_pre = result_step["fdm_a_spec_ms2"][idx_pre]
    a_at_change = result_step["fdm_a_spec_ms2"][idx_change]

    # Max a_spec in first 5 steps after change
    post_slice = slice(idx_change, min(idx_change + 5, len(result_step["fdm_a_spec_ms2"])))
    a_post_max = np.max(np.abs(result_step["fdm_a_spec_ms2"][post_slice]))
    a_jump = abs(a_at_change - a_pre)

    # n_z_residual at change vs baseline
    nz_at_change = result_step["fdm_n_z_residual"][idx_change]
    nz_baseline = result_baseline["fdm_n_z_residual"][idx_change]
    nz_deviation = abs(nz_at_change - nz_baseline)

    # H1: a_spec jumps abruptly at change
    h1 = a_jump > 0.1
    # H2: a_spec saturates near cap
    h2 = a_post_max > (cap_a_spec * 0.9)
    # H3: n_z_residual moves significantly when only tas_target changes
    h3 = nz_deviation > 0.01

    print(f"\n  H1 (mixed signal in a_spec head):")
    print(f"    a_spec before change:  {a_pre:.4f} m/s²")
    print(f"    a_spec at change:      {a_at_change:.4f} m/s²")
    print(f"    jump magnitude:        {a_jump:.4f} m/s²")
    print(f"    => {'CONFIRMED' if h1 else 'NOT CONFIRMED'} (jump > 0.1 m/s²)")

    print(f"\n  H2 (tanh saturation at cap={cap_a_spec} m/s²):")
    print(f"    max |a_spec| in 5 steps post-change: {a_post_max:.4f} m/s²")
    print(f"    saturation threshold (90% of cap):   {cap_a_spec * 0.9:.4f} m/s²")
    print(f"    => {'CONFIRMED' if h2 else 'NOT CONFIRMED'}")

    print(f"\n  H3 (n_z cross-coupling with tas_target):")
    print(f"    n_z_residual at change (speed-step scenario): {nz_at_change:.6f}")
    print(f"    n_z_residual at same step (baseline):         {nz_baseline:.6f}")
    print(f"    deviation:                                    {nz_deviation:.6f}")
    print(f"    => {'CONFIRMED' if h3 else 'NOT CONFIRMED'} (deviation > 0.01)")

    print(f"\n  VERDICT:")
    causes = []
    if h1:
        causes.append("H1")
    if h2:
        causes.append("H2")
    if h3:
        causes.append("H3")
    if causes:
        print(f"    Most likely cause(s): {' + '.join(causes)}")
    else:
        print("    No hypothesis strongly confirmed — check for other factors.")

    # Recommended fix
    print(f"\n  RECOMMENDED FIX:")
    if h2 and not h1:
        print("    Increase cap_a_spec (currently 2.5) to reduce saturation,")
        print("    or use a softer activation (e.g. SELU/ELU) instead of tanh*cap.")
    elif h1 and not h2:
        print("    Decouple the a_spec head: split into a gravity-comp sub-head")
        print("    (predicts g*sin(gamma) residual) and a thrust-excess sub-head")
        print("    (responds to tas_diff). Or feed tas_diff only to a residual term.")
    elif h1 and h2:
        print("    Both mixed signal AND saturation are active.")
        print("    Primary fix: increase cap_a_spec to reduce saturation.")
        print("    Secondary fix: add a direct gravity-compensation path (g*sin(gamma))")
        print("    so the NN only needs to predict the residual.")
    elif h3:
        print("    Add .detach() on the TAS input inside PhysicsLayer so gradient")
        print("    from the 1/V term does not flow into the n_z head during training.")
    else:
        print("    Further investigation needed. Check training data distribution")
        print("    and whether the model was trained with target-step transitions.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load model ---
    if not MODEL_DIR.exists():
        print(f"ERROR: Model not found at {MODEL_DIR}", file=sys.stderr)
        raise SystemExit(1)

    predictor = NodeFDMPredictor(model_path=MODEL_DIR, device="cpu")
    arch = predictor.meta.architecture_name
    params = predictor.meta.model_params
    step = predictor.meta.step

    print(f"\n{'='*70}")
    print("  EXPERIMENT: Speed target step — hypothesis triage")
    print(f"{'='*70}")
    print(f"  Model:        {MODEL_DIR}")
    print(f"  Architecture: {arch}  params={params}")
    print(f"  Step:         {step} s  method={predictor.meta.method}")

    # Confirm 3 layers (trajectory, data_ode, physics) and n_z_residual output
    layer_names = [ls.name for ls in predictor.spec.layers]
    ode_out = [ls.output_cols for ls in predictor.spec.layers if ls.name == "data_ode"]
    print(f"  Layers:       {layer_names}")
    print(f"  ODE outputs:  {ode_out[0] if ode_out else 'N/A'}")

    # Timing
    total_time_s = 64  # 16 steps × 4 s
    n_steps = int(total_time_s / STEP_S)
    change_step = int(32 / STEP_S)  # step index 8 = t=32s

    # -------------------------------------------------------------------
    # Scenario 1: Cruise with speed step (230 → 250 at t=32s)
    # -------------------------------------------------------------------
    x0_s1, u_s1, e_s1 = build_scenario(
        n_steps=n_steps,
        alt=10_000.0,
        gamma=0.0,
        tas=230.0,
        alt_target=10_000.0,
        gamma_target=0.0,
        gamma_known=1.0,
        tas_target_before=230.0,
        tas_target_after=250.0,
        change_step=change_step,
    )
    result_s1 = run_rollout(predictor, x0_s1, u_s1, e_s1)
    print_scenario(
        "Scenario 1 — Cruise with speed step (tas: 230→250 at t=32s)",
        result_s1,
        u_s1,
        REPORT_TIMES_S,
    )

    # -------------------------------------------------------------------
    # Scenario 2: Baseline — no target change
    # -------------------------------------------------------------------
    x0_s2, u_s2, e_s2 = build_scenario(
        n_steps=n_steps,
        alt=10_000.0,
        gamma=0.0,
        tas=230.0,
        alt_target=10_000.0,
        gamma_target=0.0,
        gamma_known=1.0,
        tas_target_before=230.0,
        tas_target_after=230.0,
        change_step=None,
    )
    result_s2 = run_rollout(predictor, x0_s2, u_s2, e_s2)
    print_scenario(
        "Scenario 2 — Baseline cruise (no target change)",
        result_s2,
        u_s2,
        REPORT_TIMES_S,
    )

    # -------------------------------------------------------------------
    # Scenario 3: Speed AND gamma step at t=32s
    # -------------------------------------------------------------------
    x0_s3, u_s3, e_s3 = build_scenario(
        n_steps=n_steps,
        alt=10_000.0,
        gamma=0.0,
        tas=230.0,
        alt_target=10_000.0,
        gamma_target=0.0,
        gamma_known=1.0,
        tas_target_before=230.0,
        tas_target_after=250.0,
        gamma_target_after=0.05,
        change_step=change_step,
    )
    result_s3 = run_rollout(predictor, x0_s3, u_s3, e_s3)
    print_scenario(
        "Scenario 3 — Combined speed + gamma step (tas: 230→250, gamma: 0→0.05 at t=32s)",
        result_s3,
        u_s3,
        REPORT_TIMES_S,
    )

    # -------------------------------------------------------------------
    # Hypothesis triage (Scenario 1 vs Scenario 2 baseline)
    # -------------------------------------------------------------------
    diagnose_hypotheses(
        result_step=result_s1,
        result_baseline=result_s2,
        u_step=u_s1,
        change_step=change_step,
    )

    print()


if __name__ == "__main__":
    main()
