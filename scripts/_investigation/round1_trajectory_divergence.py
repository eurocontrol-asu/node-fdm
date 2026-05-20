"""Round 1 falsification — m_0 trajectory-divergence measurement.

Falsification loop, round 1. Hypothesis H1: ``m_0`` is genuinely weakly
identifiable in ``full_hybrid_v3`` (CL mass-aware). Concretely: perturbing
the MassEncoder-predicted initial mass ``m_0`` by factor 1.3 displaces the
*predicted state trajectory* by < 1*sigma_obs per state dimension at the
end of the 60-step rollout.

CRITICAL DISTINCTION (the whole point of the round): this script measures

    || z_pred_perturbed[step] - z_pred_baseline[step] ||      (RMS over batch)

prediction-vs-prediction, NOT prediction-vs-observed-target. ``identifiability_test``
in trainer.py reports a *loss ratio* (distance to the observed target dx);
a perturbation can shift the predicted trajectory "sideways" while keeping
the same MSE-distance to target. This script reads the trajectory itself.

Method — the rollout block below is a *faithful copy* of the rollout in
``ODETrainer._compute_batch_loss`` (trainer.py ~L598-647): same ``x0``
construction, same ``_override_m0_factor`` injection point, same
``BatchNeuralODE`` + ``ClampedEuler`` integrator, same ``t_grid``. The
only change is that it returns the predicted trajectory tensor ``x_pred``
(batch, time, n_x) instead of reducing it to a scalar loss.

State vector ``x_cols`` (hybrid schema, 5 dims):
    [raw_alt_m, fdm_gamma_rad, era_tas_ms, fdm_heading_rad, fdm_mass_kg]
Indices: altitude=0, gamma=1, TAS=2, heading=3, mass=4.

Observation-noise floors (ADS-B; PHASE_1.5 doc s6/s10):
    altitude  ~ 5     m
    TAS       ~ 1     m/s
    gamma     ~ 1e-3  rad

Run (from repo root)::

    uv run python scripts/_investigation/round1_trajectory_divergence.py

Emits ``scripts/_investigation/round1_trajectory_divergence.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the exact in-process model reload helper from the parity script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase15_parity_report import _build_trainer_for_model  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"
OUT_MD = Path(__file__).resolve().with_suffix(".md")

DEVICE = "cpu"
VAL_LIMIT = 2000
M0_FACTOR = 1.3

# State-vector column indices (hybrid schema X_COLS).
IDX = {"altitude": 0, "gamma": 1, "TAS": 2, "heading": 3, "mass": 4}

# ADS-B observation-noise floors per state dimension (PHASE_1.5 doc s6/s10).
SIGMA_OBS = {
    "altitude": 5.0,    # m
    "gamma": 1.0e-3,    # rad
    "TAS": 1.0,         # m/s
}

# Rollout steps to report in the divergence table.
SAMPLE_STEPS = [1, 10, 30, 59]

MODELS = [
    ("full_hybrid_v3", "CL mass-aware (arch node_adsb_hybrid_v3) — hypothesis target"),
    ("full_hybrid_v2", "Newton baseline (arch node_adsb_hybrid_v2) — CONTROL"),
]


# ---------------------------------------------------------------------------
# Rollout — faithful copy of ODETrainer._compute_batch_loss rollout block
# ---------------------------------------------------------------------------


def _rollout_trajectory(trainer, batch: tuple) -> torch.Tensor:
    """Run the ODE rollout for one batch and return ``x_pred``.

    Verbatim port of trainer.py ``_compute_batch_loss`` L598-647 — same
    ``x0`` build, same ``_override_m0_factor`` hook, same integrator. The
    ``_override_m0_factor`` attribute is read from ``trainer`` exactly as
    the original does, so setting it on the trainer before this call
    perturbs ``m_0`` identically to ``identifiability_test``.

    Returns ``x_pred`` of shape ``(batch, time, n_x)``.
    """
    from node_fdm.models.batch_neural_ode import BatchNeuralODE
    from node_fdm.models.projected_integrator import (
        ClampedEuler,
        ClampedRK4,
        _clamp_columns,
    )
    from torchdiffeq import odeint

    if batch[0].device == trainer.device:
        tensors = batch
    else:
        tensors = tuple(t.to(trainer.device, non_blocking=True) for t in batch)
    x_seq, u_seq, e_seq = tensors[0], tensors[1], tensors[2]

    flight_features: torch.Tensor | None = None
    n_features = len(getattr(trainer.spec, "flight_feature_cols", []) or [])
    for t in tensors[4:]:
        if t.ndim == 3 and n_features > 0 and t.shape[-1] == n_features:
            flight_features = t

    seq_len = x_seq.shape[1]
    x0 = x_seq[:, 0, :]
    if trainer.mass_encoder is not None and flight_features is not None:
        m_0 = trainer.mass_encoder(flight_features[:, 0, :])
        if trainer._override_m0_factor is not None:
            m_0 = m_0 * trainer._override_m0_factor
        x0 = torch.cat([x0[:, :4], m_0.to(x0.dtype).unsqueeze(-1)], dim=-1)

    t_grid = torch.arange(
        0,
        seq_len * trainer.config.step,
        trainer.config.step,
        dtype=torch.float32,
        device=trainer.device,
    )

    trainer.model.reset_history()

    x_bounds_idx = trainer._resolve_bounds(trainer.spec.x_bounds, trainer.spec.x_cols)
    dx_bounds_idx = trainer._resolve_bounds(trainer.spec.dx_bounds, trainer.spec.dx_cols)

    func = BatchNeuralODE(trainer.model, u_seq, e_seq, t_grid, dx_bounds=dx_bounds_idx)

    if x_bounds_idx:
        project_fn = lambda x: _clamp_columns(x, x_bounds_idx)  # noqa: E731
        method = trainer.config.method
        solver_kwargs = {"atol": 1e-6, "rtol": 1e-3, "step_size": trainer.config.step}
        if method == "euler":
            solver = ClampedEuler(func, x0, project_fn=project_fn, **solver_kwargs)
            x_pred = solver.integrate(t_grid)
        elif method == "rk4":
            solver = ClampedRK4(func, x0, project_fn=project_fn, **solver_kwargs)
            x_pred = solver.integrate(t_grid)
        else:
            msg = f"Method '{method}' does not support state projection."
            raise ValueError(msg)
    else:
        x_pred = odeint(func, x0, t_grid, method=trainer.config.method)

    # (time, batch, n_x) -> (batch, time, n_x)
    return x_pred.permute(1, 0, 2)


def _collect_trajectories(trainer, override: float | None) -> torch.Tensor:
    """Roll out the whole val set; return a (N_total, time, n_x) tensor.

    ``override`` is written to ``trainer._override_m0_factor`` for the
    duration of the pass — same mechanism as ``identifiability_test``.
    """
    from torch.utils.data import DataLoader

    from node_fdm.trainer import _collate_flight_samples

    val_loader = DataLoader(
        trainer.val_dataset,
        batch_size=trainer.config.val_batch_size,
        shuffle=False,
        num_workers=trainer.config.num_workers,
        collate_fn=_collate_flight_samples,
    )
    trainer.model.eval()
    chunks: list[torch.Tensor] = []
    trainer._override_m0_factor = override
    try:
        with torch.no_grad():
            for batch in val_loader:
                chunks.append(_rollout_trajectory(trainer, batch).detach().cpu())
    finally:
        trainer._override_m0_factor = None
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Divergence analysis
# ---------------------------------------------------------------------------


def _divergence_curve(
    z_base: torch.Tensor, z_pert: torch.Tensor
) -> dict[str, np.ndarray]:
    """Per-dim RMS-over-batch trajectory divergence as a function of step.

    Returns ``{dim: array[time]}`` where ``array[step]`` is
    ``sqrt( mean_batch( (z_pert - z_base)^2 ) )`` for that state dim at
    that rollout step — the prediction-vs-prediction displacement.
    """
    diff = (z_pert - z_base).numpy()  # (N, time, n_x)
    out: dict[str, np.ndarray] = {}
    for dim, idx in IDX.items():
        if dim == "heading" or dim == "mass":
            # heading wraps; mass is the perturbed quantity itself — not a
            # validatable observed state. Keep them only for context.
            out[dim] = np.sqrt(np.mean(diff[:, :, idx] ** 2, axis=0))
        else:
            out[dim] = np.sqrt(np.mean(diff[:, :, idx] ** 2, axis=0))
    return out


def _analyze(model_name: str, desc: str) -> dict:
    """Reload one model, roll out baseline + perturbed, compute divergence."""
    trainer, _, _, meta = _build_trainer_for_model(
        model_name, CONFIG_PATH, DEVICE, VAL_LIMIT
    )
    has_mass_encoder = trainer.mass_encoder is not None

    z_base = _collect_trajectories(trainer, None)
    z_pert = _collect_trajectories(trainer, M0_FACTOR)

    n_samples, n_time, _ = z_base.shape
    curves = _divergence_curve(z_base, z_pert)

    # Sanity: how much did m_0 actually move? (mass dim, step 0 already
    # perturbed at x0). Report the realised initial-mass perturbation.
    init_mass_base = z_base[:, 0, IDX["mass"]].numpy()
    init_mass_pert = z_pert[:, 0, IDX["mass"]].numpy()
    mass_shift_kg = float(np.sqrt(np.mean((init_mass_pert - init_mass_base) ** 2)))
    mass_base_mean = float(np.mean(init_mass_base))

    return {
        "model": model_name,
        "desc": desc,
        "arch": meta["architecture_name"],
        "has_mass_encoder": has_mass_encoder,
        "n_samples": n_samples,
        "n_time": n_time,
        "curves": curves,
        "mass_base_mean_kg": mass_base_mean,
        "mass_shift_kg": mass_shift_kg,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _final_step_verdict(result: dict) -> tuple[str, dict[str, float]]:
    """Classify a model's final-step divergence against the falsifier.

    Falsifier dims = altitude, gamma, TAS. Returns ``(verdict, ratios)``
    where ``ratios[dim]`` = final-step divergence / sigma_obs[dim].
    """
    final = result["n_time"] - 1
    ratios: dict[str, float] = {}
    for dim in SIGMA_OBS:
        ratios[dim] = float(result["curves"][dim][final] / SIGMA_OBS[dim])
    max_ratio = max(ratios.values())
    if max_ratio > 2.0:
        verdict = "DIVERGES (>2 sigma_obs)"
    elif max_ratio < 1.0:
        verdict = "NOISE-FLOOR (<1 sigma_obs all dims)"
    else:
        verdict = "INTERMEDIATE (1-2 sigma_obs)"
    return verdict, ratios


def _write_report(results: list[dict]) -> None:
    """Emit the markdown report with the divergence table + verdict."""
    lines: list[str] = [
        "# Round 1 — m_0 trajectory-divergence measurement",
        "",
        "Falsification loop, round 1. **H1**: `m_0` is genuinely weakly",
        "identifiable in `full_hybrid_v3` (CL mass-aware) — perturbing the",
        "predicted initial mass by factor 1.3 displaces the *predicted state",
        "trajectory* by < 1*sigma_obs per state dim at the end of the 60-step",
        "rollout.",
        "",
        "**Falsifier**: H1 dies if the final-step divergence",
        "`||z_perturbed - z_baseline||` exceeds **2*sigma_obs** on any of",
        "{altitude, gamma, TAS}. H1 survives if divergence stays **< 1*sigma_obs**",
        "on all dims.",
        "",
        "**Measured quantity** — prediction-vs-prediction:",
        "`sqrt( mean_batch( (z_pred_perturbed[step] - z_pred_baseline[step])^2 ) )`",
        "per state dim. This is NOT the loss-vs-target distance that",
        "`trainer.identifiability_test` reports.",
        "",
        "## Setup",
        "",
        f"- `val_limit` = {VAL_LIMIT}, device = `{DEVICE}`, m_0 factor = {M0_FACTOR}",
        f"- Rollout = faithful copy of `ODETrainer._compute_batch_loss` "
        f"(60-step euler, ClampedEuler integrator)",
        "- sigma_obs (ADS-B, PHASE_1.5 doc s6/s10): "
        f"altitude = {SIGMA_OBS['altitude']:g} m, "
        f"TAS = {SIGMA_OBS['TAS']:g} m/s, "
        f"gamma = {SIGMA_OBS['gamma']:g} rad",
        "",
    ]

    for r in results:
        lines += [
            f"## {r['model']} — {r['desc']}",
            "",
            f"- architecture: `{r['arch']}`",
            f"- MassEncoder present: `{r['has_mass_encoder']}`",
            f"- val samples rolled out: {r['n_samples']}, rollout steps: {r['n_time']}",
            f"- baseline mean m_0: {r['mass_base_mean_kg']:.4g} "
            f"(note: x_cols mass channel; see caveat below)",
            f"- realised RMS m_0 shift (perturbed - baseline): "
            f"{r['mass_shift_kg']:.4g}",
            "",
            "### Divergence curve — RMS(z_pert - z_base) at sampled steps",
            "",
            "| step | altitude (m) | gamma (rad) | TAS (m/s) |",
            "|-----:|-------------:|------------:|----------:|",
        ]
        for step in SAMPLE_STEPS:
            if step >= r["n_time"]:
                continue
            a = r["curves"]["altitude"][step]
            g = r["curves"]["gamma"][step]
            t = r["curves"]["TAS"][step]
            lines.append(
                f"| {step} | {a:.6g} | {g:.6g} | {t:.6g} |"
            )

        verdict, ratios = _final_step_verdict(r)
        final = r["n_time"] - 1
        lines += [
            "",
            f"### Final-step (step {final}) divergence in units of sigma_obs",
            "",
            "| dim | divergence | sigma_obs | divergence / sigma_obs |",
            "|-----|-----------:|----------:|-----------------------:|",
        ]
        for dim in ("altitude", "gamma", "TAS"):
            d = r["curves"][dim][final]
            lines.append(
                f"| {dim} | {d:.6g} | {SIGMA_OBS[dim]:g} | {ratios[dim]:.4g} |"
            )
        lines += [
            "",
            f"**{r['model']} verdict**: `{verdict}` "
            f"(max ratio = {max(ratios.values()):.4g}).",
            "",
        ]

    # ----- Cross-model verdict ------------------------------------------
    v3 = next(r for r in results if r["model"] == "full_hybrid_v3")
    v2 = next(r for r in results if r["model"] == "full_hybrid_v2")
    v3_verdict, v3_ratios = _final_step_verdict(v3)
    v2_verdict, v2_ratios = _final_step_verdict(v2)
    v3_max = max(v3_ratios.values())
    v2_max = max(v2_ratios.values())

    if v3_max < 1.0:
        hypo = "H1 SURVIVES"
        why = (
            "v3 final-step divergence stays below 1*sigma_obs on every dim — "
            "perturbing m_0 by 1.3x genuinely does not move the v3 trajectory. "
            "m_0 is weakly identifiable in the CL formulation."
        )
    elif v3_max > 2.0:
        hypo = "H1 DEAD — H2 confirmed"
        why = (
            "v3 final-step divergence exceeds 2*sigma_obs on at least one dim — "
            "m_0 DOES move the predicted trajectory substantially. The flat "
            "val_loss masked a real trajectory effect (sideways shift keeping "
            "MSE-to-target constant)."
        )
    else:
        hypo = f"INCONCLUSIVE — v3 max ratio {v3_max:.3g} falls in the 1-2 sigma_obs band"
        why = (
            "v3 final-step divergence lands between the survive (<1) and "
            "falsify (>2) thresholds — neither H1 nor H2 is cleanly supported."
        )

    lines += [
        "## Cross-model verdict",
        "",
        "| model | role | final-step max ratio | classification |",
        "|-------|------|---------------------:|----------------|",
        f"| full_hybrid_v3 | CL mass-aware (target) | {v3_max:.4g} | {v3_verdict} |",
        f"| full_hybrid_v2 | Newton (control) | {v2_max:.4g} | {v2_verdict} |",
        "",
        "The Newton control (v2) is expected to be identifiable — a large "
        "divergence there confirms the measurement actually detects an m_0 "
        "effect when one exists.",
        "",
        "### Caveat — x_cols mass channel",
        "",
        "`fdm_mass_kg` in `x_seq` is zero-padded in the val loader (stats_dict "
        "shows mean/std ~ 0). The perturbation is applied to the "
        "MassEncoder-predicted `m_0` injected into `x0` (trainer.py L601-604) — "
        "this matches `identifiability_test` exactly. The `baseline mean m_0` "
        "row reflects the x_cols channel, not the encoder output magnitude; "
        "the `realised RMS m_0 shift` row is the meaningful perturbation size.",
        "",
        f"## VERDICT: {hypo}",
        "",
        why,
        "",
    ]

    OUT_MD.write_text("\n".join(lines))
    print(f"report written: {OUT_MD}")
    print(f"VERDICT: {hypo}")


def main() -> int:
    """Run the divergence measurement for both models and write the report."""
    results = [_analyze(name, desc) for name, desc in MODELS]
    _write_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
