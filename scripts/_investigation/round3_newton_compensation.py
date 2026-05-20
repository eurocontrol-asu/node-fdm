"""Round 3 falsification — does the Newton v2 NN backbone leak the same way as v3?

Falsification loop, round 3. Round 2 established that in ``full_hybrid_v3``
(CL-mass-aware), perturbing the MassEncoder-predicted ``m_0`` by 1.3x leaves
the val loss flat because the NN backbone (``data_ode_long``) actively
compensates the perturbation through the rollout feedback loop:

* NN outputs (``fdm_cl_residual``, ``fdm_t_minus_d_norm``) diverge by 51-71%
  RMS-relative vs the unperturbed baseline.
* The frozen-NN counterfactual (perturbed mass, NN replayed from baseline)
  raises the loss by +239% — but the live NN removes 99.97% of that penalty.
* AC4 ("backbone never sees m as input") is preserved syntactically, but the
  closed-loop ``trajectory <-> backbone`` path leaks mass information through
  the diverged state.

OPEN QUESTION (round 3): does the Newton baseline ``full_hybrid_v2`` (arch
``node_adsb_hybrid_v2``, formulation
``L = lift_residual_norm * m_ref * g + m * g``) leak the SAME way?

H3a (the hypothesis to falsify) — Newton v2 also exhibits near-total NN
compensation under ``m_0 x 1.3`` perturbation: frozen-NN loss penalty is
large (>100x baseline drift) and the live-NN recovery is >= 95% (matching
v3's 99.97%).

H3a survives -> v3 is just more efficient at a leak both formulations share;
rolling back to v2 saves nothing. H3a dies if Newton v2's live-NN recovery
is < 80% (Cas B / Cas C).

THREE OUTCOMES
--------------
* Cas A — recovery >= 95%: both formulations leak; v3 just more efficient.
* Cas B — recovery 60-80%: v2 leaks materially less; rollback worth considering.
* Cas C — recovery < 40%: v2 essentially clean; v3 uniquely pathological.

METHOD — IDENTICAL TO ROUND 2
-----------------------------
Three rollouts per batch (faithful copy of ``ODETrainer._compute_batch_loss``):

  1. baseline       — m_0 unperturbed,   NN live
  2. perturbed-live — m_0 x 1.3,         NN live
  3. perturbed-frozen — m_0 x 1.3, but ``data_ode_long`` outputs FROZEN to
     step-matched baseline values. The analytical PhysicsLayer still runs
     with the perturbed mass, but in the Newton branch — i.e. it reconstructs
     ``L = lift_residual_norm * m_ref * g + m * g`` instead of v3's
     ``L = q * S * (m * g / (q * S) + cl_residual) = m * g + q * S * cl_residual``.

THE ONLY V2-SPECIFIC ADAPTATION
-------------------------------
v2's ``data_ode_long`` emits ``(fdm_t_minus_d_norm, fdm_lift_residual_norm)``
instead of v3's ``(fdm_t_minus_d_norm, fdm_cl_residual)``. The capture/replay
machinery keys off ``NN_OUTPUT_COLS`` so swapping the channel names is
sufficient — no other code path needs to know which lift-reconstruction
formulation is active (the PhysicsLayer branches automatically on which
NN-output channel is present in its input dict).

Run (from repo root)::

    uv run python scripts/_investigation/round3_newton_compensation.py

Emits ``scripts/_investigation/round3_newton_compensation.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the round-2 helpers verbatim (capture wrapper, rollout, loss).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase15_parity_report import _build_trainer_for_model  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — only MODEL_NAME and NN_OUTPUT_COLS differ from round 2
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"
OUT_MD = Path(__file__).resolve().with_suffix(".md")

DEVICE = "cpu"
VAL_LIMIT = 2000
M0_FACTOR = 1.3
MODEL_NAME = "full_hybrid_v2"

# NN backbone output channels emitted by data_ode_long for arch
# ``node_adsb_hybrid_v2`` (Newton formulation). The CL-mode counterpart
# ``fdm_cl_residual`` is REPLACED by ``fdm_lift_residual_norm`` here; the
# PhysicsLayer branches on which channel is present (cf. layers/physics.py
# §L179-235).
NN_OUTPUT_COLS = ["fdm_t_minus_d_norm", "fdm_lift_residual_norm"]

SAMPLE_STEPS = [1, 10, 30, 59]
SWEEP_FACTORS = [0.7, 0.85, 1.0, 1.15, 1.3]
H3A_RECOVERY_SURVIVE = 0.95  # >= 95% -> Cas A (H3a survives)
H3A_RECOVERY_FALSIFY = 0.80  # < 80%  -> Cas B / C (H3a dies)
CAS_B_LOWER = 0.60
CAS_C_UPPER = 0.40


# ---------------------------------------------------------------------------
# data_ode_long capture / replay wrapper — identical to round 2
# ---------------------------------------------------------------------------


class _NNRecorder(torch.nn.Module):
    """Wraps the data_ode_long layer to record or replay its per-step outputs.

    Modes (verbatim from round 2):

    * ``record``  — call the wrapped layer, store each step's NN output dict.
    * ``replay``  — call the wrapped layer (so the rest of vect_dict stays
      consistent) but OVERWRITE the NN output channels with the recorded
      baseline values for the matching step index.
    * ``off``     — passthrough.

    The step counter is reset per batch via :meth:`reset`. Within a batch,
    the ODE integrator (ClampedEuler) calls the dynamics function once per
    grid step in order, so a monotone counter correctly indexes the rollout
    step.
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self._layer = layer
        self.mode = "off"
        self._step = 0
        self._recorded: list[dict[str, torch.Tensor]] = []

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """nn.Module entry point — delegates to :meth:`_intercept`."""
        return self._intercept(x_dict)

    def reset(self) -> None:
        """Reset per-batch step counter (keep recorded buffer if recording)."""
        self._step = 0

    def start_record(self) -> None:
        """Begin a fresh recording pass."""
        self.mode = "record"
        self._step = 0
        self._recorded = []

    def start_replay(self) -> None:
        """Replay the previously recorded outputs on the next pass."""
        self.mode = "replay"
        self._step = 0

    def off(self) -> None:
        """Disable interception (plain passthrough)."""
        self.mode = "off"
        self._step = 0

    @property
    def recorded_len(self) -> int:
        """Number of recorded steps in the last record pass."""
        return len(self._recorded)

    def _intercept(
        self, x_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Record / replay / passthrough around the wrapped layer call."""
        out = self._layer(x_dict)
        if self.mode == "record":
            self._recorded.append(
                {c: out[c].detach().clone() for c in NN_OUTPUT_COLS if c in out}
            )
            self._step += 1
            return out
        if self.mode == "replay":
            frozen = dict(out)
            if self._step < len(self._recorded):
                rec = self._recorded[self._step]
                for c in NN_OUTPUT_COLS:
                    if c in rec and c in frozen:
                        frozen[c] = rec[c]
            self._step += 1
            return frozen
        return out


# ---------------------------------------------------------------------------
# Rollout — faithful copy of ODETrainer._compute_batch_loss (verbatim round 2)
# ---------------------------------------------------------------------------


def _rollout(trainer, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run the ODE rollout for one batch.

    Verbatim port of ``_compute_batch_loss`` L598-647. The NN recorder (if
    installed) is reset at the start so its step counter aligns with this
    batch's rollout.
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
    rec = getattr(trainer, "_nn_recorder", None)
    if rec is not None:
        rec.reset()

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

    return x_pred.permute(1, 0, 2), x_seq, seq_len


# ---------------------------------------------------------------------------
# Loss-to-target — faithful copy of trainer's trajectory MSE (verbatim round 2)
# ---------------------------------------------------------------------------


def _loss_to_target(trainer, x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    """Trajectory MSE-to-target, matching ``_compute_batch_loss`` (MSE branch)."""
    import math

    pred = x_pred[:, 1:, :]
    true = x_true[:, 1:, :]
    residual = pred - true
    if trainer._heading_idx is not None:
        two_pi = 2.0 * math.pi
        raw = residual[..., trainer._heading_idx]
        wrapped = ((raw + math.pi) % two_pi) - math.pi
        residual = residual.clone()
        residual[..., trainer._heading_idx] = wrapped
    residual_norm = residual / trainer._norm_std
    residual_weighted = residual_norm * trainer._alpha_weights
    loss = trainer.loss_fn(residual_weighted, torch.zeros_like(residual_weighted))
    return float(loss.item())


# ---------------------------------------------------------------------------
# Per-pass drivers — identical to round 2
# ---------------------------------------------------------------------------


def _val_loader(trainer):
    """Build the val DataLoader exactly like ``identifiability_test``."""
    from torch.utils.data import DataLoader

    from node_fdm.trainer import _collate_flight_samples

    return DataLoader(
        trainer.val_dataset,
        batch_size=trainer.config.val_batch_size,
        shuffle=False,
        num_workers=trainer.config.num_workers,
        collate_fn=_collate_flight_samples,
    )


def _capture_nn_and_traj(trainer, override: float | None) -> dict:
    """Roll out the val set, capturing NN outputs + trajectory + loss."""
    rec: _NNRecorder = trainer._nn_recorder
    trainer.model.eval()
    trainer._override_m0_factor = override

    nn_chunks: dict[str, list[torch.Tensor]] = {c: [] for c in NN_OUTPUT_COLS}
    x_pred_chunks: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0
    try:
        with torch.no_grad():
            for batch in _val_loader(trainer):
                rec.start_record()
                x_pred, x_true, seq_len = _rollout(trainer, batch)
                recorded = rec._recorded
                for col in NN_OUTPUT_COLS:
                    steps = [d[col] for d in recorded if col in d]
                    if steps:
                        nn_chunks[col].append(torch.stack(steps, dim=1).cpu())
                x_pred_chunks.append(x_pred.detach().cpu())
                total_loss += _loss_to_target(trainer, x_pred, x_true)
                n_batches += 1
    finally:
        trainer._override_m0_factor = None
        rec.off()

    nn_out = {
        c: torch.cat(nn_chunks[c], dim=0) for c in NN_OUTPUT_COLS if nn_chunks[c]
    }
    return {
        "nn": nn_out,
        "x_pred": torch.cat(x_pred_chunks, dim=0),
        "loss": total_loss / max(n_batches, 1),
    }


def _frozen_nn_loss(trainer, override: float | None) -> float:
    """Roll out the val set at perturbed m_0 with NN outputs FROZEN to baseline."""
    rec: _NNRecorder = trainer._nn_recorder
    trainer.model.eval()

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in _val_loader(trainer):
            # Pass A — baseline m_0, record NN outputs.
            trainer._override_m0_factor = None
            rec.start_record()
            _rollout(trainer, batch)
            # Pass B — perturbed m_0, replay frozen NN outputs.
            trainer._override_m0_factor = override
            rec.start_replay()
            x_pred, x_true, _ = _rollout(trainer, batch)
            total_loss += _loss_to_target(trainer, x_pred, x_true)
            n_batches += 1
    trainer._override_m0_factor = None
    rec.off()
    return total_loss / max(n_batches, 1)


def _sweep_loss(trainer, factor: float | None) -> float:
    """Mean val loss-to-target at one m_0 factor (recorder disabled)."""
    rec: _NNRecorder = trainer._nn_recorder
    rec.off()
    trainer.model.eval()
    trainer._override_m0_factor = factor
    total_loss = 0.0
    n_batches = 0
    try:
        with torch.no_grad():
            for batch in _val_loader(trainer):
                x_pred, x_true, _ = _rollout(trainer, batch)
                total_loss += _loss_to_target(trainer, x_pred, x_true)
                n_batches += 1
    finally:
        trainer._override_m0_factor = None
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Divergence analysis — identical to round 2
# ---------------------------------------------------------------------------


def _rms_relative_divergence(
    base: torch.Tensor, pert: torch.Tensor
) -> np.ndarray:
    """Per-step RMS *relative* divergence between two (N,time) NN-output tensors."""
    b = base.numpy()
    p = pert.numpy()
    diff_rms = np.sqrt(np.mean((p - b) ** 2, axis=0))
    base_rms = np.sqrt(np.mean(b**2, axis=0))
    eps = 1e-12
    return diff_rms / np.maximum(base_rms, eps)


def _abs_divergence(base: torch.Tensor, pert: torch.Tensor) -> np.ndarray:
    """Per-step absolute RMS divergence (same numerator, no normalisation)."""
    b = base.numpy()
    p = pert.numpy()
    return np.sqrt(np.mean((p - b) ** 2, axis=0))


def _base_rms(base: torch.Tensor) -> np.ndarray:
    """Per-step RMS magnitude of a baseline NN-output channel."""
    return np.sqrt(np.mean(base.numpy() ** 2, axis=0))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v: float) -> str:
    """Compact float formatting for table cells."""
    return f"{v:.6g}"


# v3 numbers from scripts/_investigation/round2_nn_compensation.md, for the
# side-by-side comparison cell. If round 2 is ever rerun and the file
# regenerated with different numbers, update these — they live here on
# purpose (separate artefact, no shared state).
V3_REFERENCE = {
    "baseline_loss": 0.0226051,
    "perturbed_live_loss": 0.0226223,
    "perturbed_frozen_loss": 0.0767264,
    "live_penalty_pct": 0.07613,
    "frozen_penalty_pct": 239.4,
    "recovery_pct": 99.97,
    "channel_peaks": {
        "fdm_t_minus_d_norm": 51.23,
        "fdm_cl_residual": 70.78,
    },
}


def _classify(recovery: float) -> tuple[str, str]:
    """Map recovery fraction -> (cas_label, verdict_line)."""
    if recovery >= H3A_RECOVERY_SURVIVE:
        return (
            "Cas A",
            "Cas A — both formulations leak equally",
        )
    if recovery < CAS_C_UPPER:
        return (
            "Cas C",
            "Cas C — v2 essentially clean",
        )
    if recovery < H3A_RECOVERY_FALSIFY and recovery >= CAS_B_LOWER:
        return (
            "Cas B",
            "Cas B — v2 leaks materially less, rollback worth considering",
        )
    # 80% <= recovery < 95% -> ambiguous band (per mission brief)
    if recovery >= H3A_RECOVERY_FALSIFY:
        return (
            "AMBIGUOUS",
            f"INCONCLUSIVE — recovery {recovery * 100:.3g}% falls in the "
            f"80-95% ambiguity band",
        )
    # 40% <= recovery < 60% -> between Cas B and Cas C
    return (
        "Cas B-/C+",
        f"INCONCLUSIVE — recovery {recovery * 100:.3g}% sits between Cas B "
        f"and Cas C bands",
    )


def _write_report(res: dict) -> None:
    """Emit the markdown report: NN divergence, 3-way loss, m_0 sweep, verdict."""
    nn_div = res["nn_divergence"]
    losses = res["losses"]
    sweep = res["sweep"]
    n_samples = res["n_samples"]
    n_time = res["n_time"]
    frozen_ok = res["frozen_ok"]
    frozen_note = res["frozen_note"]

    lines: list[str] = [
        "# Round 3 — does the Newton v2 NN backbone leak the same way as v3?",
        "",
        "Falsification loop, round 3. Round 2 found that in `full_hybrid_v3`",
        "(CL mass-aware), the NN backbone removes 99.97% of the frozen-NN",
        "penalty under m_0 x 1.3 — an architecture leak through the rollout",
        "feedback loop, despite AC4 (backbone never sees `m` as input) being",
        "syntactically preserved. Round 3 measures whether `full_hybrid_v2`",
        "(Newton formulation, `L = lift_residual_norm * m_ref * g + m * g`)",
        "leaks the SAME way.",
        "",
        "**H3a (the hypothesis to falsify)** — Newton v2 also exhibits",
        "near-total NN compensation: live-NN recovery >= 95% (matching v3's",
        "99.97%). If true, v3 is just a more efficient version of a leak both",
        "formulations share; rolling back to v2 saves nothing.",
        "",
        "**Falsifier** — H3a dies if recovery < 80% (Cas B: v2 leaks",
        "materially less; rollback worth considering — or Cas C: v2",
        "essentially clean, < 40%). The 80-95% band is ambiguous.",
        "",
        "## Setup",
        "",
        f"- model: `{MODEL_NAME}` (arch `node_adsb_hybrid_v2`, Newton)",
        f"- `val_limit` = {VAL_LIMIT}, device = `{DEVICE}`, m_0 factor = {M0_FACTOR}",
        f"- val samples rolled out: {n_samples}, rollout steps: {n_time}",
        "- rollout = faithful copy of `ODETrainer._compute_batch_loss`",
        "  (60-step euler, ClampedEuler integrator) — reused from round 2",
        "- NN outputs captured by wrapping the `data_ode_long` layer at",
        "  runtime (no source file modified)",
        "- NN output channels (Newton): `fdm_t_minus_d_norm`, "
        "`fdm_lift_residual_norm`",
        "  (vs v3's `fdm_t_minus_d_norm`, `fdm_cl_residual`)",
        "- PhysicsLayer auto-branches on which lift channel is present "
        "(`layers/physics.py` §L179-235)",
        "- loss-to-target = trainer trajectory MSE (`residual / norm_std`,",
        "  alpha-weighted, MSE reduction)",
        "",
        "## 1. NN-output RMS *relative* divergence (baseline vs m_0x1.3)",
        "",
        "Per channel, per rollout step:",
        "`sqrt(mean_N((pert - base)^2)) / sqrt(mean_N(base^2))`.",
        "",
    ]

    for col in NN_OUTPUT_COLS:
        if col not in nn_div:
            lines.append(f"_(channel `{col}` not captured)_\n")
            continue
        d = nn_div[col]
        n_steps_captured = d["rel"].shape[0]
        lines += [
            f"### `{col}`",
            "",
            f"(captured {n_steps_captured} NN-output steps for a {n_time}-state",
            "rollout — euler does seq_len-1 dynamics evaluations per batch.)",
            "",
            "| step | RMS rel. divergence | abs. divergence | baseline RMS magnitude |",
            "|-----:|--------------------:|----------------:|-----------------------:|",
        ]
        for step in SAMPLE_STEPS:
            if step >= n_steps_captured:
                continue
            lines.append(
                f"| {step} | {d['rel'][step] * 100:.4g}% | "
                f"{_fmt(d['abs'][step])} | {_fmt(d['base'][step])} |"
            )
        max_rel = float(np.max(d["rel"]))
        max_rel_step = int(np.argmax(d["rel"]))
        lines += [
            "",
            f"Peak RMS relative divergence: **{max_rel * 100:.4g}%** "
            f"(step {max_rel_step}).",
            "",
        ]

    # ----- 3-way loss comparison ---------------------------------------
    lines += [
        "## 2. Three-way loss-to-target comparison (the counterfactual)",
        "",
        "Counterfactual question — *is the NN-output divergence loss-reducing?*",
        "",
        "* **baseline** — m_0 unperturbed, NN live.",
        "* **perturbed, NN live** — m_0 x 1.3, NN free to react across steps.",
        "* **perturbed, NN frozen** — m_0 x 1.3, but `data_ode_long` outputs",
        "  frozen to their step-matched baseline values; the analytical",
        "  PhysicsLayer still runs with the perturbed mass (Newton branch).",
        "",
        "| rollout | loss-to-target | Δ vs baseline | Δ vs baseline (rel) |",
        "|---------|---------------:|--------------:|--------------------:|",
    ]
    lb = losses["baseline"]
    ll = losses["perturbed_live"]
    lines.append(f"| baseline | {_fmt(lb)} | 0 | 0% |")
    lines.append(
        f"| perturbed, NN live | {_fmt(ll)} | {ll - lb:+.3g} | "
        f"{(ll - lb) / lb * 100:+.4g}% |"
    )
    recovery = float("nan")
    if frozen_ok:
        lf = losses["perturbed_frozen"]
        lines.append(
            f"| perturbed, NN frozen | {_fmt(lf)} | {lf - lb:+.3g} | "
            f"{(lf - lb) / lb * 100:+.4g}% |"
        )
        sanity = losses.get("sanity_frozen_at_1.0", float("nan"))
        lines.append(
            f"| _sanity: frozen at m_0x1.0_ | {_fmt(sanity)} | "
            f"{sanity - lb:+.3g} | {(sanity - lb) / lb * 100:+.4g}% |"
        )
        lines += [
            "",
            "_Sanity check_ — replaying the baseline NN outputs at m_0x1.0 "
            "(unperturbed mass) must reproduce the baseline loss exactly. "
            "Any drift would indicate a broken step counter / replay wiring.",
        ]
        frozen_penalty = lf - lb
        live_penalty = ll - lb
        if abs(frozen_penalty) > 1e-12:
            recovery = (frozen_penalty - live_penalty) / frozen_penalty
        lines += [
            "",
            f"**Frozen-NN penalty** (perturbed-frozen - baseline): "
            f"`{frozen_penalty:+.4g}` ({frozen_penalty / lb * 100:+.4g}%).",
            f"**Live-NN penalty** (perturbed-live - baseline): "
            f"`{live_penalty:+.4g}` ({live_penalty / lb * 100:+.4g}%).",
            f"**Recovery ratio** "
            f"`1 - (live_penalty / frozen_penalty)` = **{recovery * 100:.4g}%**.",
            "",
        ]
    else:
        lines += [
            "",
            f"**Frozen-NN counterfactual: {frozen_note}**",
            "",
        ]

    # ----- m_0 sweep ---------------------------------------------------
    lines += [
        "## 3. Loss-curvature sweep over m_0 factor",
        "",
        "`identifiability_test`-style val loss at a range of m_0 factors.",
        "",
        "| m_0 factor | val loss-to-target | Δ vs factor 1.0 (rel) |",
        "|-----------:|-------------------:|----------------------:|",
    ]
    loss_at_1 = next(loss for f, loss in sweep if abs(f - 1.0) < 1e-9)
    for f, loss in sweep:
        rel = (loss - loss_at_1) / loss_at_1 * 100
        lines.append(f"| {f:g} | {_fmt(loss)} | {rel:+.4g}% |")
    sweep_losses = [loss for _, loss in sweep]
    sweep_span = (max(sweep_losses) - min(sweep_losses)) / loss_at_1 * 100
    argmin_factor = sweep[int(np.argmin(sweep_losses))][0]
    lines += [
        "",
        f"Sweep loss span (max-min)/loss@1.0 = **{sweep_span:.4g}%**. "
        f"Argmin at factor **{argmin_factor:g}**.",
        "",
    ]

    # ----- Side-by-side with v3 ---------------------------------------
    lines += [
        "## 4. Side-by-side: v2 (Newton, this round) vs v3 (CL, round 2)",
        "",
        "| metric | v2 (Newton) | v3 (CL) |",
        "|---|---:|---:|",
        f"| baseline loss | {_fmt(lb)} | {_fmt(V3_REFERENCE['baseline_loss'])} |",
        f"| perturbed-live loss | {_fmt(ll)} | "
        f"{_fmt(V3_REFERENCE['perturbed_live_loss'])} |",
    ]
    if frozen_ok:
        lf = losses["perturbed_frozen"]
        lines += [
            f"| perturbed-frozen loss | {_fmt(lf)} | "
            f"{_fmt(V3_REFERENCE['perturbed_frozen_loss'])} |",
            f"| live penalty (% baseline) | {(ll - lb) / lb * 100:+.4g}% | "
            f"+{V3_REFERENCE['live_penalty_pct']:.4g}% |",
            f"| frozen penalty (% baseline) | {(lf - lb) / lb * 100:+.4g}% | "
            f"+{V3_REFERENCE['frozen_penalty_pct']:.4g}% |",
            f"| recovery ratio | **{recovery * 100:.4g}%** | "
            f"**{V3_REFERENCE['recovery_pct']:.4g}%** |",
        ]
    lines += [
        f"| NN-out peak div. (t_minus_d_norm) | "
        f"{float(np.max(nn_div['fdm_t_minus_d_norm']['rel'])) * 100:.4g}% | "
        f"{V3_REFERENCE['channel_peaks']['fdm_t_minus_d_norm']:.4g}% |",
        f"| NN-out peak div. (lift channel) | "
        + (
            f"{float(np.max(nn_div['fdm_lift_residual_norm']['rel'])) * 100:.4g}%"
            if "fdm_lift_residual_norm" in nn_div
            else "n/a"
        )
        + f" *(lift_residual_norm)* | "
        f"{V3_REFERENCE['channel_peaks']['fdm_cl_residual']:.4g}% "
        "*(cl_residual)* |",
        "",
        "Note: the lift-channel rows compare DIFFERENT physical quantities "
        "(v2's `lift_residual_norm` and v3's `cl_residual`); they are listed "
        "side-by-side because each is the lift-NN-output of its respective "
        "PhysicsLayer branch, not because they are directly commensurable.",
        "",
    ]

    # ----- Verdict -----------------------------------------------------
    lines += ["## Verdict", ""]
    if not frozen_ok:
        verdict = f"INCONCLUSIVE — {frozen_note}"
        lines += [
            f"Frozen-NN counterfactual unavailable: {frozen_note}. Cannot "
            f"compute recovery ratio, so H3a is neither confirmed nor falsified.",
            "",
        ]
    else:
        cas_label, verdict = _classify(recovery)
        lines += [
            f"Recovery ratio = **{recovery * 100:.4g}%**. Classification: "
            f"**{cas_label}**.",
            "",
        ]
        if cas_label == "Cas A":
            lines += [
                f"v2 removes {recovery * 100:.4g}% of the frozen-NN penalty, "
                f"matching v3's {V3_REFERENCE['recovery_pct']:.4g}%. The Newton "
                f"formulation leaks the SAME way: the rollout feedback loop "
                f"reads the diverged state and feeds compensating NN outputs "
                f"into the analytical PhysicsLayer. v3 is not uniquely "
                f"pathological — it's just the same leak with a different "
                f"lift reconstruction. Rolling back to v2 saves nothing; the "
                f"fix must repair the MassEncoder coupling itself.",
                "",
            ]
        elif cas_label == "Cas B":
            lines += [
                f"v2 removes only {recovery * 100:.4g}% of the frozen-NN "
                f"penalty (vs v3's {V3_REFERENCE['recovery_pct']:.4g}%). The "
                f"Newton formulation leaks materially less than the CL one. "
                f"v3 broke something specific to the CL reconstruction; "
                f"rollback to v2 is worth considering as a partial mitigation "
                f"while the deeper coupling fix is designed.",
                "",
            ]
        elif cas_label == "Cas C":
            lines += [
                f"v2 removes only {recovery * 100:.4g}% of the frozen-NN "
                f"penalty (vs v3's {V3_REFERENCE['recovery_pct']:.4g}%). The "
                f"Newton formulation is essentially clean. v3 is uniquely "
                f"pathological — the CL formulation `L = m·g + q·S·cl_residual` "
                f"introduced the leak. Investigate the CL branch specifically "
                f"before Phase 2.",
                "",
            ]
        else:
            lines += [
                f"Recovery ratio = {recovery * 100:.4g}% falls outside the "
                f"three pre-registered bands. The result is qualitatively "
                f"intermediate; report numbers and decide downstream.",
                "",
            ]

    lines += [f"## VERDICT: {verdict}", ""]

    OUT_MD.write_text("\n".join(lines))
    print(f"report written: {OUT_MD}")
    print(f"VERDICT: {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the round-3 measurement for full_hybrid_v2 and write the report."""
    trainer, _, _, _ = _build_trainer_for_model(
        MODEL_NAME, CONFIG_PATH, DEVICE, VAL_LIMIT
    )

    # Configured-regime assertion (same as round 2): the loss-to-target
    # replica only matches _compute_batch_loss in the (no w_tensor, no
    # per-col Huber) branch.
    if trainer._huber_beta_per_col is not None:
        msg = (
            f"{MODEL_NAME} uses per-column Huber betas — the loss-to-target "
            f"replica in this script only mirrors the MSE branch. Aborting "
            f"rather than silently reporting a mismatched loss."
        )
        raise RuntimeError(msg)

    # Install the NN recorder by wrapping the live data_ode_long layer.
    long_layer = trainer.model.layers_dict["data_ode_long"]
    recorder = _NNRecorder(long_layer)
    trainer.model.layers_dict["data_ode_long"] = recorder  # type: ignore[assignment]
    trainer._nn_recorder = recorder  # type: ignore[attr-defined]

    # --- Pass 1+2: baseline + perturbed-live, capturing NN outputs --------
    base = _capture_nn_and_traj(trainer, override=None)
    pert = _capture_nn_and_traj(trainer, override=M0_FACTOR)

    n_samples = base["x_pred"].shape[0]
    n_time = base["x_pred"].shape[1]

    nn_divergence: dict[str, dict[str, np.ndarray]] = {}
    for col in NN_OUTPUT_COLS:
        if col in base["nn"] and col in pert["nn"]:
            b = base["nn"][col]
            p = pert["nn"][col]
            nn_divergence[col] = {
                "rel": _rms_relative_divergence(b, p),
                "abs": _abs_divergence(b, p),
                "base": _base_rms(b),
            }

    # --- Pass 3: frozen-NN counterfactual --------------------------------
    frozen_ok = True
    frozen_note = ""
    frozen_loss = float("nan")
    sanity = float("nan")
    try:
        frozen_loss = _frozen_nn_loss(trainer, override=M0_FACTOR)
        sanity = _frozen_nn_loss(trainer, override=None)
        if abs(sanity - base["loss"]) / max(abs(base["loss"]), 1e-12) > 1e-3:
            frozen_ok = False
            frozen_note = (
                f"replay sanity check failed: frozen-NN replay at m_0x1.0 "
                f"gave loss {sanity:.6g} vs baseline {base['loss']:.6g} "
                f"(>0.1% mismatch) — replay wiring not trustworthy"
            )
    except Exception as exc:  # noqa: BLE001 — document any blocker, don't crash
        frozen_ok = False
        frozen_note = f"frozen-NN counterfactual raised {type(exc).__name__}: {exc}"

    losses = {
        "baseline": base["loss"],
        "perturbed_live": pert["loss"],
        "perturbed_frozen": frozen_loss,
        "sanity_frozen_at_1.0": sanity,
    }

    # --- Secondary: m_0 factor sweep -------------------------------------
    sweep: list[tuple[float, float]] = []
    for f in SWEEP_FACTORS:
        factor = None if abs(f - 1.0) < 1e-9 else f
        sweep.append((f, _sweep_loss(trainer, factor)))

    _write_report(
        {
            "nn_divergence": nn_divergence,
            "losses": losses,
            "sweep": sweep,
            "n_samples": n_samples,
            "n_time": n_time,
            "frozen_ok": frozen_ok,
            "frozen_note": frozen_note,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
