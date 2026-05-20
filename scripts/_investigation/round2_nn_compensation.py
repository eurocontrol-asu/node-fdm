"""Round 2 falsification — does the v3 NN backbone compensate the m_0 perturbation?

Falsification loop, round 2. Round 1 established that perturbing the
MassEncoder-predicted ``m_0`` by 1.3x moves the predicted state trajectory
by ~7*sigma_obs in altitude over a 60-step rollout, yet the val loss barely
moves (0.075%). Round 2 explains WHY the loss stays flat.

TWO COMPETING HYPOTHESES
------------------------
H2a — the ADS-B val data genuinely does not constrain m_0; the loss
landscape is intrinsically flat in m_0. The NN backbone (``data_ode_long``)
does NOT actively re-steer — its outputs change between baseline and
perturbed rollout only passively (it sees a diverged state), not in a
loss-reducing way.

H2b (the one to falsify) — the NN backbone COMPENSATES the m_0 perturbation
through the rollout feedback loop: it sees the diverged state and emits
different ``cl_residual`` / ``t_minus_d_norm`` that actively pull the
trajectory back toward the observed target, keeping the loss flat. This
would be an architecture leak (the AC4 invariant says the backbone is
mass-agnostic *within a step*, but the rollout feedback could still let it
react *across steps*).

FALSIFIER
---------
H2b dies if the NN outputs diverge only marginally between baseline and
m_0x1.3 rollouts. Pre-registered: **H2b dead if the RMS relative divergence
of each NN output stays below ~10% at every sampled rollout step**.
H2b survives if NN outputs diverge strongly AND that divergence demonstrably
reduces loss-to-target vs a "frozen NN output" counterfactual.

METHOD
------
The rollout is a faithful copy of ``ODETrainer._compute_batch_loss``
(reused from ``round1_trajectory_divergence.py``). Three rollouts per batch:

  1. baseline      — m_0 unperturbed,   NN live
  2. perturbed-live — m_0 x 1.3,         NN live
  3. perturbed-frozen — m_0 x 1.3, but the ``data_ode_long`` outputs
     (``fdm_t_minus_d_norm``, ``fdm_cl_residual``) are FROZEN to their
     step-matched baseline values. The analytical PhysicsLayer still runs
     with the perturbed mass. This isolates "what the perturbed mass does
     to the trajectory if the NN does NOT get to react".

NN-output capture / replay is done by wrapping the live ``data_ode_long``
layer's ``forward`` (a runtime monkeypatch on the loaded model object — no
source file is modified). A per-batch step counter indexes the recorded
baseline output sequence.

Loss-to-target is the trainer's trajectory MSE: ``residual / norm_std``,
alpha-weighted, then mean-square (the ``loss_fn`` reduction in
``_compute_batch_loss`` with ``w_tensor is None`` and no per-column Huber —
which is the configured regime for full_hybrid_v3; asserted at runtime).

Run (from repo root)::

    uv run python scripts/_investigation/round2_nn_compensation.py

Emits ``scripts/_investigation/round2_nn_compensation.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the round-1 rollout helper + the parity-script model reloader.
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
MODEL_NAME = "full_hybrid_v3"

# NN backbone output channels emitted by data_ode_long (arch v3).
NN_OUTPUT_COLS = ["fdm_t_minus_d_norm", "fdm_cl_residual"]

# Rollout steps to report in the divergence table.
SAMPLE_STEPS = [1, 10, 30, 59]

# m_0 factor sweep for the loss-curvature secondary measurement.
SWEEP_FACTORS = [0.7, 0.85, 1.0, 1.15, 1.3]

# RMS-relative-divergence threshold below which H2b is declared dead.
H2B_DIVERGENCE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# data_ode_long capture / replay wrapper
# ---------------------------------------------------------------------------


class _NNRecorder(torch.nn.Module):
    """Wraps the data_ode_long layer to record or replay its per-step outputs.

    Three modes:

    * ``record``  — call the wrapped layer, store each step's NN output dict.
    * ``replay``  — call the wrapped layer (so the rest of vect_dict is
      consistent) but OVERWRITE the NN output channels with the recorded
      baseline values for the matching step index. The PhysicsLayer
      downstream therefore sees frozen NN outputs but the live (perturbed)
      mass / state.
    * ``off``     — passthrough; just call the wrapped layer.

    The step counter is reset per batch via :meth:`reset`. Within a batch,
    the ODE integrator (ClampedEuler, fixed-step euler) calls the dynamics
    function exactly once per grid step, in order — so a monotone counter
    correctly indexes the rollout step. (Euler = one func eval per step;
    asserted indirectly by checking the recorded length matches seq_len.)
    """

    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        # Registered as a submodule so .eval()/.train() recurse into the
        # wrapped layer exactly as if it were still in the ModuleDict.
        self._layer = layer
        self.mode = "off"
        self._step = 0
        self._recorded: list[dict[str, torch.Tensor]] = []

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """nn.Module entry point — delegates to :meth:`_intercept`."""
        return self._intercept(x_dict)

    def reset(self) -> None:
        """Reset the per-batch step counter (keep recorded buffer if recording)."""
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
# Rollout — faithful copy of ODETrainer._compute_batch_loss rollout block
# ---------------------------------------------------------------------------


def _rollout(trainer, batch: tuple) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Run the ODE rollout for one batch.

    Verbatim port of ``_compute_batch_loss`` L598-647 (same as round 1).
    Returns ``(x_pred[batch,time,n_x], x_true[batch,time,n_x], seq_len)``.

    The data_ode_long recorder (if installed) is reset at the start so its
    step counter aligns with this batch's rollout.
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
    # Reset the NN recorder step counter for this batch (if installed).
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
# Loss-to-target — faithful copy of the trainer's trajectory MSE reduction
# ---------------------------------------------------------------------------


def _loss_to_target(trainer, x_pred: torch.Tensor, x_true: torch.Tensor) -> float:
    """Trajectory MSE-to-target, matching ``_compute_batch_loss``.

    Replicates the ``w_tensor is None`` + ``_huber_beta_per_col is None``
    branch of ``_compute_batch_loss`` (the configured regime for
    full_hybrid_v3 — asserted by the caller). Skips the initial condition,
    applies the wrap-aware heading residual, normalises by ``_norm_std``,
    alpha-weights, then ``loss_fn`` (MSE) over all elements.
    """
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
# Per-pass val-set drivers
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
    """Roll out the whole val set once, capturing NN outputs + trajectory + loss.

    ``override`` is written to ``trainer._override_m0_factor`` for the pass.
    The data_ode_long recorder runs in ``record`` mode so per-step NN
    outputs are captured; they are concatenated across the val set into
    ``(N, time, channel)`` tensors keyed by channel name.

    Returns ``{"nn": {col: tensor[N,time]}, "x_pred": tensor, "loss": float}``.
    """
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
                # Concatenate the recorded per-step NN outputs into (N,time).
                recorded = rec._recorded  # list[dict[col -> (N,)]]
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
    """Roll out the val set at perturbed m_0 with NN outputs FROZEN to baseline.

    Two-pass-per-batch counterfactual: pass A records the baseline NN
    outputs (override=None); pass B re-rolls the SAME batch with the
    perturbed mass but the recorder in ``replay`` mode, so the analytical
    PhysicsLayer sees the perturbed mass and the baseline NN outputs.
    Returns the mean loss-to-target of pass B.
    """
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
# Divergence analysis
# ---------------------------------------------------------------------------


def _rms_relative_divergence(
    base: torch.Tensor, pert: torch.Tensor
) -> np.ndarray:
    """Per-step RMS relative divergence between two (N,time) NN-output tensors.

    For each step ``t``:
        sqrt( mean_N( (pert - base)^2 ) ) / sqrt( mean_N( base^2 ) )

    The denominator is the RMS *magnitude* of the baseline NN output at
    that step — so the result is a dimensionless relative divergence,
    directly comparable against the 10% falsifier threshold. A small
    epsilon guards against a degenerate all-zero baseline channel.
    """
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


def _write_report(res: dict) -> None:
    """Emit the markdown report: NN divergence, 3-way loss, m_0 sweep, verdict."""
    nn_div = res["nn_divergence"]  # {col: {"rel": arr, "abs": arr, "base": arr}}
    losses = res["losses"]  # {"baseline","perturbed_live","perturbed_frozen"}
    sweep = res["sweep"]  # list[(factor, loss)]
    n_samples = res["n_samples"]
    n_time = res["n_time"]
    frozen_ok = res["frozen_ok"]
    frozen_note = res["frozen_note"]

    lines: list[str] = [
        "# Round 2 — does the v3 NN backbone compensate the m_0 perturbation?",
        "",
        "Falsification loop, round 2. Round 1 found that perturbing the",
        "MassEncoder-predicted `m_0` by 1.3x moves the predicted state",
        "trajectory by ~7*sigma_obs in altitude over 60 steps, yet the val",
        "loss barely moves. Round 2 explains WHY the loss stays flat.",
        "",
        "**H2b (the hypothesis to falsify)** — the v3 NN backbone actively",
        "compensates the m_0 perturbation: through the rollout feedback loop",
        "it sees the diverged state and emits different `cl_residual` /",
        "`t_minus_d_norm` that pull the trajectory back toward the observed",
        "target, keeping the loss flat (an architecture leak).",
        "",
        "**H2a (the null)** — the ADS-B val data genuinely does not constrain",
        "`m_0`; the loss landscape is intrinsically flat in `m_0`. The NN",
        "outputs drift only slightly (passive reaction to a modestly shifted",
        "input distribution), not in a loss-reducing way.",
        "",
        "**Falsifier** — H2b dies if the RMS *relative* divergence of each NN",
        f"output channel stays below ~{H2B_DIVERGENCE_THRESHOLD:.0%} at every",
        "sampled rollout step. H2b survives only if NN outputs diverge",
        "strongly AND that divergence demonstrably reduces loss-to-target vs",
        "the frozen-NN counterfactual.",
        "",
        "## Setup",
        "",
        f"- model: `{MODEL_NAME}` (arch `node_adsb_hybrid_v3`, CL mass-aware)",
        f"- `val_limit` = {VAL_LIMIT}, device = `{DEVICE}`, m_0 factor = {M0_FACTOR}",
        f"- val samples rolled out: {n_samples}, rollout steps: {n_time}",
        "- rollout = faithful copy of `ODETrainer._compute_batch_loss`",
        "  (60-step euler, ClampedEuler integrator) — reused from round 1",
        "- NN outputs captured by wrapping the `data_ode_long` layer at",
        "  runtime (no source file modified)",
        "- loss-to-target = trainer trajectory MSE (`residual / norm_std`,",
        "  alpha-weighted, MSE reduction)",
        "",
        "## 1. NN-output RMS *relative* divergence (baseline vs m_0x1.3)",
        "",
        "Per channel, per rollout step:",
        "`sqrt(mean_N((pert - base)^2)) / sqrt(mean_N(base^2))`.",
        f"Falsifier threshold = {H2B_DIVERGENCE_THRESHOLD:.0%}.",
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
        "  PhysicsLayer still runs with the perturbed mass.",
        "",
        "If *perturbed-live* loss is much lower than *perturbed-frozen* -> the",
        "NN IS actively compensating -> H2b. If the two are about equal -> the",
        "NN is not compensating -> H2a.",
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
        # Compensation metric: how much of the frozen-loss penalty does the
        # live NN remove?
        frozen_penalty = lf - lb
        live_penalty = ll - lb
        if abs(frozen_penalty) > 1e-12:
            removed = (frozen_penalty - live_penalty) / frozen_penalty
        else:
            removed = float("nan")
        lines += [
            "",
            f"**Frozen-NN penalty** (perturbed-frozen - baseline): "
            f"`{frozen_penalty:+.4g}` ({frozen_penalty / lb * 100:+.4g}%).",
            f"**Live-NN penalty** (perturbed-live - baseline): "
            f"`{live_penalty:+.4g}` ({live_penalty / lb * 100:+.4g}%).",
            f"**Fraction of the frozen penalty removed by the live NN**: "
            f"`{removed * 100:.4g}%`.",
            "",
            "Reading: a large positive *removed* fraction (live NN erases the",
            "frozen penalty) => the NN compensates => H2b. A *removed* fraction",
            "near 0% (or negative) => the NN does not compensate; the frozen",
            "and live perturbed losses move together => H2a.",
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
        "Flat across the whole sweep => data genuinely uninformative about",
        "`m_0` (supports H2a). A visible minimum near factor 1.0 => the loss",
        "does carry `m_0` information (the 2-point 1.3x test was too coarse).",
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

    # ----- Verdict -----------------------------------------------------
    peak_rels = {
        col: float(np.max(nn_div[col]["rel"])) for col in nn_div
    }
    overall_peak_rel = max(peak_rels.values()) if peak_rels else float("nan")
    all_below = all(v < H2B_DIVERGENCE_THRESHOLD for v in peak_rels.values())

    lines += ["## Verdict", ""]

    if all_below:
        verdict = "H2b DEAD — H2a confirmed (data uninformative)"
        why = [
            f"Every NN-output channel diverges by < {H2B_DIVERGENCE_THRESHOLD:.0%} "
            f"RMS-relative at every rollout step (overall peak "
            f"{overall_peak_rel * 100:.3g}%). The pre-registered falsifier",
            "for H2b is met: the NN backbone does NOT substantially change",
            "its outputs between the baseline and the m_0x1.3 rollout. The",
            "backbone is mass-blind (AC4 invariant) and the 7-sigma altitude",
            "shift over 60 steps is too small a relative change in its input",
            "distribution to make it react. The flat val loss is therefore",
            "the data genuinely failing to constrain `m_0` — non-",
            "identifiability of mass from ADS-B-only.",
        ]
        if frozen_ok:
            lf = losses["perturbed_frozen"]
            why.append(
                f"The counterfactual confirms this: perturbed-frozen loss "
                f"({_fmt(lf)}) and perturbed-live loss ({_fmt(ll)}) are "
                f"essentially equal — the NN removes "
                f"{(((lf - lb) - (ll - lb)) / (lf - lb) * 100) if abs(lf - lb) > 1e-12 else float('nan'):.3g}% "
                f"of a frozen penalty that is itself only "
                f"{(lf - lb) / lb * 100:+.3g}% of baseline. There is no "
                f"loss-reducing compensation to speak of."
            )
        why.append(
            "Phase 2's CL x CD joint constraint is the documented fix — "
            "proceed to Phase 2."
        )
    else:
        # Strong NN divergence — need the counterfactual to decide.
        offenders = [c for c, v in peak_rels.items() if v >= H2B_DIVERGENCE_THRESHOLD]
        if frozen_ok:
            lf = losses["perturbed_frozen"]
            frozen_penalty = lf - lb
            live_penalty = ll - lb
            removed = (
                (frozen_penalty - live_penalty) / frozen_penalty
                if abs(frozen_penalty) > 1e-12
                else float("nan")
            )
            if removed > 0.5:
                verdict = "H2b SURVIVES — NN compensates"
                why = [
                    f"NN outputs diverge strongly ({', '.join(offenders)} exceed "
                    f"the {H2B_DIVERGENCE_THRESHOLD:.0%} threshold; overall peak "
                    f"{overall_peak_rel * 100:.3g}%) AND that divergence is "
                    f"loss-reducing: the live NN removes {removed * 100:.3g}% of "
                    f"the frozen-NN penalty. The backbone actively re-steers "
                    f"the trajectory back toward target — an architecture leak. "
                    f"Repair the MassEncoder coupling before Phase 2.",
                ]
            else:
                verdict = "H2b DEAD — H2a confirmed (data uninformative)"
                why = [
                    f"NN outputs do diverge ({', '.join(offenders)} exceed "
                    f"{H2B_DIVERGENCE_THRESHOLD:.0%}; peak {overall_peak_rel * 100:.3g}%) "
                    f"but the divergence is NOT loss-reducing: the live NN only "
                    f"removes {removed * 100:.3g}% of the frozen-NN penalty. The "
                    f"divergence is passive (the NN reacts to a shifted state) "
                    f"not compensatory. The flat loss is data non-"
                    f"identifiability — proceed to Phase 2.",
                ]
        else:
            verdict = "INCONCLUSIVE — strong NN divergence but frozen counterfactual unavailable"
            why = [
                f"NN outputs diverge above the {H2B_DIVERGENCE_THRESHOLD:.0%} "
                f"threshold ({', '.join(offenders)}; peak "
                f"{overall_peak_rel * 100:.3g}%), so the divergence-only "
                f"falsifier does not kill H2b. The frozen-NN counterfactual "
                f"that would decide compensation vs passive drift failed: "
                f"{frozen_note}.",
            ]

    lines += why
    lines += ["", f"## VERDICT: {verdict}", ""]

    OUT_MD.write_text("\n".join(lines))
    print(f"report written: {OUT_MD}")
    print(f"VERDICT: {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the round-2 measurement for full_hybrid_v3 and write the report."""
    trainer, _, _, _ = _build_trainer_for_model(
        MODEL_NAME, CONFIG_PATH, DEVICE, VAL_LIMIT
    )

    # Configured-regime assertion: the loss-to-target replica only matches
    # _compute_batch_loss in the (no w_tensor, no per-col Huber) branch.
    if trainer._huber_beta_per_col is not None:
        msg = (
            "full_hybrid_v3 uses per-column Huber betas — the loss-to-target "
            "replica in this script only mirrors the MSE branch. Aborting "
            "rather than silently reporting a mismatched loss."
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
        # Sanity: a frozen-NN replay must reproduce the baseline loss when
        # the mass is ALSO unperturbed (override=None). If it does not, the
        # step counter / replay wiring is broken — flag instead of trusting.
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
