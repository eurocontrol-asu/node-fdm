"""Test clamping strategies on gradient norms.

Compares 3 configurations on the same data:
  A) Baseline — no clamping (current behavior)
  B) Clamp x only — physical bounds on state in BatchNeuralODE
  C) Clamp x + dx — physical bounds on both state and derivatives

Usage:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/test_clamping.py
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from node_fdm.architectures.registry import get as get_arch_spec
from node_fdm.dataset import FlightSample, compute_stats
from node_fdm.loader import get_train_val_data
from node_fdm.losses import get_loss
from node_fdm.models.batch_neural_ode import BatchNeuralODE
from node_fdm.models.fdm import FlightDynamicsModel
from node_fdm.trainer import TrainingConfig, _collate_flight_samples
from node_fdm_pipeline.config import PipelineConfig
from node_fdm_pipeline.resolver import resolve_architecture

import polars as pl


# ── Physical bounds ──────────────────────────────────────────────────

X_BOUNDS = {
    # (min, max) per x_col index for adsb: [alt, gamma, tas]
    0: (-500.0, 20_000.0),   # alt (m)
    1: (-0.3, 0.3),          # gamma (rad)
    2: (0.0, 350.0),         # TAS (m/s)
}

DX_BOUNDS = {
    # (min, max) per dx_col index: [d_vz, d_gamma, d_tas]
    0: (-50.0, 50.0),        # d_alt (m/s)
    1: (-0.03, 0.03),        # d_gamma (rad/s)
    2: (-5.0, 5.0),          # d_tas (m/s²)
}


# ── Modified ODE wrappers ───────────────────────────────────────────

def _clamp_columns(x: torch.Tensor, bounds: dict[int, tuple[float, float]]) -> torch.Tensor:
    """Clamp specific columns without in-place ops (autograd safe)."""
    cols = []
    for i in range(x.shape[-1]):
        if i in bounds:
            lo, hi = bounds[i]
            cols.append(torch.clamp(x[..., i], lo, hi))
        else:
            cols.append(x[..., i])
    return torch.stack(cols, dim=-1)


def _soft_clamp_columns(x: torch.Tensor, bounds: dict[int, tuple[float, float]]) -> torch.Tensor:
    """Tanh soft clamp on specific columns (autograd safe, gradient never zero)."""
    cols = []
    for i in range(x.shape[-1]):
        if i in bounds:
            lo, hi = bounds[i]
            mid = (hi + lo) / 2.0
            half = (hi - lo) / 2.0
            cols.append(mid + half * torch.tanh((x[..., i] - mid) / half))
        else:
            cols.append(x[..., i])
    return torch.stack(cols, dim=-1)


class ClampedXNeuralODE(nn.Module):
    """BatchNeuralODE with hard clamp on x before passing to model."""

    def __init__(
        self,
        model: nn.Module,
        u_seq: torch.Tensor,
        e_seq: torch.Tensor,
        t_grid: torch.Tensor,
        x_bounds: dict[int, tuple[float, float]],
    ) -> None:
        super().__init__()
        self.inner = BatchNeuralODE(model, u_seq, e_seq, t_grid)
        self.x_bounds = x_bounds

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_clamped = _clamp_columns(x, self.x_bounds)
        return self.inner.forward(t, x_clamped)


class ClampedXDXNeuralODE(nn.Module):
    """BatchNeuralODE with hard clamp on x AND soft clamp on dx."""

    def __init__(
        self,
        model: nn.Module,
        u_seq: torch.Tensor,
        e_seq: torch.Tensor,
        t_grid: torch.Tensor,
        x_bounds: dict[int, tuple[float, float]],
        dx_bounds: dict[int, tuple[float, float]],
    ) -> None:
        super().__init__()
        self.inner = BatchNeuralODE(model, u_seq, e_seq, t_grid)
        self.x_bounds = x_bounds
        self.dx_bounds = dx_bounds

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_clamped = _clamp_columns(x, self.x_bounds)
        dx = self.inner.forward(t, x_clamped)
        return _soft_clamp_columns(dx, self.dx_bounds)


# ── Helpers ──────────────────────────────────────────────────────────

def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def _euler_clamped_integrate(
    model: FlightDynamicsModel,
    x0: torch.Tensor,
    u_seq: torch.Tensor,
    e_seq: torch.Tensor,
    t_grid: torch.Tensor,
    x_bounds: dict[int, tuple[float, float]],
    dx_bounds: dict[int, tuple[float, float]],
) -> torch.Tensor:
    """Euler integration with hard clamp on x after each step + soft clamp dx.

    Returns (batch, time, n_x) — same layout as odeint permuted.
    """
    dt = t_grid[1] - t_grid[0]
    seq_len = t_grid.shape[0]
    x = x0  # (batch, n_x)
    trajectory = [x]

    for i in range(seq_len - 1):
        # Clamp x before model eval (hard)
        x_input = _clamp_columns(x, x_bounds)

        # Interpolate u, e at timestep i
        u_t = u_seq[:, i, :]
        e_t = e_seq[:, i, :]

        model.reset_history()
        dx = model(x_input, u_t, e_t)

        # Soft clamp dx
        dx = _soft_clamp_columns(dx, dx_bounds)

        # Integrate
        x = x + dt * dx

        # Hard clamp x AFTER integration — the key difference vs odeint
        x = _clamp_columns(x, x_bounds)

        trajectory.append(x)

    return torch.stack(trajectory, dim=1)  # (batch, time, n_x)


def run_batch(
    model: FlightDynamicsModel,
    batch: tuple[torch.Tensor, ...],
    config: TrainingConfig,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    mode: str,
) -> dict:
    """Run forward + backward on a batch. Returns metrics."""
    model.zero_grad()
    tensors = tuple(t.to(device) for t in batch)
    x_seq, u_seq, e_seq = tensors[0], tensors[1], tensors[2]

    seq_len = x_seq.shape[1]
    x0 = x_seq[:, 0, :]
    t_grid = torch.arange(
        0, seq_len * config.step, config.step,
        dtype=torch.float32, device=device,
    )

    model.reset_history()

    if mode == "euler_clamped":
        # Custom Euler loop with clamp x after each step
        x_pred = _euler_clamped_integrate(
            model, x0, u_seq, e_seq, t_grid, X_BOUNDS, DX_BOUNDS,
        )
    else:
        if mode == "baseline":
            func = BatchNeuralODE(model, u_seq, e_seq, t_grid)
        elif mode == "clamp_x":
            func = ClampedXNeuralODE(model, u_seq, e_seq, t_grid, X_BOUNDS)
        elif mode == "clamp_x_dx":
            func = ClampedXDXNeuralODE(model, u_seq, e_seq, t_grid, X_BOUNDS, DX_BOUNDS)
        else:
            msg = f"Unknown mode: {mode}"
            raise ValueError(msg)

        try:
            x_pred = odeint(func, x0, t_grid, method=config.method)
        except Exception as exc:
            return {"loss": float("inf"), "grad_norm": float("inf"), "error": str(exc)}

        x_pred = x_pred.permute(1, 0, 2)

    pred = x_pred[:, 1:, :]
    true = x_seq[:, 1:, :]

    pred_n = (pred - norm_mean) / norm_std
    true_n = (true - norm_mean) / norm_std
    loss = loss_fn(pred_n, true_n)
    loss_val = loss.item()

    if not (torch.isfinite(loss)):
        return {"loss": loss_val, "grad_norm": float("inf")}

    loss.backward()

    # Sanitize NaN grads (same as trainer)
    for p in model.parameters():
        if p.grad is not None:
            torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)

    gn = grad_norm(model)

    # Track trajectory stats
    with torch.no_grad():
        x_min = x_pred.min(dim=1).values.min(dim=0).values
        x_max = x_pred.max(dim=1).values.max(dim=0).values

    return {
        "loss": loss_val,
        "grad_norm": gn,
        "x_pred_min": x_min.tolist(),
        "x_pred_max": x_max.tolist(),
    }


def euler_trajectory(
    model: FlightDynamicsModel,
    sample: FlightSample,
    config: TrainingConfig,
    device: torch.device,
    mode: str,
) -> dict:
    """Manual Euler integration to track state evolution."""
    with torch.no_grad():
        x_cur = sample.x[0:1, :].to(device).clone()
        dt = config.step
        trajectory = [x_cur.numpy().flatten().copy()]

        for step_i in range(sample.x.shape[0] - 1):
            u_t = sample.u[step_i:step_i+1, :].to(device)
            e_t = sample.e[step_i:step_i+1, :].to(device)

            # Clamp x before model call
            x_input = x_cur.clone()
            if mode in ("clamp_x", "clamp_x_dx", "euler_clamped"):
                for ci, (lo, hi) in X_BOUNDS.items():
                    x_input[:, ci] = torch.clamp(x_input[:, ci], lo, hi)

            model.reset_history()
            dx = model(x_input, u_t, e_t)

            # Soft clamp dx
            if mode in ("clamp_x_dx", "euler_clamped"):
                for ci, (lo, hi) in DX_BOUNDS.items():
                    mid = (hi + lo) / 2.0
                    half = (hi - lo) / 2.0
                    dx[:, ci] = mid + half * torch.tanh((dx[:, ci] - mid) / half)

            x_cur = x_cur + dt * dx

            # Clamp x after integration (euler_clamped mode)
            if mode == "euler_clamped":
                for ci, (lo, hi) in X_BOUNDS.items():
                    x_cur[:, ci] = torch.clamp(x_cur[:, ci], lo, hi)

            trajectory.append(x_cur.numpy().flatten().copy())

    traj = np.array(trajectory)
    return {
        "alt_min": traj[:, 0].min(),
        "alt_max": traj[:, 0].max(),
        "gamma_min": traj[:, 1].min(),
        "gamma_max": traj[:, 1].max(),
        "tas_min": traj[:, 2].min(),
        "tas_max": traj[:, 2].max(),
        "trajectory": traj,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    cfg = PipelineConfig.from_yaml(Path("config.yaml"))
    info = resolve_architecture("adsb")
    importlib.import_module(info.architecture_import)

    dx_col_names = [col for _, col in info.dx_cols]
    device = torch.device("cpu")

    df = pl.read_delta("data/flights.delta")
    df = df.filter(pl.col("fdm_flag_valid") & (pl.col("meta_aircraft_type") == "A320"))

    thresholds = {"raw_alt_m": 200.0, "era_tas_ms": 15.0, "fdm_gamma_rad": 0.08}
    train_ds, _ = get_train_val_data(
        df, info.x_cols, info.u_cols, info.e0_cols, dx_col_names,
        e1_cols=info.e1_cols, seq_len=60, shift=60,
        train_limit=5000, val_limit=500,
        max_step_jump=thresholds,
    )
    print(f"Train samples: {len(train_ds)}")

    # Build model + stats
    _samples = list(train_ds)
    arch_spec = get_arch_spec(info.name)
    ode_layer_spec = next(ly for ly in arch_spec.layers if ly.trainable)
    u_ode_cols = [c for c in info.u_cols if c in ode_layer_spec.input_cols]
    stats_dict = compute_stats(
        _samples, x_cols=info.x_cols, u_cols=u_ode_cols,
        e_cols=info.e0_cols, dx_cols=dx_col_names,
    )

    training_config = TrainingConfig(
        architecture_name=info.name, model_name="test_clamp",
        model_params=(3, 2, 48), step=4.0, shift=60,
        seq_len=60, batch_size=512, epochs=1, method="euler",
    )

    means = [stats_dict.get(c, {"mean": 0.0})["mean"] for c in info.x_cols]
    stds = [stats_dict.get(c, {"std": 1.0})["std"] for c in info.x_cols]
    norm_mean = torch.tensor(means, device=device)
    norm_std = torch.tensor(stds, device=device)
    loss_fn = get_loss("mse")

    modes = ["baseline", "clamp_x", "clamp_x_dx", "euler_clamped"]

    # ── Test 1: Euler trajectory on first sample ─────────────────

    print(f"\n{'='*70}")
    print("  TEST 1: Euler trajectory (sample 0, 60 steps)")
    print(f"{'='*70}")

    # Use same model weights for all modes
    torch.manual_seed(42)
    model = FlightDynamicsModel(arch_spec, stats_dict, training_config.model_params).to(device)
    state_dict = model.state_dict()

    for mode in modes:
        model.load_state_dict(state_dict)
        traj = euler_trajectory(model, train_ds[0], training_config, device, mode)
        print(f"\n  [{mode}]")
        print(f"    alt:   [{traj['alt_min']:10.1f}, {traj['alt_max']:10.1f}] m")
        print(f"    gamma: [{traj['gamma_min']:10.4f}, {traj['gamma_max']:10.4f}] rad")
        print(f"    tas:   [{traj['tas_min']:10.1f}, {traj['tas_max']:10.1f}] m/s")

    # ── Test 2: Gradient norms on 200 samples ────────────────────

    print(f"\n{'='*70}")
    print("  TEST 2: Gradient norms (200 samples, individual)")
    print(f"{'='*70}")

    n_test = 200
    results: dict[str, list[dict]] = {m: [] for m in modes}

    for mode in modes:
        model.load_state_dict(state_dict)
        model.train()

        for si in range(min(n_test, len(train_ds))):
            batch = _collate_flight_samples([train_ds[si]])
            r = run_batch(model, batch, training_config,
                          norm_mean, norm_std, loss_fn, device, mode)
            results[mode].append(r)

        # Stats
        gn = [r["grad_norm"] for r in results[mode] if r["grad_norm"] < float("inf")]
        losses = [r["loss"] for r in results[mode] if r["loss"] < float("inf")]
        n_inf = sum(1 for r in results[mode] if r["grad_norm"] == float("inf"))
        n_nan = sum(1 for r in results[mode] if np.isnan(r["grad_norm"]))

        gn_arr = np.array(gn) if gn else np.array([0.0])
        loss_arr = np.array(losses) if losses else np.array([0.0])

        print(f"\n  [{mode}] ({len(gn)} finite, {n_inf} inf, {n_nan} nan)")
        print(f"    grad_norm:  median={np.median(gn_arr):10.2f}  "
              f"p95={np.percentile(gn_arr, 95):10.2f}  "
              f"max={gn_arr.max():10.2f}")
        print(f"    loss:       median={np.median(loss_arr):10.4f}  "
              f"p95={np.percentile(loss_arr, 95):10.4f}  "
              f"max={loss_arr.max():10.4f}")

    # ── Test 3: Batch gradients (full batches) ───────────────────

    print(f"\n{'='*70}")
    print("  TEST 3: Gradient norms (full batches of 512)")
    print(f"{'='*70}")

    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_ds, batch_size=512, shuffle=False,
        num_workers=0, collate_fn=_collate_flight_samples,
    )
    batches = list(loader)

    for mode in modes:
        model.load_state_dict(state_dict)
        model.train()

        batch_gn = []
        batch_loss = []
        for batch in batches:
            r = run_batch(model, batch, training_config,
                          norm_mean, norm_std, loss_fn, device, mode)
            batch_gn.append(r["grad_norm"])
            batch_loss.append(r["loss"])

        gn_arr = np.array([g for g in batch_gn if g < float("inf") and not np.isnan(g)])
        loss_arr = np.array([l for l in batch_loss if l < float("inf") and not np.isnan(l)])
        n_bad = len(batch_gn) - len(gn_arr)

        print(f"\n  [{mode}] ({len(gn_arr)} finite batches, {n_bad} bad)")
        if len(gn_arr) > 0:
            print(f"    grad_norm:  median={np.median(gn_arr):10.2f}  "
                  f"p95={np.percentile(gn_arr, 95):10.2f}  "
                  f"max={gn_arr.max():10.2f}")
            print(f"    loss:       median={np.median(loss_arr):10.4f}  "
                  f"p95={np.percentile(loss_arr, 95):10.4f}  "
                  f"max={loss_arr.max():10.4f}")
        else:
            print("    All batches NaN/Inf!")

    # ── Test 4: Trajectory comparison on worst sample ────────────

    print(f"\n{'='*70}")
    print("  TEST 4: Step-by-step on worst sample (highest baseline grad)")
    print(f"{'='*70}")

    # Find worst sample from baseline
    baseline_grads = [(i, r["grad_norm"]) for i, r in enumerate(results["baseline"])
                      if r["grad_norm"] < float("inf")]
    if baseline_grads:
        worst_idx = max(baseline_grads, key=lambda t: t[1])[0]
        print(f"\n  Worst sample idx={worst_idx} "
              f"(baseline grad={baseline_grads[worst_idx][1] if worst_idx < len(baseline_grads) else '?'})")

        for mode in modes:
            model.load_state_dict(state_dict)
            traj = euler_trajectory(model, train_ds[worst_idx], training_config, device, mode)
            t = traj["trajectory"]

            print(f"\n  [{mode}]")
            for step in [0, 5, 10, 20, 30, 40, 50, 59]:
                if step < len(t):
                    print(f"    step {step:2d}: alt={t[step,0]:10.1f}  "
                          f"gamma={t[step,1]:+8.4f}  tas={t[step,2]:8.1f}")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
