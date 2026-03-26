"""Gradient diagnostic script for node-fdm-v2.

Identifies aberrant data samples that produce extreme gradients,
potentially causing training divergence.

Usage:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/diagnose_gradients.py [--arch opensky] [--typecode A320] [--top 20]
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import structlog
import torch
from torch.utils.data import DataLoader

from node_fdm.dataset import FlightDataset, FlightSample, compute_stats
from node_fdm.loader import get_train_val_data
from node_fdm.losses import get_loss
from node_fdm.models.batch_neural_ode import BatchNeuralODE
from node_fdm.models.fdm import FlightDynamicsModel
from node_fdm.trainer import TrainingConfig, _collate_flight_samples
from node_fdm_pipeline.config import PipelineConfig
from node_fdm_pipeline.resolver import resolve_architecture
from torchdiffeq import odeint

log = structlog.get_logger()


# ── Helpers ──────────────────────────────────────────────────────────


def grad_norm(model: torch.nn.Module) -> float:
    """Total gradient L2 norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def per_param_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    """Per-parameter gradient norms."""
    norms: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = p.grad.data.norm(2).item()
    return norms


# ── Core diagnostic ──────────────────────────────────────────────────


def compute_single_sample_gradient(
    model: FlightDynamicsModel,
    sample: FlightSample,
    config: TrainingConfig,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    alpha_weights: torch.Tensor,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, float]]:
    """Forward + backward on a single sample. Returns (loss, grad_norm, per_param_norms)."""
    model.zero_grad()
    batch = _collate_flight_samples([sample])
    tensors = tuple(t.to(device) for t in batch)
    x_seq, u_seq, e_seq = tensors[0], tensors[1], tensors[2]

    seq_len = x_seq.shape[1]
    x0 = x_seq[:, 0, :]

    t_grid = torch.arange(
        0, seq_len * config.step, config.step,
        dtype=torch.float32, device=device,
    )

    model.reset_history()
    func = BatchNeuralODE(model, u_seq, e_seq, t_grid)

    try:
        x_pred = odeint(func, x0, t_grid, method=config.method)
    except Exception as exc:
        return float("inf"), float("inf"), {"error": str(exc)}

    x_pred = x_pred.permute(1, 0, 2)
    pred = x_pred[:, 1:, :]
    true = x_seq[:, 1:, :]

    pred_norm = (pred - norm_mean) / norm_std
    true_norm = (true - norm_mean) / norm_std
    pred_w = pred_norm * alpha_weights
    true_w = true_norm * alpha_weights

    loss = loss_fn(pred_w, true_w)

    if torch.isnan(loss) or torch.isinf(loss):
        return float(loss.item()), float("inf"), {}

    loss.backward()

    # Sanitize before measuring
    gn = grad_norm(model)
    pn = per_param_grad_norms(model)

    return loss.item(), gn, pn


def diagnose(
    arch: str,
    config_path: Path,
    typecode: str | None,
    top_k: int,
    device_str: str,
) -> None:
    """Run the full gradient diagnostic."""
    cfg = PipelineConfig.from_yaml(config_path)
    info = resolve_architecture(arch)
    importlib.import_module(info.architecture_import)

    typecodes = [typecode] if typecode else cfg.typecodes
    delta_table = cfg.paths.resolve("delta_table")
    if not delta_table.exists():
        print(f"Delta table not found: {delta_table}", file=sys.stderr)
        raise SystemExit(1)

    full_df = pl.read_delta(str(delta_table))
    full_df = full_df.filter(pl.col("fdm_flag_valid"))
    dx_col_names = [col for _, col in info.dx_cols]
    device = torch.device(device_str)

    for acft in typecodes:
        print(f"\n{'='*80}")
        print(f"  GRADIENT DIAGNOSTIC — {arch} / {acft}")
        print(f"{'='*80}\n")

        data_df = full_df.filter(pl.col("meta_aircraft_type") == acft)
        if len(data_df) == 0:
            print(f"  No data for {acft}, skipping.")
            continue

        seq_len = 60
        training_config = TrainingConfig(
            architecture_name=info.name,
            model_name=f"diag_{acft}",
            model_params=(3, 2, 48),
            step=4.0,
            shift=seq_len,
            lr=1e-3,
            seq_len=seq_len,
            batch_size=512,
            epochs=1,
            method="euler",
        )

        train_ds, val_ds = get_train_val_data(
            data_df=data_df,
            x_cols=info.x_cols,
            u_cols=info.u_cols,
            e_cols=info.e0_cols,
            e1_cols=info.e1_cols,
            dx_cols=dx_col_names,
            seq_len=training_config.seq_len,
            shift=training_config.shift,
            train_limit=5000,
            val_limit=500,
            max_step_jump=info.max_step_jump or None,
        )

        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples:   {len(val_ds)}")

        # ── Phase 1: Dataset-level anomaly scan ──────────────────────

        print(f"\n── Phase 1: Data anomaly scan ──")
        all_x = torch.stack([s.x for s in train_ds])  # (N, seq, n_x)
        all_u = torch.stack([s.u for s in train_ds])
        all_e = torch.stack([s.e for s in train_ds])
        all_dx = torch.stack([s.dx for s in train_ds])

        for name, tensor, cols in [
            ("x (state)", all_x, info.x_cols),
            ("u (control)", all_u, info.u_cols),
            ("e (env)", all_e, info.e0_cols),
            ("dx (deriv)", all_dx, dx_col_names),
        ]:
            print(f"\n  {name}:")
            for ci, cname in enumerate(cols):
                col_data = tensor[:, :, ci].numpy()
                mn, mx = np.min(col_data), np.max(col_data)
                mean, std = np.mean(col_data), np.std(col_data)
                p01 = np.percentile(col_data, 0.1)
                p999 = np.percentile(col_data, 99.9)
                ratio = mx / std if std > 1e-10 else float("inf")
                flag = " ⚠️ EXTREME RANGE" if ratio > 100 else ""
                print(f"    {cname:30s}  min={mn:12.4f}  max={mx:12.4f}  "
                      f"mean={mean:12.4f}  std={std:10.4f}  "
                      f"p0.1={p01:12.4f}  p99.9={p999:12.4f}{flag}")

        # Detect per-sample outliers: samples where any column has a value
        # beyond 5 sigma from the column mean
        print(f"\n── Phase 1b: Per-sample outlier detection (>5σ) ──")
        outlier_indices: set[int] = set()
        for name, tensor, cols in [
            ("x", all_x, info.x_cols),
            ("u", all_u, info.u_cols),
            ("e", all_e, info.e0_cols),
        ]:
            for ci, cname in enumerate(cols):
                col_data = tensor[:, :, ci]  # (N, seq)
                col_mean = col_data.mean()
                col_std = col_data.std()
                if col_std < 1e-10:
                    continue
                # Check if any timestep in the sample exceeds 5σ
                z_scores = ((col_data - col_mean) / col_std).abs()
                max_z_per_sample = z_scores.max(dim=1).values  # (N,)
                bad_mask = max_z_per_sample > 5.0
                bad_indices = torch.where(bad_mask)[0].tolist()
                if bad_indices:
                    print(f"  {name}.{cname}: {len(bad_indices)} samples with >5σ values "
                          f"(max z = {max_z_per_sample.max():.1f})")
                    outlier_indices.update(bad_indices[:50])  # cap to avoid flooding

        print(f"\n  Total unique outlier samples: {len(outlier_indices)} / {len(train_ds)}")

        # ── Phase 2: ODE forward divergence analysis ─────────────────

        print(f"\n── Phase 2: ODE forward divergence analysis ──")

        _samples = list(train_ds)
        ode_layer_spec = next(ly for ly in get_arch_spec(info.name).layers if ly.trainable)
        u_ode_cols = [c for c in info.u_cols if c in ode_layer_spec.input_cols]
        stats_dict = compute_stats(
            _samples,
            x_cols=info.x_cols,
            u_cols=u_ode_cols,
            e_cols=info.e0_cols,
            dx_cols=dx_col_names,
        )
        arch_spec = get_arch_spec(info.name)
        model = FlightDynamicsModel(arch_spec, stats_dict, training_config.model_params).to(device)
        loss_fn = get_loss("mse")

        means = [stats_dict.get(c, {"mean": 0.0})["mean"] for c in info.x_cols]
        stds = [stats_dict.get(c, {"std": 1.0})["std"] for c in info.x_cols]
        norm_mean = torch.tensor(means, device=device)
        norm_std = torch.tensor(stds, device=device)

        print(f"\n  Normalization stats used by model:")
        for ci, cname in enumerate(info.x_cols):
            s = stats_dict.get(cname, {})
            print(f"    {cname:30s}  mean={s.get('mean',0):.4f}  "
                  f"std={s.get('std',1):.4f}  max={s.get('max',0):.4f}")

        # ── 2a: Single-step model output test ────────────────────────
        print(f"\n  ── 2a: Single-step model.forward() output ──")
        sample0 = train_ds[0]
        with torch.no_grad():
            x0 = sample0.x[0:1, :].to(device)   # (1, n_x)
            u0 = sample0.u[0:1, :].to(device)    # (1, n_u)
            e0 = sample0.e[0:1, :].to(device)    # (1, n_e)
            model.reset_history()
            dx_out = model(x0, u0, e0)
            print(f"    Input x0:  {x0.numpy().flatten()}")
            print(f"    Input u0:  {u0.numpy().flatten()}")
            print(f"    Input e0:  {e0.numpy().flatten()}")
            print(f"    Output dx: {dx_out.numpy().flatten()}")
            print(f"    dx finite: {torch.isfinite(dx_out).all().item()}")
            print(f"    dx abs max: {dx_out.abs().max().item():.6f}")

        # ── 2b: Step-by-step Euler on first sample ───────────────────
        print(f"\n  ── 2b: Manual Euler integration (first sample, 60 steps) ──")
        with torch.no_grad():
            x_cur = sample0.x[0:1, :].to(device).clone()
            dt = training_config.step
            diverge_step = -1
            for step_i in range(min(60, sample0.x.shape[0])):
                u_t = sample0.u[step_i:step_i+1, :].to(device)
                e_t = sample0.e[step_i:step_i+1, :].to(device)
                model.reset_history()
                dx = model(x_cur, u_t, e_t)
                x_next = x_cur + dt * dx

                has_nan = torch.isnan(x_next).any().item()
                has_inf = torch.isinf(x_next).any().item()
                x_abs_max = x_next.abs().max().item()

                if step_i < 5 or step_i % 10 == 0 or has_nan or has_inf or x_abs_max > 1e6:
                    status = ""
                    if has_nan:
                        status = " ⚠️ NaN!"
                    elif has_inf:
                        status = " ⚠️ Inf!"
                    elif x_abs_max > 1e6:
                        status = " ⚠️ DIVERGING"
                    vals = x_next.numpy().flatten()
                    dx_vals = dx.numpy().flatten()
                    print(f"    step {step_i:3d}: x={vals}  dx={dx_vals}  |dx|_max={dx.abs().max().item():.4f}{status}")

                if has_nan or has_inf:
                    diverge_step = step_i
                    break

                x_cur = x_next

            if diverge_step >= 0:
                print(f"\n    *** ODE diverged at step {diverge_step} ***")
            else:
                print(f"\n    ODE survived 60 steps (max |x| = {x_cur.abs().max().item():.2f})")

        # ── 2c: ODE integration on first batch with step tracking ────
        print(f"\n  ── 2c: Full odeint on first batch (batch_size=16) ──")
        mini_samples = [train_ds[i] for i in range(min(16, len(train_ds)))]
        mini_batch = _collate_flight_samples(mini_samples)
        tensors = tuple(t.to(device) for t in mini_batch)
        x_seq, u_seq, e_seq = tensors[0], tensors[1], tensors[2]

        x0 = x_seq[:, 0, :]
        t_grid = torch.arange(
            0, x_seq.shape[1] * training_config.step, training_config.step,
            dtype=torch.float32, device=device,
        )

        with torch.no_grad():
            model.reset_history()
            func = BatchNeuralODE(model, u_seq, e_seq, t_grid)
            try:
                x_pred = odeint(func, x0, t_grid, method=training_config.method)
                # x_pred: (time, batch, n_x)
                print(f"    odeint succeeded. Shape: {x_pred.shape}")
                for ti in [0, 1, 2, 5, 10, 20, 40, 59]:
                    if ti < x_pred.shape[0]:
                        vals = x_pred[ti]  # (batch, n_x)
                        nan_count = torch.isnan(vals).sum().item()
                        inf_count = torch.isinf(vals).sum().item()
                        abs_max = vals[vals.isfinite()].abs().max().item() if vals.isfinite().any() else float("inf")
                        print(f"    t={ti:3d}: |x|_max={abs_max:12.2f}  "
                              f"NaN={nan_count}  Inf={inf_count}")

                # Check loss
                x_pred_p = x_pred.permute(1, 0, 2)
                pred = x_pred_p[:, 1:, :]
                true = x_seq[:, 1:, :]
                pred_n = (pred - norm_mean) / norm_std
                true_n = (true - norm_mean) / norm_std
                loss = loss_fn(pred_n, true_n)
                print(f"    Loss = {loss.item()}")
            except Exception as exc:
                print(f"    odeint FAILED: {exc}")

        # ── 2d: Per-sample ODE divergence scan ───────────────────────
        print(f"\n  ── 2d: Per-sample divergence scan (all {len(train_ds)} samples) ──")
        diverge_counts: dict[str, int] = {"ok": 0, "nan": 0, "inf": 0, "diverge": 0}
        diverge_samples: list[tuple[int, str, float]] = []

        with torch.no_grad():
            for si in range(len(train_ds)):
                sample = train_ds[si]
                x_cur = sample.x[0:1, :].to(device).clone()
                dt = training_config.step
                status = "ok"
                max_abs = 0.0

                for step_i in range(sample.x.shape[0]):
                    u_t = sample.u[step_i:step_i+1, :].to(device)
                    e_t = sample.e[step_i:step_i+1, :].to(device)
                    model.reset_history()
                    dx = model(x_cur, u_t, e_t)
                    x_cur = x_cur + dt * dx

                    cur_max = x_cur.abs().max().item()
                    if cur_max > max_abs:
                        max_abs = cur_max

                    if torch.isnan(x_cur).any():
                        status = "nan"
                        break
                    if torch.isinf(x_cur).any():
                        status = "inf"
                        break
                    if cur_max > 1e8:
                        status = "diverge"
                        break

                diverge_counts[status] += 1
                if status != "ok":
                    diverge_samples.append((si, status, max_abs))

                if si % 500 == 0:
                    print(f"    ... scanned {si}/{len(train_ds)} samples")

        print(f"\n    Results:")
        print(f"      OK:       {diverge_counts['ok']}")
        print(f"      NaN:      {diverge_counts['nan']}")
        print(f"      Inf:      {diverge_counts['inf']}")
        print(f"      Diverge:  {diverge_counts['diverge']}")

        if diverge_samples:
            print(f"\n    Top 20 worst diverging samples:")
            diverge_samples.sort(key=lambda t: t[2], reverse=True)
            for si, status, mabs in diverge_samples[:20]:
                sample = train_ds[si]
                x_vals = sample.x.numpy()
                u_vals = sample.u.numpy()
                print(f"      idx={si:5d}  status={status:8s}  max_abs={mabs:.2e}  "
                      f"alt=[{x_vals[:,0].min():.0f},{x_vals[:,0].max():.0f}]  "
                      f"gamma=[{x_vals[:,1].min():.4f},{x_vals[:,1].max():.4f}]  "
                      f"tas=[{x_vals[:,2].min():.1f},{x_vals[:,2].max():.1f}]  "
                      f"alt_tgt=[{u_vals[:,0].min():.0f},{u_vals[:,0].max():.0f}]")

        # ── Phase 3: Gradient analysis on non-diverging samples ──────

        ok_indices = [i for i in range(len(train_ds))
                      if i not in {s[0] for s in diverge_samples}]
        bad_indices = [s[0] for s in diverge_samples]

        if ok_indices:
            print(f"\n── Phase 3: Gradient norms on {min(200, len(ok_indices))} OK samples ──")
            alpha_weights = torch.ones(len(info.x_cols), device=device)
            model.train()

            test_indices = ok_indices[:200]
            ok_grads: list[tuple[int, float, float]] = []
            for si in test_indices:
                loss_val, gn_val, _ = compute_single_sample_gradient(
                    model, train_ds[si], training_config,
                    norm_mean, norm_std, alpha_weights,
                    loss_fn, device,
                )
                ok_grads.append((si, loss_val, gn_val))

            finite_grads = [(s, l, g) for s, l, g in ok_grads if g < float("inf") and not np.isnan(g)]
            nan_grads = len(ok_grads) - len(finite_grads)
            print(f"    Finite gradients: {len(finite_grads)}/{len(ok_grads)}  (NaN/Inf: {nan_grads})")

            if finite_grads:
                gn_arr = np.array([g for _, _, g in finite_grads])
                loss_arr = np.array([l for _, l, _ in finite_grads])
                print(f"    Grad norm:  median={np.median(gn_arr):.4f}  p95={np.percentile(gn_arr, 95):.4f}  max={gn_arr.max():.4f}")
                print(f"    Loss:       median={np.median(loss_arr):.6f}  p95={np.percentile(loss_arr, 95):.6f}  max={loss_arr.max():.6f}")

                # Top 10 worst
                finite_grads.sort(key=lambda t: t[2], reverse=True)
                print(f"\n    Top 10 worst gradient samples (among OK samples):")
                for si, sloss, sgn in finite_grads[:10]:
                    print(f"      idx={si}  loss={sloss:.6f}  grad_norm={sgn:.4f}")

        # ── Phase 4: Characterize diverging samples vs OK ────────────

        if diverge_samples:
            print(f"\n── Phase 4: Diverging vs OK sample characteristics ──")
            ok_set = set(ok_indices[:500])
            bad_set = set(bad_indices[:500])

            for ci, cname in enumerate(info.x_cols):
                ok_vals = torch.cat([train_ds[i].x[:, ci] for i in list(ok_set)[:500]])
                bad_vals = torch.cat([train_ds[i].x[:, ci] for i in list(bad_set)[:500]]) if bad_set else torch.tensor([])
                if len(bad_vals) > 0:
                    print(f"  x.{cname}:")
                    print(f"    OK   — mean={ok_vals.mean():.2f}  std={ok_vals.std():.2f}  "
                          f"[{ok_vals.min():.2f}, {ok_vals.max():.2f}]")
                    print(f"    BAD  — mean={bad_vals.mean():.2f}  std={bad_vals.std():.2f}  "
                          f"[{bad_vals.min():.2f}, {bad_vals.max():.2f}]")

            for ci, cname in enumerate(info.u_cols):
                ok_vals = torch.cat([train_ds[i].u[:, ci] for i in list(ok_set)[:500]])
                bad_vals = torch.cat([train_ds[i].u[:, ci] for i in list(bad_set)[:500]]) if bad_set else torch.tensor([])
                if len(bad_vals) > 0:
                    print(f"  u.{cname}:")
                    print(f"    OK   — mean={ok_vals.mean():.2f}  std={ok_vals.std():.2f}  "
                          f"[{ok_vals.min():.2f}, {ok_vals.max():.2f}]")
                    print(f"    BAD  — mean={bad_vals.mean():.2f}  std={bad_vals.std():.2f}  "
                          f"[{bad_vals.min():.2f}, {bad_vals.max():.2f}]")

            for ci, cname in enumerate(info.e0_cols):
                ok_vals = torch.cat([train_ds[i].e[:, ci] for i in list(ok_set)[:500]])
                bad_vals = torch.cat([train_ds[i].e[:, ci] for i in list(bad_set)[:500]]) if bad_set else torch.tensor([])
                if len(bad_vals) > 0:
                    print(f"  e.{cname}:")
                    print(f"    OK   — mean={ok_vals.mean():.2f}  std={ok_vals.std():.2f}  "
                          f"[{ok_vals.min():.2f}, {ok_vals.max():.2f}]")
                    print(f"    BAD  — mean={bad_vals.mean():.2f}  std={bad_vals.std():.2f}  "
                          f"[{bad_vals.min():.2f}, {bad_vals.max():.2f}]")

        print(f"\n{'='*80}")
        print(f"  DIAGNOSTIC COMPLETE — {acft}")
        print(f"{'='*80}\n")


def get_arch_spec(name: str):
    """Get architecture spec from registry."""
    from node_fdm.architectures.registry import get
    return get(name)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient diagnostic for node-fdm-v2")
    parser.add_argument("--arch", default="opensky", help="Architecture (opensky, opensky_v2, qar)")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config path")
    parser.add_argument("--typecode", default=None, help="Single typecode (default: all from config)")
    parser.add_argument("--top", type=int, default=20, help="Top K worst batches to report")
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    args = parser.parse_args()

    diagnose(
        arch=args.arch,
        config_path=Path(args.config),
        typecode=args.typecode,
        top_k=args.top,
        device_str=args.device,
    )
