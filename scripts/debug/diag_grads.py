"""Ad-hoc gradient flow diagnostic for the PhysicsLayer refactor.

Builds the same training pipeline as the CLI, pulls one batch, runs 5
gradient steps, and prints magnitudes at every level: head outputs
(raw/denormed), gradient norms on each parameter, gradient flowing back
through PhysicsLayer.

Run from project root:
    uv run python scripts/debug/diag_grads.py
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from node_fdm.loader import get_train_val_data
from node_fdm.trainer import ODETrainer, TrainingConfig
from node_fdm_pipeline.config import PipelineConfig
from node_fdm_pipeline.resolver import resolve_architecture


def _hook_dict_module(name: str, mod: torch.nn.Module, store: dict) -> None:
    """Hook a module that returns dict[str, Tensor]."""
    def fwd(_m, _inp, out) -> None:
        if isinstance(out, dict):
            for k, v in out.items():
                if torch.is_tensor(v):
                    store[f"{name}/{k}"] = v.detach().clone()
    mod.register_forward_hook(fwd)


def main() -> None:
    cfg_path = Path("config.yaml").resolve()
    cfg = PipelineConfig.from_yaml(cfg_path)
    info = resolve_architecture("adsb")
    importlib.import_module(info.architecture_import)

    import polars as pl
    delta_table = cfg.paths.resolve("delta_table")
    full_df = pl.read_delta(str(delta_table)).filter(pl.col("fdm_flag_valid"))
    data_df = full_df.filter(pl.col("meta_aircraft_type") == "A320")
    dx_col_names = [c for _, c in info.dx_cols]

    train_ds, val_ds = get_train_val_data(
        data_df=data_df,
        x_cols=info.x_cols,
        u_cols=info.u_cols,
        e_cols=info.e0_cols,
        e1_cols=info.e1_cols,
        dx_cols=dx_col_names,
        seq_len=60,
        shift=60,
        train_limit=2000,
        val_limit=500,
    )

    training_config = TrainingConfig(
        architecture_name=info.name,
        model_name=f"{info.name}_DIAG",
        model_params=(3, 2, 48),
        step=4.0,
        shift=60,
        lr=1e-3,
        weight_decay=1e-4,
        seq_len=60,
        batch_size=64,
        epochs=1,
        method="rk4",
        num_workers=0,
        lambda_tracking=0.0,
        grad_clip_norm=10.0,
    )

    models_dir = cfg.paths.resolve("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    trainer = ODETrainer(
        config=training_config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_dir=models_dir,
        device="cpu",
    )
    model = trainer.model

    print("=" * 80)
    print("Architecture layers:", model.layers_name)
    print("Trainable params per layer:")
    for name, layer in model.layers_dict.items():
        n = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"  {name}: {n}")

    data_ode = model.layers_dict["data_ode"]
    print("\n--- Head init state ---")
    for col, head in data_ode.heads.layer_dict.items():
        last = head.net[-1]
        print(
            f"  head[{col}]: |W|.max={last.weight.abs().max().item():.3e}, "
            f"bias={last.bias.item():.3e}"
        )
    print(
        f"  scale[a_spec]={data_ode.denormalizer.scale_fdm_a_spec_ms2.item():.3f} "
        f"cap={data_ode.denormalizer.cap_fdm_a_spec_ms2.item():.3f}"
    )
    print(
        f"  scale[n_z_resid]={data_ode.denormalizer.scale_fdm_n_z_residual.item():.3f} "
        f"cap={data_ode.denormalizer.cap_fdm_n_z_residual.item():.3f}"
    )

    activations: dict[str, Any] = {}
    _hook_dict_module("data_ode", data_ode, activations)
    _hook_dict_module("physics", model.layers_dict["physics"], activations)

    # Capture PhysicsLayer's INPUT dict (what it sees from upstream)
    physics_inputs: dict[str, Any] = {}

    def _capture_physics_input(_m, args, _kwargs):
        x_dict = args[0] if args else _kwargs.get("x")
        if isinstance(x_dict, dict):
            for k in (
                "fdm_a_spec_ms2",
                "fdm_n_z_residual",
                "era_tas_ms",
                "fdm_gamma_rad",
            ):
                if k in x_dict and torch.is_tensor(x_dict[k]):
                    physics_inputs[k] = x_dict[k].detach().clone()

    model.layers_dict["physics"].register_forward_pre_hook(
        _capture_physics_input, with_kwargs=True
    )

    batch = next(iter(trainer.train_loader))

    print("\n" + "=" * 80)
    print("Gradient flow over 5 steps (same batch repeated)")
    print("=" * 80)
    torch.autograd.set_detect_anomaly(True)
    for step in range(5):
        trainer.optimizer.zero_grad()
        loss = trainer._compute_batch_loss(batch)  # noqa: SLF001
        loss.backward()  # type: ignore[no-untyped-call]

        a_spec = activations.get("data_ode/fdm_a_spec_ms2")
        n_z_r = activations.get("data_ode/fdm_n_z_residual")
        d_tas = activations.get("physics/fdm_d_tas_ms2")
        d_gamma = activations.get("physics/fdm_d_gamma_rads")

        grad = {}
        for col in ("fdm_a_spec_ms2", "fdm_n_z_residual"):
            head = data_ode.heads.layer_dict[col]
            last = head.net[-1]
            grad[f"{col}/W"] = (
                last.weight.grad.abs().mean().item()
                if last.weight.grad is not None else float("nan")
            )
            grad[f"{col}/b"] = (
                last.bias.grad.abs().mean().item()
                if last.bias.grad is not None else float("nan")
            )

        # Backbone gradient too (does the signal even reach there?)
        backbone_first_w = data_ode.backbone.net[0].weight
        bb_grad = (
            backbone_first_w.grad.abs().mean().item()
            if backbone_first_w.grad is not None else float("nan")
        )

        print(f"\n[step {step}] loss={loss.item():.5f}")
        if step == 0:
            print("  PHYSICS INPUTS (what arrives at PhysicsLayer):")
            for k, v in physics_inputs.items():
                vn = v.float()
                finite = torch.isfinite(vn)
                if finite.any():
                    vf = vn[finite]
                    print(
                        f"    {k:<22s} mean={vf.mean().item():+.4e} "
                        f"std={vf.std().item():.4e} "
                        f"min={vf.min().item():+.4e} max={vf.max().item():+.4e} "
                        f"nan={(~finite).sum().item()}/{vn.numel()}"
                    )
                else:
                    print(f"    {k:<22s} ALL NaN/Inf ({vn.numel()} elements)")
        if a_spec is not None:
            print(
                f"  a_spec      mean={a_spec.mean().item():+.4e} "
                f"std={a_spec.std().item():.4e} |max|={a_spec.abs().max().item():.4e}"
            )
        if n_z_r is not None:
            print(
                f"  n_z_resid   mean={n_z_r.mean().item():+.4e} "
                f"std={n_z_r.std().item():.4e} |max|={n_z_r.abs().max().item():.4e}"
            )
        if d_tas is not None:
            print(
                f"  d_tas (out) mean={d_tas.mean().item():+.4e} "
                f"std={d_tas.std().item():.4e}"
            )
        if d_gamma is not None:
            print(
                f"  d_gamma(out) mean={d_gamma.mean().item():+.4e} "
                f"std={d_gamma.std().item():.4e}"
            )
        for k, v in grad.items():
            print(f"  grad[{k}]: {v:.4e}")
        print(f"  grad[backbone first W]: {bb_grad:.4e}")

        trainer.optimizer.step()

    print("\n" + "=" * 80)
    print("Head state after 5 steps:")
    for col, head in data_ode.heads.layer_dict.items():
        last = head.net[-1]
        print(
            f"  head[{col}]: |W|.max={last.weight.abs().max().item():.3e}, "
            f"bias={last.bias.item():.3e}"
        )


if __name__ == "__main__":
    main()
