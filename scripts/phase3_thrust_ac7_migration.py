"""Phase 3 — AC7 cybernetic-migration check.

Predicted by PHASE_1.5_CL_FORMULATION_LESSONS.md §11.5 :

> *The cybernetic leak will probably migrate towards `T_NN` once `delta C_D_NN`
>  is bounded.*

This script forward-passes BOTH v12 (Phase 2, unbounded `t_minus_d_norm`)
and v13 (Phase 3, bounded `±5 %` `t_correction`) over the same val slice
in cruise stable, and computes :

    AC7_ratio = mean(|0.05 · tanh(t_correction) · T_PS|)_v13_cruise
              / mean(|t_minus_d_norm · m_ref|)_v12_cruise

Verdict (PHASE_3_THRUST_PS_TICKET §6 Strategy Z) :
    - ratio in [0.5, 2.0] : no significant migration → ✅ AC7 PASS
    - ratio > 2.0         : leak migrated, capability ceiling reached
    - ratio < 0.5         : head even more dormant than v12 (good sign)

Output : prints verdict + writes
``data/models/_comparison/phase3_thrust_v13_ac7_migration.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from node_fdm.architectures import (  # noqa: F401
    adsb_hybrid_v12_psdrag,
    adsb_hybrid_v13_psthrust,
)
from node_fdm.architectures.registry import get as get_arch_spec
from node_fdm.trainer import ODETrainer, TrainingConfig

# Sample reference mass — same as PhysicsLayer's _M_REF_KG (midpoint of
# A320 TCDS plausible range). Used to convert v12's `t_minus_d_norm`
# back to its Newton-equivalent magnitude for the ratio.
_M_REF_KG = 0.5 * (42_600.0 + 77_000.0)


def _build_trainer(
    model_dir: Path,
    config_path: Path,
    arch_key: str,
    *,
    val_limit: int = 1000,
    lenient_load: bool = False,
) -> ODETrainer:
    """Re-construct the ODETrainer matching the saved meta + dataset.

    *lenient_load* — passes ``strict=False`` when restoring layer states.
    Needed when the layer's ``__init__`` now creates buffers (e.g. v12's
    PhysicsLayer didn't have ``ps_thrust.*`` at save time, but the
    refactored layer always instantiates ``PSThrustLayer`` for the
    Phase 3 branch).
    """
    from node_fdm.loader import get_train_val_data
    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    meta = json.loads((model_dir / "meta.json").read_text())
    arch_name = meta["architecture_name"]
    info = resolve_architecture(arch_key)
    spec = get_arch_spec(arch_name)
    cfg = PipelineConfig.from_yaml(config_path)

    training_config = TrainingConfig(
        architecture_name=arch_name,
        model_name=meta.get("model_name", model_dir.name),
        model_params=tuple(meta["model_params"]),
        seq_len=meta["seq_len"],
        shift=meta["shift"],
        step=meta["step"],
        lr=meta["lr"],
        batch_size=meta["batch_size"],
        method=meta["method"],
        activation=meta.get("activation", "silu"),
        seed=meta.get("seed", 42),
        use_mode_weights=meta.get("use_mode_weights", False),
        mode_weight_alpha=meta.get("mode_weight_alpha", 0.5),
        epochs=1,
    )

    delta_table = cfg.paths.resolve("delta_table")
    import polars as pl

    full_df = pl.read_delta(str(delta_table)).filter(pl.col("fdm_flag_valid"))
    data_df = full_df.filter(pl.col("meta_aircraft_type") == "A320")
    flight_feature_cols = list(getattr(spec, "flight_feature_cols", []) or [])
    _train_ds, val_ds = get_train_val_data(
        data_df=data_df,
        x_cols=info.x_cols,
        u_cols=info.u_cols,
        e_cols=info.e0_cols,
        e1_cols=info.e1_cols,
        dx_cols=[c for _, c in info.dx_cols],
        seq_len=training_config.seq_len,
        shift=training_config.shift,
        train_limit=val_limit,
        val_limit=val_limit,
        flight_feature_cols=flight_feature_cols or None,
        require_routing=True,
    )

    trainer = ODETrainer(
        config=training_config,
        train_dataset=val_ds,
        val_dataset=val_ds,
        model_dir=model_dir.parent,
        device="cpu",
    )
    if lenient_load:
        # Bypass the strict load — v12's saved layer state predates
        # PhysicsLayer.ps_thrust (new in Phase 3), so the buffer keys
        # for ``ps_thrust.*`` are missing from the ckpt. Loading
        # strict=False preserves all weights v12 *did* save and lets
        # the new buffers keep their __init__ values (which is what we
        # want — they're constants).
        import torch as _torch
        for name in trainer.model.layers_name:
            ckpt_path = trainer.model_dir / f"{name}.pt"
            ckpt = _torch.load(ckpt_path, weights_only=True)
            trainer.model.layers_dict[name].load_state_dict(
                ckpt["layer_state"], strict=False,
            )
        if trainer.mass_encoder is not None:
            mass_ckpt = trainer.model_dir / "mass_encoder.pt"
            if mass_ckpt.exists():
                state = _torch.load(mass_ckpt, weights_only=True)
                trainer.mass_encoder.load_state_dict(state, strict=False)
    else:
        trainer.load_model_weights(reset_loss=True)
    trainer.model.eval()
    return trainer


def _capture_head_outputs(
    trainer: ODETrainer,
    head_cols: tuple[str, ...],
    extra_cols: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Capture head outputs + extra state columns over the val loader."""
    accumulators: dict[str, list[np.ndarray]] = defaultdict(list)
    original_forward = trainer.model.layers_dict["data_ode_long"].forward

    captured: list[dict[str, torch.Tensor]] = []

    def patched_forward(vect_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = original_forward(vect_dict)
        # Capture *post-cap* head outputs (what feeds the PhysicsLayer).
        captured.append({k: out[k].detach().clone() for k in head_cols if k in out})
        return out

    trainer.model.layers_dict["data_ode_long"].forward = patched_forward  # type: ignore[method-assign]

    with torch.no_grad():
        for batch in trainer.val_loader:
            trainer.model.reset_history()
            _ = trainer._compute_batch_loss(batch)
            hist = trainer.model.history
            for k in extra_cols:
                if k in hist:
                    accumulators[k].append(hist[k].cpu().numpy().reshape(-1))

    trainer.model.layers_dict["data_ode_long"].forward = original_forward  # type: ignore[method-assign]

    # Concat captured head outputs.
    for k in head_cols:
        arrs = [c[k].cpu().numpy().reshape(-1) for c in captured if k in c]
        if arrs:
            accumulators[k] = [np.concatenate(arrs)]

    return {k: (v[0] if len(v) == 1 else np.concatenate(v)) for k, v in accumulators.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v12-dir", default="data/models/full_hybrid_v12", type=Path,
    )
    parser.add_argument(
        "--v13-dir", default="data/models/full_hybrid_v13", type=Path,
    )
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument(
        "--report",
        default="data/models/_comparison/phase3_thrust_v13_ac7_migration.md",
        type=Path,
    )
    args = parser.parse_args()

    print(f"Loading v12 from {args.v12_dir} ...")
    trainer_v12 = _build_trainer(
        args.v12_dir, args.config, "adsb_hybrid_v12_psdrag",
        val_limit=args.val_limit, lenient_load=True,
    )
    v12_data = _capture_head_outputs(
        trainer_v12,
        head_cols=("fdm_t_minus_d_norm",),
        extra_cols=("raw_alt_m", "era_tas_ms"),
    )
    print(f"  captured {len(v12_data.get('fdm_t_minus_d_norm', []))} head samples")

    print(f"Loading v13 from {args.v13_dir} ...")
    trainer_v13 = _build_trainer(
        args.v13_dir, args.config, "adsb_hybrid_v13_psthrust",
        val_limit=args.val_limit,
    )
    v13_data = _capture_head_outputs(
        trainer_v13,
        head_cols=("fdm_t_correction",),
        extra_cols=("raw_alt_m", "era_tas_ms", "fdm_T_PS"),
    )
    print(f"  captured {len(v13_data.get('fdm_t_correction', []))} head samples")

    # Cruise-stable mask : same approximation as forward-pass (alt+TAS based).
    def cruise_mask(alt: np.ndarray, tas: np.ndarray) -> np.ndarray:
        return (alt >= 9_000.0) & (tas >= 180.0) & (tas <= 270.0)

    # --- v12 magnitude : mean(|t_minus_d_norm * m_ref|) in cruise stable.
    v12_alt = v12_data["raw_alt_m"]
    v12_tas = v12_data["era_tas_ms"]
    v12_head = v12_data["fdm_t_minus_d_norm"]
    n12 = min(len(v12_alt), len(v12_head))
    m12 = cruise_mask(v12_alt[:n12], v12_tas[:n12])
    v12_magnitude_N = np.abs(v12_head[:n12][m12] * _M_REF_KG)
    v12_mean_N = float(v12_magnitude_N.mean()) if m12.sum() > 0 else float("nan")
    print(f"\nv12 cruise stable n = {m12.sum()}")
    print(f"v12 mean |t_minus_d_norm * m_ref| = {v12_mean_N:.1f} N")

    # --- v13 magnitude : mean(|0.05 * tanh(t_correction) * T_PS|) in cruise stable.
    v13_alt = v13_data["raw_alt_m"]
    v13_tas = v13_data["era_tas_ms"]
    v13_head = v13_data["fdm_t_correction"]
    v13_t_ps = v13_data["fdm_T_PS"]
    n13 = min(len(v13_alt), len(v13_head), len(v13_t_ps))
    m13 = cruise_mask(v13_alt[:n13], v13_tas[:n13])
    v13_correction_factor = 0.05 * np.tanh(v13_head[:n13])
    v13_magnitude_N = np.abs(v13_correction_factor[m13] * v13_t_ps[:n13][m13])
    v13_mean_N = float(v13_magnitude_N.mean()) if m13.sum() > 0 else float("nan")
    print(f"v13 cruise stable n = {m13.sum()}")
    print(f"v13 mean |0.05 tanh(t_corr) * T_PS| = {v13_mean_N:.1f} N")

    # --- Ratio + verdict.
    ratio = v13_mean_N / v12_mean_N if v12_mean_N > 0 else float("nan")
    print(f"\nAC7 ratio = {ratio:.3f}")

    if 0.5 <= ratio <= 2.0:
        verdict = "PASS"
        message = (
            "No significant cybernetic migration. The bounded ±5 % "
            "thrust correction operates at a magnitude comparable to "
            "v12's unbounded longitudinal head — the leak did not "
            "concentrate into the new head."
        )
    elif ratio > 2.0:
        verdict = "FAIL — leak migrated"
        message = (
            f"AC7 ratio {ratio:.2f} > 2.0. The leak has migrated to "
            "t_correction : the bounded head is doing more "
            "compensation work than v12's unbounded head did. "
            "Capability ceiling reached — document and recommend "
            "Phase 1.8a before Phase 4."
        )
    else:
        verdict = "PASS — head even quieter"
        message = (
            f"AC7 ratio {ratio:.2f} < 0.5. The bounded head is more "
            "dormant than v12's unbounded head — the analytical "
            "T_PS prior carries the bulk of the signal."
        )

    print(f"AC7 verdict : {verdict}")
    print(f"  {message}")

    lines = [
        "# Phase 3 — AC7 cybernetic-migration check",
        "",
        "> Predicted by `PHASE_1.5_CL_FORMULATION_LESSONS.md §11.5`. The "
        "central scientific test of Phase 3.",
        "",
        "## Method",
        "",
        "Forward-pass both v12 (Phase 2, unbounded `t_minus_d_norm`) and v13 "
        "(Phase 3, bounded ±5 % `t_correction`) over the same val slice. "
        "Capture the longitudinal head's post-cap output, restrict to a "
        "cruise-stable mask (alt ≥ 9000 m, TAS ∈ [180, 270] m/s), and "
        "compute the magnitude of each head's Newton-equivalent "
        "contribution :",
        "",
        "- **v12** : `|t_minus_d_norm · m_ref|` (unbounded — the whole "
        "longitudinal force the NN predicts).",
        "- **v13** : `|0.05 · tanh(t_correction) · T_PS|` (bounded — the "
        "absolute size of the correction the NN applies on top of the "
        "analytical T_PS prior).",
        "",
        f"`m_ref = {_M_REF_KG:.0f} kg` (PhysicsLayer `_M_REF_KG`, midpoint "
        "of A320 TCDS plausible range).",
        "",
        "## Numbers",
        "",
        f"- v12 cruise samples : **{int(m12.sum())}**",
        f"- v12 mean |t_minus_d_norm · m_ref| : **{v12_mean_N:.1f} N**",
        f"- v13 cruise samples : **{int(m13.sum())}**",
        f"- v13 mean |0.05·tanh(t_corr)·T_PS| : **{v13_mean_N:.1f} N**",
        "",
        f"- **AC7 ratio = {ratio:.3f}**",
        "- target : ratio ∈ [0.5, 2.0]",
        f"- status : **{verdict}**",
        "",
        message,
        "",
        "## Interpretation",
        "",
        "Per `PHASE_1.5_CL_FORMULATION_LESSONS.md §11.5`, the bound on "
        "`delta C_D_NN` (Phase 2) was expected to push the closed-loop "
        "leakage onto `T_NN`. v12 left `t_minus_d_norm` unbounded, so "
        "any migration would manifest there. v13 bounds *both* drag and "
        "thrust corrections to ±5 % of their analytical priors — if the "
        "leak still concentrates, AC7 would catch it as a large ratio.",
    ]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(f"\nWrote {args.report}")

    return 0 if "PASS" in verdict else 1


if __name__ == "__main__":
    sys.exit(main())
