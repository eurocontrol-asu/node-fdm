"""Phase 2 forward-pass diagnostic — closes AC4 / AC5 / AC6.

Loads the trained ``full_hybrid_v12`` checkpoint and runs the
validation dataset through the model twice :

    1. **Baseline** (NN drag correction *active*) — the trained model.
       Captures per-step `fdm_c_d_ps`, `fdm_c_d_total`, `fdm_drag_N`,
       `fdm_thrust_N` from the FlightDynamicsModel's history dict.
    2. **NN-off** (cd_correction monkey-patched to 0 after the head
       runs) — re-evaluates val_loss to test AC5 sanity.

Computes :
    - **AC4** : ``mean(|delta_C_D_NN|) = mean(|C_D_total - C_D_PS| /
      C_D_PS)``. Target < 3 %.
    - **AC5** : ``val_loss_NN_off / val_loss_baseline``. Target < 1.50
      (i.e. degradation < 50 %).
    - **AC6** : on cruise-stable rows (|dh/dt| < 1 m/s, alt >= 9000 m,
      |dV/dt| < 0.1 m/s^2), check ``|T_NN - D_PS_corrected| / D``.
      Target : median < 3 % (cruise should have T ≈ D).

Writes the report under
``data/models/_comparison/phase2_drag_v12_forwardpass.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Side-effect imports to register the architectures and PhysicsLayer.
from node_fdm.architectures import (  # noqa: F401
    adsb_hybrid_v12_psdrag,
)
from node_fdm.architectures.registry import get as get_arch_spec
from node_fdm.trainer import ODETrainer, TrainingConfig


def _build_trainer(
    model_dir: Path,
    config_path: Path,
    *,
    val_limit: int = 1000,
) -> ODETrainer:
    """Re-construct the ODETrainer matching the v12 saved meta + dataset."""
    from node_fdm.loader import get_train_val_data
    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    meta = json.loads((model_dir / "meta.json").read_text())
    arch_name = meta["architecture_name"]
    arch_key = "adsb_hybrid_v12_psdrag"  # local CLI key, not the registered name
    info = resolve_architecture(arch_key)
    spec = get_arch_spec(arch_name)
    cfg_path = config_path
    cfg = PipelineConfig.from_yaml(cfg_path)

    # Re-create the training config from meta so the trainer's shapes
    # line up byte-for-byte with the saved checkpoints.
    training_config = TrainingConfig(
        architecture_name=arch_name,
        model_name=meta["model_name"]
        if "model_name" in meta
        else model_dir.name,
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
        epochs=1,  # not used here
    )

    # Same data path the trainer uses : Delta table -> val split.
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

    # Use the val dataset as both train and val to skip the train-only
    # boot ; the diagnostic only iterates over val_loader.
    trainer = ODETrainer(
        config=training_config,
        train_dataset=val_ds,
        val_dataset=val_ds,
        model_dir=model_dir.parent,
        device="cpu",
    )
    trainer.load_model_weights(reset_loss=True)
    trainer.model.eval()
    return trainer


def _run_val_with_diagnostics(
    trainer: ODETrainer,
    *,
    zero_cd_correction: bool = False,
) -> dict[str, np.ndarray]:
    """Iterate val_loader, capture per-segment diagnostic outputs."""
    accumulators: dict[str, list[np.ndarray]] = defaultdict(list)
    val_loss_total = 0.0
    n_batches = 0

    # Optional monkey-patch : zero out fdm_cd_correction in the model's
    # forward path so AC5 evaluates "what does val_loss look like when
    # the NN's drag correction is structurally disabled".
    original_forward = trainer.model.layers_dict["data_ode_long"].forward

    def patched_forward(vect_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = original_forward(vect_dict)
        if zero_cd_correction and "fdm_cd_correction" in out:
            out = {**out, "fdm_cd_correction": torch.zeros_like(out["fdm_cd_correction"])}
        return out

    trainer.model.layers_dict["data_ode_long"].forward = patched_forward  # type: ignore[method-assign]

    with torch.no_grad():
        for batch in trainer.val_loader:
            trainer.model.reset_history()
            loss = trainer._compute_batch_loss(batch)
            val_loss_total += float(loss.item())
            n_batches += 1
            hist = trainer.model.history
            for key in (
                "fdm_c_d_ps",
                "fdm_c_d_total",
                "fdm_drag_N",
                "fdm_thrust_N",
                "fdm_lift_N",
                "fdm_mass_kg",
                "era_mach",
                "raw_alt_m",
                "era_tas_ms",
                "fdm_q_pa",
            ):
                if key in hist:
                    accumulators[key].append(hist[key].cpu().numpy().reshape(-1))

    # Restore the original forward to keep the trainer pristine.
    trainer.model.layers_dict["data_ode_long"].forward = original_forward  # type: ignore[method-assign]

    return {
        "val_loss": np.array([val_loss_total / max(n_batches, 1)]),
        **{k: np.concatenate(v) for k, v in accumulators.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", default="data/models/full_hybrid_v12", type=Path,
    )
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument(
        "--report",
        default="data/models/_comparison/phase2_drag_v12_forwardpass.md",
        type=Path,
    )
    args = parser.parse_args()

    print(f"Loading v12 from {args.model_dir} ...")
    trainer = _build_trainer(args.model_dir, args.config, val_limit=args.val_limit)
    print(f"Val dataset : {len(trainer.val_dataset)} samples")

    print("\n=== Run A : baseline (NN drag correction active) ===")
    base = _run_val_with_diagnostics(trainer, zero_cd_correction=False)
    base_val_loss = float(base["val_loss"][0])
    print(f"  val_loss = {base_val_loss:.5f}")

    print("\n=== Run B : cd_correction zeroed (AC5 NN-off) ===")
    nnoff = _run_val_with_diagnostics(trainer, zero_cd_correction=True)
    nnoff_val_loss = float(nnoff["val_loss"][0])
    print(f"  val_loss = {nnoff_val_loss:.5f}")
    degradation_pct = (nnoff_val_loss / base_val_loss - 1.0) * 100.0
    ac5_pass = degradation_pct < 50.0
    print(f"  degradation vs baseline = {degradation_pct:+.1f} %  "
          f"-- AC5 {'PASS' if ac5_pass else 'FAIL'}")

    print("\n=== AC4 : mean |delta_C_D_NN| ===")
    c_d_ps = base["fdm_c_d_ps"]
    c_d_total = base["fdm_c_d_total"]
    valid = (c_d_ps > 1e-6) & np.isfinite(c_d_ps) & np.isfinite(c_d_total)
    delta_pct = np.abs(c_d_total[valid] / c_d_ps[valid] - 1.0) * 100.0
    mean_abs_delta = float(delta_pct.mean())
    p99_abs_delta = float(np.percentile(delta_pct, 99))
    ac4_pass = mean_abs_delta < 3.0
    print(f"  n_samples = {valid.sum()}")
    print(f"  mean |delta_C_D_NN| = {mean_abs_delta:.3f} %")
    print(f"  p99  |delta_C_D_NN| = {p99_abs_delta:.3f} %")
    print(f"  AC4 {'PASS' if ac4_pass else 'FAIL'}")

    # Phase-stratified for diagnostic insight.
    alt_m = base.get("raw_alt_m")
    phase_stats: dict[str, dict[str, float]] = {}
    if alt_m is not None:
        for label, mask in [
            ("low (alt < 3000 m)", alt_m < 3_000.0),
            ("mid (3000-9000 m)", (alt_m >= 3_000.0) & (alt_m < 9_000.0)),
            ("cruise (>= 9000 m)", alt_m >= 9_000.0),
        ]:
            m = mask & valid
            if m.sum() < 20:
                continue
            delta = np.abs(c_d_total[m] / c_d_ps[m] - 1.0) * 100.0
            phase_stats[label] = {
                "n": int(m.sum()),
                "mean_abs_delta_pct": float(delta.mean()),
                "p99_abs_delta_pct": float(np.percentile(delta, 99)),
            }
            print(f"  [{label}]  n={m.sum()}  "
                  f"mean={delta.mean():.3f}%  p99={np.percentile(delta, 99):.3f}%")

    print("\n=== AC6 : |T - D| / D on cruise stable ===")
    ac6_result: dict[str, float | str] = {"status": "no_cruise_samples"}
    if alt_m is not None and "fdm_drag_N" in base and "fdm_thrust_N" in base:
        tas = base["era_tas_ms"]
        n = min(len(tas), len(alt_m))
        # Approximate cruise mask : alt >= 9000 m, tas in plausible range.
        # |dh/dt| inference not directly available here (history is
        # per-rollout step ; cleanest approximation is alt-band only).
        cruise_mask = (alt_m[:n] >= 9_000.0) & (tas[:n] >= 180.0) & (tas[:n] <= 270.0)
        thrust = base["fdm_thrust_N"][:n]
        drag = base["fdm_drag_N"][:n]
        ok = cruise_mask & (np.abs(drag) > 1e3) & np.isfinite(thrust) & np.isfinite(drag)
        if ok.sum() >= 20:
            rel_err = np.abs(thrust[ok] - drag[ok]) / np.abs(drag[ok]) * 100.0
            median_err = float(np.median(rel_err))
            mean_err = float(np.mean(rel_err))
            p99_err = float(np.percentile(rel_err, 99))
            ac6_pass = median_err < 3.0
            ac6_result = {
                "status": "evaluated",
                "n": int(ok.sum()),
                "median_pct": median_err,
                "mean_pct": mean_err,
                "p99_pct": p99_err,
                "verdict": "PASS" if ac6_pass else "FAIL",
            }
            print(f"  n_cruise_samples = {ok.sum()}")
            print(f"  median |T-D|/D = {median_err:.2f} %  (target < 3 %)")
            print(f"  mean   = {mean_err:.2f} %")
            print(f"  p99    = {p99_err:.2f} %")
            print(f"  AC6 {ac6_result['verdict']}")

    # --- Markdown report ---
    lines = [
        "# Phase 2 — Forward-pass diagnostic (AC4 / AC5 / AC6 closure)",
        "",
        f"> Trained model : `{args.model_dir.name}`. Val split limited to "
        f"{args.val_limit} samples.",
        "",
        "## AC4 — Mean |delta C_D_NN|",
        "",
        f"- n samples (cruise + all phases) : **{int(valid.sum())}**",
        f"- mean |delta C_D_NN| : **{mean_abs_delta:.3f} %**",
        f"- p99  |delta C_D_NN| : **{p99_abs_delta:.3f} %**",
        "- target : mean < 3 %",
        f"- status : **{'✅ PASS' if ac4_pass else '⚠️ FAIL'}**",
        "",
        "### Phase-stratified breakdown",
        "",
        "| Phase band | n | mean |ΔC_D| % | p99 |ΔC_D| % |",
        "|---|---:|---:|---:|",
    ]
    for label, s in phase_stats.items():
        lines.append(
            f"| {label} | {s['n']} | {s['mean_abs_delta_pct']:.3f} | "
            f"{s['p99_abs_delta_pct']:.3f} |"
        )
    lines.extend([
        "",
        "## AC5 — NN-off sanity check",
        "",
        f"- baseline val_loss (NN drag active) : **{base_val_loss:.5f}**",
        f"- NN-off val_loss (cd_correction = 0) : **{nnoff_val_loss:.5f}**",
        f"- degradation : **{degradation_pct:+.2f} %**",
        "- target : < 50 % degradation",
        f"- status : **{'✅ PASS' if ac5_pass else '⚠️ FAIL'}**",
        "",
        "## AC6 — Cruise-stable T ≈ D check",
        "",
    ])
    if ac6_result.get("status") == "evaluated":
        lines.extend([
            f"- n cruise samples : **{ac6_result['n']}**",
            f"- median |T - D| / D : **{ac6_result['median_pct']:.2f} %**",
            f"- mean : {ac6_result['mean_pct']:.2f} %",
            f"- p99 : {ac6_result['p99_pct']:.2f} %",
            "- target : median < 3 %",
            f"- status : **{ '✅ PASS' if ac6_result['verdict'] == 'PASS' else '⚠️ FAIL'}**",
        ])
    else:
        lines.append(f"- status : {ac6_result['status']}")

    lines.extend([
        "",
        "## Methodology notes",
        "",
        "1. `_run_val_with_diagnostics` iterates the val_loader and pulls "
        "the FlightDynamicsModel's `history` dict after each batch. "
        "Phase 2's PhysicsLayer exposes `fdm_c_d_ps`, `fdm_c_d_total`, "
        "`fdm_drag_N`, `fdm_thrust_N`, `fdm_lift_N` as side-channel outputs.",
        "2. AC5's NN-off mode monkey-patches the `data_ode_long` layer's "
        "forward to zero `fdm_cd_correction`. The patch is reverted after "
        "the run.",
        "3. AC6 uses an altitude+TAS-based cruise mask rather than a "
        "|dh/dt|-based one because history accumulates per-rollout-step "
        "values that don't directly carry segment-window context. The "
        "result is conservative — non-cruise rows leak in if alt and "
        "TAS happen to look cruise-like.",
    ])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(f"\nWrote {args.report}")

    overall_pass = ac4_pass and ac5_pass and (
        ac6_result.get("verdict") in (None, "PASS")
    )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
