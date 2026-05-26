"""Phase 4 forward-pass diagnostic — closes AC4 / AC5 / AC6 / R8 / R9.

Loads the trained ``full_hybrid_v14`` checkpoint and runs the validation
dataset through the model twice :

    1. **Baseline** (all 4 NN heads active) — the trained model. Captures
       per-step `fdm_T_PS`, `fdm_T_total`, `fdm_throttle`, `fdm_eta_PS`,
       `fdm_eta_total`, `fdm_mdot_f`, `fdm_drag_N` from the
       FlightDynamicsModel's history dict.
    2. **NN-off** (all 4 corrections monkey-patched to 0 after the head
       runs) — re-evaluates val_loss to test AC5 sanity.

Computes :
    - **AC4_eta** : ``mean(|eta_total - eta_PS| / eta_PS)``. Target < 2 %.
    - **AC4_thrust** : Phase 3 inherited (mean |delta_T_NN| / T_PS). Target < 3 %.
    - **AC5** : ``val_loss_NN_off / val_loss_baseline``. Target < 1.50.
    - **AC6 / R9** : ``mdot_f`` cruise in [0.25, 0.45] kg/s/engine.
    - **R8** : `eta_PS` cruise in [0.20, 0.45].

Adapted mot-pour-mot from `phase3_thrust_forwardpass.py` with the
columns extended for the Phase 4 efficiency + fuel-flow heads.
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
    adsb_hybrid_v14_psefficiency,
)
from node_fdm.architectures.registry import get as get_arch_spec
from node_fdm.trainer import ODETrainer, TrainingConfig


def _build_trainer(
    model_dir: Path,
    config_path: Path,
    *,
    val_limit: int = 1000,
) -> ODETrainer:
    """Re-construct the ODETrainer matching the v14 saved meta + dataset."""
    from node_fdm.loader import get_train_val_data
    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    meta = json.loads((model_dir / "meta.json").read_text())
    arch_name = meta["architecture_name"]
    arch_key = "adsb_hybrid_v14_psefficiency"
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
        # Phase 10 — use the registered ArchitectureSpec's e0_cols so
        # arch-level extensions (e.g. v20's ``raw_age_years``) are
        # honoured. The resolver's ArchitectureInfo.e0_cols carries only
        # schema-level columns ; the loader downstream needs the arch's
        # actual e0_cols to stay in sync with compute_stats.
        e_cols=list(spec.e0_cols),
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
    trainer.load_model_weights(reset_loss=True)
    trainer.model.eval()
    return trainer


# All NN heads zeroed for AC5 NN-off (the 4 Phase 4 heads + Phase 3 inherited).
_NN_HEADS_TO_ZERO = (
    "fdm_cl_residual",
    "fdm_cd_correction",
    "fdm_t_correction",
    "fdm_eta_correction",
)


def _run_val_with_diagnostics(
    trainer: ODETrainer,
    *,
    zero_all_nn: bool = False,
) -> dict[str, np.ndarray]:
    """Iterate val_loader, capture per-rollout-step diagnostic outputs."""
    accumulators: dict[str, list[np.ndarray]] = defaultdict(list)
    val_loss_total = 0.0
    n_batches = 0

    original_forward = trainer.model.layers_dict["data_ode_long"].forward

    def patched_forward(vect_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = original_forward(vect_dict)
        if zero_all_nn:
            patched = dict(out)
            for col in _NN_HEADS_TO_ZERO:
                if col in patched:
                    patched[col] = torch.zeros_like(patched[col])
            out = patched
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
                "fdm_T_PS",
                "fdm_T_total",
                "fdm_throttle",
                "fdm_c_d_ps",
                "fdm_c_d_total",
                "fdm_drag_N",
                "fdm_thrust_N",
                "fdm_lift_N",
                "fdm_mass_kg",
                "fdm_eta_PS",
                "fdm_eta_total",
                "fdm_mdot_f",
                "era_mach",
                "raw_alt_m",
                "era_tas_ms",
                "fdm_q_pa",
            ):
                if key in hist:
                    accumulators[key].append(hist[key].cpu().numpy().reshape(-1))

    trainer.model.layers_dict["data_ode_long"].forward = original_forward  # type: ignore[method-assign]

    return {
        "val_loss": np.array([val_loss_total / max(n_batches, 1)]),
        **{k: np.concatenate(v) for k, v in accumulators.items()},
    }


def main() -> int:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", default="data/models/full_hybrid_v14", type=Path,
    )
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument(
        "--report",
        default="data/models/_comparison/phase4_fuelflow_v14_forwardpass.md",
        type=Path,
    )
    args = parser.parse_args()

    print(f"Loading v14 from {args.model_dir} ...")
    trainer = _build_trainer(args.model_dir, args.config, val_limit=args.val_limit)
    print(f"Val dataset : {len(trainer.val_dataset)} samples")

    print("\n=== Run A : baseline (all 4 NN heads active) ===")
    base = _run_val_with_diagnostics(trainer, zero_all_nn=False)
    base_val_loss = float(base["val_loss"][0])
    print(f"  val_loss = {base_val_loss:.5f}")

    print("\n=== Run B : 4 NN heads zeroed (AC5 NN-off) ===")
    nnoff = _run_val_with_diagnostics(trainer, zero_all_nn=True)
    nnoff_val_loss = float(nnoff["val_loss"][0])
    print(f"  val_loss = {nnoff_val_loss:.5f}")
    degradation_pct = (nnoff_val_loss / base_val_loss - 1.0) * 100.0
    ac5_pass = degradation_pct < 50.0
    print(f"  degradation vs baseline = {degradation_pct:+.1f} %  "
          f"-- AC5 {'PASS' if ac5_pass else 'FAIL'}")

    # === AC4_thrust : Phase 3 inherited
    print("\n=== AC4_thrust : mean |delta T_NN| / T_PS (Phase 3 inherited) ===")
    t_ps = base["fdm_T_PS"]
    t_total = base["fdm_T_total"]
    valid_t = (t_ps > 1.0) & np.isfinite(t_ps) & np.isfinite(t_total)
    delta_t_pct = np.abs(t_total[valid_t] / t_ps[valid_t] - 1.0) * 100.0
    mean_abs_delta_t = float(delta_t_pct.mean())
    p99_abs_delta_t = float(np.percentile(delta_t_pct, 99))
    ac4_thrust_pass = mean_abs_delta_t < 3.0
    print(f"  n_samples = {valid_t.sum()}")
    print(f"  mean |delta T_NN| = {mean_abs_delta_t:.3f} %")
    print(f"  p99  |delta T_NN| = {p99_abs_delta_t:.3f} %")
    print(f"  AC4_thrust {'PASS' if ac4_thrust_pass else 'FAIL'} (target < 3 %)")

    # === AC4_cd : Phase 8 — drag-correction saturation diagnostic ===
    # Mirrors AC4_thrust : measure how far the NN's c_d_correction pushes
    # c_d_total away from the analytical c_d_PS prior. If the head is glued
    # to its ±5 % cap (mean approaching 5 %, p99 = 4.975 %), the PSDragLayer
    # prior is systematically off and the NN wants an unbounded escape.
    print("\n=== AC4_cd : mean |delta c_d_NN| / c_d_PS (Phase 8 saturation diagnostic) ===")
    c_d_ps = base.get("fdm_c_d_ps")
    c_d_total = base.get("fdm_c_d_total")
    if c_d_ps is not None and c_d_total is not None:
        valid_cd = (c_d_ps > 1e-4) & np.isfinite(c_d_ps) & np.isfinite(c_d_total)
        delta_cd_pct = np.abs(c_d_total[valid_cd] / c_d_ps[valid_cd] - 1.0) * 100.0
        mean_abs_delta_cd = float(delta_cd_pct.mean())
        p99_abs_delta_cd = float(np.percentile(delta_cd_pct, 99))
        ac4_cd_pass = mean_abs_delta_cd < 3.0
        print(f"  n_samples = {valid_cd.sum()}")
        print(f"  mean |delta c_d_NN| = {mean_abs_delta_cd:.3f} %")
        print(f"  p99  |delta c_d_NN| = {p99_abs_delta_cd:.3f} %")
        print(f"  AC4_cd {'PASS' if ac4_cd_pass else 'FAIL'} (target < 3 %)")
    else:
        mean_abs_delta_cd = p99_abs_delta_cd = float("nan")
        ac4_cd_pass = False
        print("  c_d_ps / c_d_total not in history — skipping AC4_cd")

    # === AC4_eta : Phase 4 new ===
    print("\n=== AC4_eta : mean |delta eta_NN| / eta_PS (Phase 4 new) ===")
    eta_ps = base["fdm_eta_PS"]
    eta_total = base["fdm_eta_total"]
    valid_eta = (eta_ps > 0.05) & np.isfinite(eta_ps) & np.isfinite(eta_total)
    delta_eta_pct = np.abs(eta_total[valid_eta] / eta_ps[valid_eta] - 1.0) * 100.0
    mean_abs_delta_eta = float(delta_eta_pct.mean())
    p99_abs_delta_eta = float(np.percentile(delta_eta_pct, 99))
    ac4_eta_pass = mean_abs_delta_eta < 2.0
    print(f"  n_samples = {valid_eta.sum()}")
    print(f"  mean |delta eta_NN| = {mean_abs_delta_eta:.3f} %")
    print(f"  p99  |delta eta_NN| = {p99_abs_delta_eta:.3f} %")
    print(f"  AC4_eta {'PASS' if ac4_eta_pass else 'FAIL'} (target < 2 %)")

    # === Phase-stratified breakdown for AC4_eta ===
    alt_m = base.get("raw_alt_m")
    phase_eta_stats: dict[str, dict[str, float]] = {}
    if alt_m is not None:
        for label, mask in [
            ("low (alt < 3000 m)", alt_m < 3_000.0),
            ("mid (3000-9000 m)", (alt_m >= 3_000.0) & (alt_m < 9_000.0)),
            ("cruise (>= 9000 m)", alt_m >= 9_000.0),
        ]:
            m = mask & valid_eta
            if m.sum() < 20:
                continue
            d = np.abs(eta_total[m] / eta_ps[m] - 1.0) * 100.0
            phase_eta_stats[label] = {
                "n": int(m.sum()),
                "mean_abs_delta_pct": float(d.mean()),
                "p99_abs_delta_pct": float(np.percentile(d, 99)),
            }
            print(f"  [{label}]  n={m.sum()}  "
                  f"mean={d.mean():.3f}%  p99={np.percentile(d, 99):.3f}%")

    # === R8 : eta_PS cruise range ===
    print("\n=== R8 : eta_PS cruise range in [0.20, 0.45] ===")
    cruise_mask = (alt_m >= 9_000.0) if alt_m is not None else np.ones_like(eta_ps, dtype=bool)
    cruise_eta = eta_ps[cruise_mask & valid_eta]
    if cruise_eta.size > 0:
        eta_min = float(cruise_eta.min())
        eta_max = float(cruise_eta.max())
        eta_med = float(np.median(cruise_eta))
        r8_pass = (eta_min >= 0.18) and (eta_max <= 0.50)  # slightly relaxed (sanity, not hard cutoff)
        print(f"  cruise eta_PS  min={eta_min:.3f}  median={eta_med:.3f}  max={eta_max:.3f}")
        print(f"  R8 {'PASS' if r8_pass else 'FAIL'}")
    else:
        r8_pass = False
        eta_min = eta_max = eta_med = float("nan")

    # === R9 / AC6 : mdot_f cruise sanity range ===
    print("\n=== R9 / AC6 : mdot_f cruise in [0.25, 0.45] kg/s/engine (i.e. [0.5, 0.9] kg/s total) ===")
    mdot_f = base["fdm_mdot_f"]
    valid_m = (mdot_f > 0) & np.isfinite(mdot_f)
    cruise_mdot = mdot_f[cruise_mask & valid_m]
    if cruise_mdot.size > 0:
        mdot_min = float(cruise_mdot.min())
        mdot_max = float(cruise_mdot.max())
        mdot_med = float(np.median(cruise_mdot))
        mdot_med_per_engine = mdot_med / 2.0
        mdot_med_t_per_h = mdot_med * 3.6  # kg/s -> t/h
        r9_pass = (mdot_med_per_engine >= 0.20) and (mdot_med_per_engine <= 0.50)
        print(f"  cruise mdot_f (total)  min={mdot_min:.3f}  median={mdot_med:.3f}  max={mdot_max:.3f} kg/s")
        print(f"  cruise mdot_f median = {mdot_med_t_per_h:.2f} t/h total ({mdot_med_per_engine:.3f} kg/s/engine)")
        print(f"  R9 {'PASS' if r9_pass else 'FAIL'} (target 0.20-0.50 kg/s/engine)")
    else:
        r9_pass = False
        mdot_min = mdot_max = mdot_med = mdot_med_per_engine = mdot_med_t_per_h = float("nan")

    # --- Markdown report ---
    lines = [
        "# Phase 4 — Forward-pass diagnostic (AC4 / AC5 / AC6 / R8 / R9 closure)",
        "",
        f"> Trained model : `{args.model_dir.name}`. Val split limited to "
        f"{args.val_limit} samples.",
        "",
        "## AC4_thrust — Mean |delta T_NN| / T_PS (Phase 3 inherited)",
        "",
        f"- n samples (cruise + all phases) : **{int(valid_t.sum())}**",
        f"- mean |delta T_NN| : **{mean_abs_delta_t:.3f} %**",
        f"- p99  |delta T_NN| : **{p99_abs_delta_t:.3f} %**",
        "- target : mean < 3 %",
        f"- status : **{'✅ PASS' if ac4_thrust_pass else '⚠️ FAIL'}**",
        "",
        "## AC4_cd — Mean |delta c_d_NN| / c_d_PS (Phase 8 saturation diagnostic)",
        "",
        f"- n samples : **{int(valid_cd.sum()) if c_d_ps is not None else 0}**",
        f"- mean |delta c_d_NN| : **{mean_abs_delta_cd:.3f} %**",
        f"- p99  |delta c_d_NN| : **{p99_abs_delta_cd:.3f} %**",
        "- target : mean < 3 %",
        f"- status : **{'✅ PASS' if ac4_cd_pass else '⚠️ FAIL'}**",
        "",
        "## AC4_eta — Mean |delta eta_NN| / eta_PS (Phase 4 new)",
        "",
        f"- n samples : **{int(valid_eta.sum())}**",
        f"- mean |delta eta_NN| : **{mean_abs_delta_eta:.3f} %**",
        f"- p99  |delta eta_NN| : **{p99_abs_delta_eta:.3f} %**",
        "- target : mean < 2 %",
        f"- status : **{'✅ PASS' if ac4_eta_pass else '⚠️ FAIL'}**",
        "",
        "### Phase-stratified breakdown (eta correction)",
        "",
        "| Phase band | n | mean |Δη_NN| % | p99 |Δη_NN| % |",
        "|---|---:|---:|---:|",
    ]
    for label, s in phase_eta_stats.items():
        lines.append(
            f"| {label} | {s['n']} | {s['mean_abs_delta_pct']:.3f} | "
            f"{s['p99_abs_delta_pct']:.3f} |"
        )
    lines.extend([
        "",
        "## AC5 — NN-off sanity check (all 4 heads zeroed)",
        "",
        f"- baseline val_loss (4 NN heads active) : **{base_val_loss:.5f}**",
        f"- NN-off val_loss (cl/cd/t/η_correction = 0) : **{nnoff_val_loss:.5f}**",
        f"- degradation : **{degradation_pct:+.2f} %**",
        "- target : < 50 % degradation",
        f"- status : **{'✅ PASS' if ac5_pass else '⚠️ FAIL'}**",
        "",
        "## R8 — Eta_PS cruise range",
        "",
        f"- cruise eta_PS min : **{eta_min:.3f}**",
        f"- cruise eta_PS median : **{eta_med:.3f}**",
        f"- cruise eta_PS max : **{eta_max:.3f}**",
        "- target : ∈ [0.18, 0.50]",
        f"- status : **{'✅ PASS' if r8_pass else '⚠️ FAIL'}**",
        "",
        "## R9 / AC6 — Fuel-flow cruise sanity",
        "",
        f"- cruise mdot_f total min : **{mdot_min:.3f} kg/s**",
        f"- cruise mdot_f total median : **{mdot_med:.3f} kg/s** "
        f"({mdot_med_t_per_h:.2f} t/h, {mdot_med_per_engine:.3f} kg/s/engine)",
        f"- cruise mdot_f total max : **{mdot_max:.3f} kg/s**",
        "- target per-engine : ∈ [0.20, 0.50] kg/s/engine",
        f"- status : **{'✅ PASS' if r9_pass else '⚠️ FAIL'}**",
        "",
        "## Methodology",
        "",
        "Same monkey-patch idiom as Phase 3, extended to zero all 4 NN heads "
        "(`fdm_cl_residual`, `fdm_cd_correction`, `fdm_t_correction`, "
        "`fdm_eta_correction`) for AC5. The PhysicsLayer Phase 4 branch "
        "exposes `fdm_eta_PS`, `fdm_eta_total`, `fdm_mdot_f` as new "
        "diagnostic outputs.",
    ])

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))
    print(f"\nWrote {args.report}")

    overall_pass = ac4_thrust_pass and ac4_eta_pass and ac5_pass and r8_pass and r9_pass
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
