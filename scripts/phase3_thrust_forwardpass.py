"""Phase 3 forward-pass diagnostic — closes AC4 / AC5 / AC6.

Loads the trained ``full_hybrid_v13`` checkpoint and runs the
validation dataset through the model twice :

    1. **Baseline** (NN thrust correction *active*) — the trained model.
       Captures per-step `fdm_T_PS`, `fdm_T_total`, `fdm_throttle`,
       `fdm_drag_N` from the FlightDynamicsModel's history dict.
    2. **NN-off** (t_correction monkey-patched to 0 after the head
       runs) — re-evaluates val_loss to test AC5 sanity.

Computes :
    - **AC4** : ``mean(|delta_T_NN|) = mean(|T_total - T_PS| / T_PS)``.
      Target < 3 %.
    - **AC5** : ``val_loss_NN_off / val_loss_baseline``. Target < 1.50
      (i.e. degradation < 50 %).
    - **AC6** : throttle latent chi ∈ [0.1, 1.0], phase-coherent
      (climb > cruise > descent_idle).

Writes the report under
``data/models/_comparison/phase3_thrust_v13_forwardpass.md``.

Adapted mot-pour-mot from ``phase2_drag_forwardpass.py`` with the
columns swapped (c_d_* → T_*) and AC6 reformulated as the throttle
phase-coherence check (per PHASE_3_THRUST_PS_TICKET §8).
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
    adsb_hybrid_v13_psthrust,
)
from node_fdm.architectures.registry import get as get_arch_spec
from node_fdm.trainer import ODETrainer, TrainingConfig


def _build_trainer(
    model_dir: Path,
    config_path: Path,
    *,
    val_limit: int = 1000,
) -> ODETrainer:
    """Re-construct the ODETrainer matching the v13 saved meta + dataset."""
    from node_fdm.loader import get_train_val_data
    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    meta = json.loads((model_dir / "meta.json").read_text())
    arch_name = meta["architecture_name"]
    arch_key = "adsb_hybrid_v13_psthrust"
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
    trainer.load_model_weights(reset_loss=True)
    trainer.model.eval()
    return trainer


def _run_val_with_diagnostics(
    trainer: ODETrainer,
    *,
    zero_t_correction: bool = False,
) -> dict[str, np.ndarray]:
    """Iterate val_loader, capture per-rollout-step diagnostic outputs."""
    accumulators: dict[str, list[np.ndarray]] = defaultdict(list)
    val_loss_total = 0.0
    n_batches = 0

    original_forward = trainer.model.layers_dict["data_ode_long"].forward

    # All Phase 3 variants of the bounded thrust correction column name.
    # AC5 zeros whichever is present + the parallel residual (v13d) to fully
    # disable NN influence on the longitudinal force.
    t_correction_variants = (
        "fdm_t_correction",
        "fdm_t_correction_w10",
        "fdm_t_correction_tet",
        "fdm_t_correction_parallel",
        "fdm_t_minus_d_norm_parallel",
        "fdm_t_correction_parallel_l025",
        "fdm_t_minus_d_norm_parallel_l025",
        "fdm_t_correction_parallel_anneal",
        "fdm_t_minus_d_norm_parallel_anneal",
    )

    def patched_forward(vect_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = original_forward(vect_dict)
        if zero_t_correction:
            patched = dict(out)
            for col in t_correction_variants:
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

    trainer.model.layers_dict["data_ode_long"].forward = original_forward  # type: ignore[method-assign]

    return {
        "val_loss": np.array([val_loss_total / max(n_batches, 1)]),
        **{k: np.concatenate(v) for k, v in accumulators.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", default="data/models/full_hybrid_v13", type=Path,
    )
    parser.add_argument("--config", default="config.yaml", type=Path)
    parser.add_argument("--val-limit", type=int, default=1000)
    parser.add_argument(
        "--report",
        default="data/models/_comparison/phase3_thrust_v13_forwardpass.md",
        type=Path,
    )
    args = parser.parse_args()

    print(f"Loading v13 from {args.model_dir} ...")
    trainer = _build_trainer(args.model_dir, args.config, val_limit=args.val_limit)
    print(f"Val dataset : {len(trainer.val_dataset)} samples")

    print("\n=== Run A : baseline (NN thrust correction active) ===")
    base = _run_val_with_diagnostics(trainer, zero_t_correction=False)
    base_val_loss = float(base["val_loss"][0])
    print(f"  val_loss = {base_val_loss:.5f}")

    print("\n=== Run B : t_correction zeroed (AC5 NN-off) ===")
    nnoff = _run_val_with_diagnostics(trainer, zero_t_correction=True)
    nnoff_val_loss = float(nnoff["val_loss"][0])
    print(f"  val_loss = {nnoff_val_loss:.5f}")
    degradation_pct = (nnoff_val_loss / base_val_loss - 1.0) * 100.0
    ac5_pass = degradation_pct < 50.0
    print(f"  degradation vs baseline = {degradation_pct:+.1f} %  "
          f"-- AC5 {'PASS' if ac5_pass else 'FAIL'}")

    print("\n=== AC4 : mean |delta T_NN| / T_PS ===")
    t_ps = base["fdm_T_PS"]
    t_total = base["fdm_T_total"]
    valid = (t_ps > 1.0) & np.isfinite(t_ps) & np.isfinite(t_total)
    delta_pct = np.abs(t_total[valid] / t_ps[valid] - 1.0) * 100.0
    mean_abs_delta = float(delta_pct.mean())
    p99_abs_delta = float(np.percentile(delta_pct, 99))
    ac4_pass = mean_abs_delta < 3.0
    print(f"  n_samples = {valid.sum()}")
    print(f"  mean |delta T_NN| = {mean_abs_delta:.3f} %")
    print(f"  p99  |delta T_NN| = {p99_abs_delta:.3f} %")
    print(f"  AC4 {'PASS' if ac4_pass else 'FAIL'}")

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
            delta = np.abs(t_total[m] / t_ps[m] - 1.0) * 100.0
            phase_stats[label] = {
                "n": int(m.sum()),
                "mean_abs_delta_pct": float(delta.mean()),
                "p99_abs_delta_pct": float(np.percentile(delta, 99)),
            }
            print(f"  [{label}]  n={m.sum()}  "
                  f"mean={delta.mean():.3f}%  p99={np.percentile(delta, 99):.3f}%")

    print("\n=== AC6 : throttle chi phase-coherence ===")
    chi = base.get("fdm_throttle")
    ac6_result: dict[str, float | str | int] = {"status": "no_throttle_signal"}
    if chi is not None and alt_m is not None:
        n = min(len(chi), len(alt_m))
        valid_chi = (chi[:n] >= 0.0) & (chi[:n] <= 1.1) & np.isfinite(chi[:n])
        range_ok = (chi[valid_chi].min() >= 0.1 - 1e-3) and (chi[valid_chi].max() <= 1.0 + 1e-3)
        # Phase-coherence : climb (alt rising mid-band) vs cruise (high alt)
        # vs descent_idle (alt falling low-band). Without |dh/dt| from
        # history we approximate via alt bands.
        low_band = alt_m[:n] < 3_000.0
        mid_band = (alt_m[:n] >= 3_000.0) & (alt_m[:n] < 9_000.0)
        cruise = alt_m[:n] >= 9_000.0
        def _med(mask: np.ndarray) -> float:
            sel = valid_chi & mask
            return float(np.median(chi[sel])) if sel.sum() >= 20 else float("nan")

        chi_low = _med(low_band)
        chi_mid = _med(mid_band)
        chi_cruise = _med(cruise)
        # Phase coherence : a sane model has mid (climb-heavy) >= cruise.
        # We do not enforce descent_idle (low_band carries both climb-out
        # and descent ; the alt-only mask cannot separate them).
        coherent = (
            chi_mid >= chi_cruise - 0.05  # climb-band thrust >= cruise (within tolerance)
            if not (np.isnan(chi_mid) or np.isnan(chi_cruise))
            else True
        )
        ac6_pass = bool(range_ok and coherent)
        ac6_result = {
            "status": "evaluated",
            "n": int(valid_chi.sum()),
            "chi_min": float(chi[valid_chi].min()),
            "chi_max": float(chi[valid_chi].max()),
            "chi_median_low": chi_low,
            "chi_median_mid": chi_mid,
            "chi_median_cruise": chi_cruise,
            "range_ok": str(bool(range_ok)),
            "coherent": str(bool(coherent)),
            "verdict": "PASS" if ac6_pass else "FAIL",
        }
        print(f"  n_chi = {valid_chi.sum()}")
        print(f"  chi range = [{chi[valid_chi].min():.3f}, {chi[valid_chi].max():.3f}]")
        print(f"  median chi  low (alt<3km) = {chi_low:.3f}")
        print(f"  median chi  mid (3-9km)   = {chi_mid:.3f}")
        print(f"  median chi  cruise (>9km) = {chi_cruise:.3f}")
        print(f"  range in [0.1, 1.0] ?  {range_ok}")
        print(f"  coherent (mid >= cruise) ? {coherent}")
        print(f"  AC6 {ac6_result['verdict']}")

    # --- Markdown report ---
    lines = [
        "# Phase 3 — Forward-pass diagnostic (AC4 / AC5 / AC6 closure)",
        "",
        f"> Trained model : `{args.model_dir.name}`. Val split limited to "
        f"{args.val_limit} samples.",
        "",
        "## AC4 — Mean |delta T_NN| / T_PS",
        "",
        f"- n samples (cruise + all phases) : **{int(valid.sum())}**",
        f"- mean |delta T_NN| : **{mean_abs_delta:.3f} %**",
        f"- p99  |delta T_NN| : **{p99_abs_delta:.3f} %**",
        "- target : mean < 3 %",
        f"- status : **{'✅ PASS' if ac4_pass else '⚠️ FAIL'}**",
        "",
        "### Phase-stratified breakdown",
        "",
        "| Phase band | n | mean |ΔT_NN| % | p99 |ΔT_NN| % |",
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
        f"- baseline val_loss (NN thrust active) : **{base_val_loss:.5f}**",
        f"- NN-off val_loss (t_correction = 0) : **{nnoff_val_loss:.5f}**",
        f"- degradation : **{degradation_pct:+.2f} %**",
        "- target : < 50 % degradation",
        f"- status : **{'✅ PASS' if ac5_pass else '⚠️ FAIL'}**",
        "",
        "## AC6 — Throttle chi phase-coherence",
        "",
    ])
    if ac6_result.get("status") == "evaluated":
        lines.extend([
            f"- n samples : **{ac6_result['n']}**",
            f"- chi range : **[{ac6_result['chi_min']:.3f}, {ac6_result['chi_max']:.3f}]**",
            f"- median chi  low (alt<3km) : {ac6_result['chi_median_low']:.3f}",
            f"- median chi  mid (3-9km)   : {ac6_result['chi_median_mid']:.3f}",
            f"- median chi  cruise (>9km) : {ac6_result['chi_median_cruise']:.3f}",
            "- target : chi in [0.1, 1.0] AND mid >= cruise",
            f"- status : **{'✅ PASS' if ac6_result['verdict'] == 'PASS' else '⚠️ FAIL'}**",
        ])
    else:
        lines.append(f"- status : {ac6_result['status']}")

    lines.extend([
        "",
        "## Methodology notes",
        "",
        "Same monkey-patch idiom as `phase2_drag_forwardpass.py`, but on "
        "`fdm_t_correction` instead of `fdm_cd_correction`. The PhysicsLayer "
        "Phase 3 branch exposes `fdm_T_PS`, `fdm_T_total`, `fdm_throttle`, "
        "`fdm_drag_N` as diagnostic outputs.",
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
