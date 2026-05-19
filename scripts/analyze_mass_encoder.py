"""Analyse a trained ``node_adsb_hybrid_v1`` model end-to-end.

Reports:

1. ``mass_encoder.effective_coefficients()`` after training.
2. Distribution of predicted ``m_0`` on the val set (median, p5/p95,
   percentage saturated at OEW / MTOW).
3. ``identifiability_test(factor=1.3)`` — Phase 1 gate: ratio > 1.20.
4. Loss curves summary (best epoch, train/val).

Run after a training run completes::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/analyze_mass_encoder.py --model-name full_hybrid

Optional flags:

* ``--train-limit / --val-limit`` — sub-sample for faster diagnostics.
* ``--device`` — defaults to cpu.
* ``--factor`` — perturbation factor for the identifiability test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import structlog

structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
log = structlog.get_logger("analyze_mass_encoder")


def _load_training_losses(model_dir: Path) -> tuple[int, float, float]:
    """Return ``(best_epoch, best_val_loss, last_train_loss)``."""
    csv_path = model_dir / "training_losses.csv"
    if not csv_path.exists():
        return -1, float("nan"), float("nan")
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return -1, float("nan"), float("nan")
    best_row = min(rows, key=lambda r: float(r["val_loss"]))
    return (
        int(float(best_row["epoch"])),
        float(best_row["val_loss"]),
        float(rows[-1]["train_loss"]),
    )


def _format_m0_distribution(m0_values: np.ndarray, oew: float, mtow: float) -> str:
    """Return a multi-line text summary of the ``m_0`` distribution."""
    if m0_values.size == 0:
        return "  (no m_0 predictions)"
    plage = mtow - oew
    saturated_low = np.sum(m0_values < oew + 0.05 * plage) / m0_values.size
    saturated_high = np.sum(m0_values > mtow - 0.05 * plage) / m0_values.size
    p5, p25, p50, p75, p95 = np.percentile(m0_values, [5, 25, 50, 75, 95])
    lines = [
        f"  n_segments_evaluated = {m0_values.size}",
        f"  median (kg)          = {p50:,.0f}",
        f"  p5/p95 (kg)          = {p5:,.0f} / {p95:,.0f}",
        f"  p25/p75 (kg)         = {p25:,.0f} / {p75:,.0f}",
        f"  saturated OEW (<5%)  = {100 * saturated_low:.1f}%   (target < 5%)",
        f"  saturated MTOW (>95%) = {100 * saturated_high:.1f}%   (target < 5%)",
        f"  bounds               = [{oew:,.0f}, {mtow:,.0f}] kg",
    ]
    return "\n".join(lines)


def _predict_m0_on_val(trainer, val_dataset) -> np.ndarray:
    """Run the MassEncoder on every val segment's t_0 features."""
    if trainer.mass_encoder is None:
        return np.empty((0,), dtype=np.float32)
    trainer.model.eval()
    trainer.mass_encoder.eval()
    all_m0: list[float] = []
    with torch.no_grad():
        for sample in val_dataset:
            if sample.flight_features is None:
                continue
            features = sample.flight_features[0].unsqueeze(0).to(trainer.device)
            m0 = trainer.mass_encoder(features)
            all_m0.append(float(m0.item()))
    return np.asarray(all_m0, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse a trained hybrid mass-encoder model.")
    parser.add_argument(
        "--model-name",
        default="full_hybrid",
        help="Sub-directory name under data/models/ containing the checkpoint.",
    )
    parser.add_argument("--config", default="config.yaml", help="Pipeline config YAML.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--val-limit", type=int, default=2000)
    parser.add_argument("--factor", type=float, default=1.3)
    args = parser.parse_args()

    from node_fdm.architectures.registry import get as get_arch_spec
    from node_fdm.loader import get_train_val_data
    from node_fdm.trainer import ODETrainer, TrainingConfig
    from node_fdm_data.delta import read_delta_table
    from node_fdm_data.schemas.adsb_hybrid import (
        A320_MTOW_KG,
        A320_OEW_KG,
    )

    # Trigger architecture self-registration.
    import node_fdm.architectures.adsb_hybrid  # noqa: F401

    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(Path(args.config))
    models_dir = cfg.paths.resolve("models_dir")
    model_dir = models_dir / args.model_name
    if not model_dir.exists():
        log.error("model_dir_missing", path=str(model_dir))
        return 1

    meta_path = model_dir / "meta.json"
    meta = json.loads(meta_path.read_text())

    spec = get_arch_spec(meta["architecture_name"])
    flight_feature_cols = list(spec.flight_feature_cols or [])

    delta_path = cfg.paths.resolve("delta_table")
    log.info("reading_delta", path=str(delta_path))
    df = read_delta_table(delta_path).filter(pl.col("fdm_flag_valid"))

    if cfg.typecodes:
        df = df.filter(pl.col("meta_aircraft_type").is_in(cfg.typecodes))

    dx_col_names = [c for _, c in spec.dx_cols]
    log.info("loading_data", train_limit=args.train_limit, val_limit=args.val_limit)
    train_ds, val_ds = get_train_val_data(
        data_df=df,
        x_cols=spec.x_cols,
        u_cols=spec.u_cols,
        e_cols=spec.e0_cols,
        e1_cols=spec.e1_cols,
        dx_cols=dx_col_names,
        seq_len=meta["seq_len"],
        shift=meta["shift"],
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        flight_feature_cols=flight_feature_cols or None,
        require_routing=True,
    )

    config = TrainingConfig(
        architecture_name=meta["architecture_name"],
        model_name=args.model_name,
        model_params=tuple(meta["model_params"]),
        step=meta["step"],
        shift=meta["shift"],
        seq_len=meta["seq_len"],
        lr=meta["lr"],
        batch_size=meta["batch_size"],
        epochs=meta["epochs"],
        method=meta["method"],
        seed=meta.get("seed"),
        activation=meta.get("activation", "silu"),
        use_mode_weights=meta.get("use_mode_weights", False),
        mode_weight_alpha=meta.get("mode_weight_alpha", 0.5),
    )

    trainer = ODETrainer(
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_dir=models_dir,
        device=args.device,
    )
    trainer.load_model_weights()

    print()
    print("=" * 70)
    print(f"Mass-encoder analysis — model: {args.model_name}")
    print(f"Architecture: {meta['architecture_name']}")
    print(f"Device: {args.device}    seed: {meta.get('seed')}")
    print(f"epochs={meta['epochs']}  batch={meta['batch_size']}  seq_len={meta['seq_len']}")
    print("=" * 70)

    # --- 1. Loss summary -----------------------------------------------
    best_epoch, best_val, last_train = _load_training_losses(model_dir)
    print()
    print("[1] Training-loss summary")
    print(f"  best_epoch     = {best_epoch}")
    print(f"  best_val_loss  = {best_val:.6f}")
    print(f"  last_train_loss= {last_train:.6f}")

    # --- 2. Effective coefficients -------------------------------------
    if trainer.mass_encoder is None:
        log.error("no_mass_encoder", msg="this script expects the hybrid arch")
        return 2
    coefs = trainer.mass_encoder.effective_coefficients()
    print()
    print("[2] mass_encoder.effective_coefficients()")
    for k, v in coefs.items():
        sign = "+" if v >= 0 else "-"
        print(f"  {k:<32} = {sign}{abs(v):.4f}")

    # --- 3. m_0 distribution -------------------------------------------
    m0 = _predict_m0_on_val(trainer, val_ds)
    print()
    print("[3] Predicted m_0 distribution on val set")
    print(_format_m0_distribution(m0, A320_OEW_KG, A320_MTOW_KG))

    # --- 4. Identifiability test ---------------------------------------
    print()
    print(f"[4] Identifiability test (factor = {args.factor})")
    result = trainer.identifiability_test(factor=args.factor)
    print(f"  baseline_mse  = {result['baseline_mse']:.6f}")
    print(f"  perturbed_mse = {result['perturbed_mse']:.6f}")
    print(f"  ratio         = {result['ratio']:.4f}")
    ratio = result["ratio"]
    if ratio > 1.20:
        verdict = "PASS (ratio > 1.20 — identifiable)"
    elif ratio > 1.0:
        verdict = "weak (1.0 < ratio < 1.20 — identifiable but borderline)"
    else:
        verdict = "FAIL (ratio <= 1.0 — m_0 not identified, check input separation)"
    print(f"  verdict       = {verdict}")

    # --- 5. Phase 1 success criteria checklist -------------------------
    print()
    print("[5] Phase 1 success criteria checklist (PHASE_1_MASS_ENCODER.md §9)")
    a320_median = float(np.median(m0)) if m0.size else float("nan")
    in_window = 60_000 <= a320_median <= 73_000
    sat_oew = float(np.mean(m0 < A320_OEW_KG + 0.05 * (A320_MTOW_KG - A320_OEW_KG))) if m0.size else 0.0
    sat_mtow = float(np.mean(m0 > A320_MTOW_KG - 0.05 * (A320_MTOW_KG - A320_OEW_KG))) if m0.size else 0.0
    sign_checks = []
    for col in spec.flight_feature_cols or []:
        v = coefs.get(col, 0.0)
        sign_checks.append((col, v))
    b_dist_total = abs(coefs.get("dist_total_flight", 0.0))
    checks = [
        ("m_0 median in [60t, 73t]", in_window, f"{a320_median:,.0f} kg"),
        ("Saturated OEW (<5%)", sat_oew < 0.05, f"{100 * sat_oew:.1f}%"),
        ("Saturated MTOW (<5%)", sat_mtow < 0.05, f"{100 * sat_mtow:.1f}%"),
        ("|b_dist_total| > 0.1", b_dist_total > 0.1, f"{b_dist_total:.3f}"),
        ("Identifiability ratio > 1.20", ratio > 1.20, f"{ratio:.3f}"),
    ]
    for name, passed, value in checks:
        marker = "OK" if passed else "FAIL"
        print(f"  [{marker}] {name:<32} {value}")

    print()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
