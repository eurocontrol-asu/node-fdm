#!/usr/bin/env python3
"""Validate training pipeline output (model artifacts + convergence).

Checks model artifact completeness, metadata correctness, weight health,
forward pass functionality, and produces a train/val loss convergence plot
after running ``fdm train``.

Run:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/validate_04_training.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> tuple[Path, Path, list[str]]:
    """Load pipeline config and return (models_dir, figure_dir, typecodes)."""
    sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm-pipeline" / "src"))
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(PROJECT_ROOT / "config.yaml")
    models_dir = cfg.paths.resolve("models_dir")
    figure_dir = cfg.paths.data_dir / "figures"
    return models_dir, figure_dir, cfg.typecodes


def _status(ok: bool, msg: str) -> bool:
    """Print a status line and return the check result."""
    icon = "✅" if ok else "❌"
    print(f"  {icon} {msg}")
    return ok


# ---------------------------------------------------------------------------
# T1 — Model artifacts exist (AC1)
# ---------------------------------------------------------------------------


def validate_artifacts(model_dir: Path) -> tuple[bool, list[Path]]:
    """Check that meta.json and layer checkpoint .pt files exist.

    Args:
        model_dir: Path to the trained model directory.

    Returns:
        Tuple of (all checks passed, list of .pt checkpoint paths).
    """
    ok = True

    meta_path = model_dir / "meta.json"
    ok = _status(meta_path.exists(), f"meta.json found at {meta_path}") and ok

    pt_files = sorted(model_dir.glob("*.pt"))
    ok = (
        _status(
            len(pt_files) > 0, f"{len(pt_files)} layer checkpoint(s): {[f.name for f in pt_files]}"
        )
        and ok
    )

    return ok, pt_files


# ---------------------------------------------------------------------------
# T2 — Metadata validation (AC2)
# ---------------------------------------------------------------------------


def validate_metadata(model_dir: Path) -> tuple[bool, dict[str, object] | None]:
    """Parse meta.json and validate required fields.

    Args:
        model_dir: Path to the trained model directory.

    Returns:
        Tuple of (all checks passed, parsed meta dict or None).
    """
    meta_path = model_dir / "meta.json"
    if not meta_path.exists():
        _status(False, "meta.json not found — skipping metadata validation")
        return False, None

    meta = json.loads(meta_path.read_text())
    ok = True

    # Required top-level fields
    for field in ("architecture_name", "model_params", "step", "lr", "seq_len", "batch_size"):
        ok = _status(field in meta, f"Field '{field}': {meta.get(field)}") and ok

    # stats_dict with per-column mean/std/max
    has_stats = "stats_dict" in meta and isinstance(meta["stats_dict"], dict)
    ok = _status(has_stats, f"stats_dict: {len(meta.get('stats_dict', {}))} columns") and ok

    if has_stats:
        for col, stats in meta["stats_dict"].items():
            for key in ("mean", "std", "max"):
                if key not in stats:
                    ok = _status(False, f"stats_dict['{col}'] missing '{key}'")

    _status(True, f"Architecture: {meta.get('architecture_name')}")
    return ok, meta


# ---------------------------------------------------------------------------
# T3 — Weight health (AC3)
# ---------------------------------------------------------------------------


def validate_weights(model_dir: Path, pt_files: list[Path]) -> tuple[bool, int]:
    """Load layer checkpoints and verify all weights are finite.

    Args:
        model_dir: Path to the trained model directory.
        pt_files: List of .pt checkpoint file paths.

    Returns:
        Tuple of (all weights finite, total parameter count).
    """
    import torch

    ok = True
    total_params = 0

    for pt_path in pt_files:
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
        layer_state = ckpt.get("layer_state", ckpt)

        for name, param in layer_state.items():
            if isinstance(param, torch.Tensor):
                finite = bool(torch.isfinite(param).all())
                if not finite:
                    ok = _status(False, f"NaN/Inf in {pt_path.name}:{name}")
                total_params += param.numel()

    ok = _status(ok, f"All weights finite ({total_params:,} parameters)") and ok
    return ok, total_params


# ---------------------------------------------------------------------------
# T4 — Forward pass (AC4)
# ---------------------------------------------------------------------------


def validate_forward_pass(model_dir: Path) -> bool:
    """Instantiate predictor and run a forward pass with synthetic input.

    Args:
        model_dir: Path to the trained model directory.

    Returns:
        True if forward pass produces finite output.
    """
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm" / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm-data" / "src"))
    from node_fdm.predictor import NodeFDMPredictor

    predictor = NodeFDMPredictor(model_path=model_dir, device="cpu")

    # Synthetic inputs: x_init (4,), u_seq (60, 4), e_seq (60, 4)
    x_init = np.array([0.0, 10000.0, 0.0, 250.0], dtype=np.float32)
    u_seq = np.tile([10000.0, 0.78, 150.0, 0.0], (60, 1)).astype(np.float32)
    e_seq = np.tile([5.0, 100.0, 500.0, 220.0], (60, 1)).astype(np.float32)

    result = predictor.predict_flight(x_init, u_seq, e_seq)

    ok = isinstance(result, dict) and len(result) > 0
    ok = _status(ok, f"Forward pass returned {len(result)} columns: {list(result.keys())}") and ok

    all_finite = all(np.isfinite(v).all() for v in result.values())
    ok = _status(all_finite, "All predictions finite") and ok

    return ok


# ---------------------------------------------------------------------------
# Plot — Convergence curve (AC5)
# ---------------------------------------------------------------------------


def plot_convergence(model_dir: Path, figure_dir: Path) -> bool:
    """Generate train/val loss convergence chart from training_losses.csv.

    Args:
        model_dir: Path to the trained model directory.
        figure_dir: Directory to save the chart.

    Returns:
        True if chart was saved successfully.
    """
    import altair as alt
    import polars as pl

    csv_path = model_dir / "training_losses.csv"
    if not csv_path.exists():
        return _status(False, f"training_losses.csv not found at {csv_path}")

    losses = pl.read_csv(csv_path)

    # Melt to long format for Altair
    long = losses.unpivot(
        index="epoch",
        on=["train_loss", "val_loss"],
        variable_name="split",
        value_name="loss",
    )

    chart = (
        alt.Chart(long.to_pandas())
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("loss:Q", title="Loss", scale=alt.Scale(type="log")),
            color=alt.Color("split:N", title="Split"),
            tooltip=["epoch", "split", "loss"],
        )
        .properties(title="Training convergence — train vs val loss", width=600, height=300)
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "04_training_loss.html"
    chart.save(output)
    return _status(True, f"Convergence chart saved to {output}")


# ---------------------------------------------------------------------------
# Metrics (AC6)
# ---------------------------------------------------------------------------


def print_metrics(model_dir: Path, total_params: int) -> None:
    """Print training metrics summary.

    Args:
        model_dir: Path to the trained model directory.
        total_params: Total number of model parameters.
    """
    import polars as pl

    _status(True, f"Model parameters: {total_params:,}")

    csv_path = model_dir / "training_losses.csv"
    if not csv_path.exists():
        _status(False, "training_losses.csv not found — no loss metrics")
        return

    losses = pl.read_csv(csv_path)
    n_epochs = len(losses)
    final = losses.tail(1)
    train_loss = final["train_loss"][0]
    val_loss = final["val_loss"][0]
    ratio = val_loss / train_loss if train_loss > 0 else float("inf")

    _status(True, f"Epochs: {n_epochs}")
    _status(True, f"Final train loss: {train_loss:.6f}")
    _status(True, f"Final val loss:   {val_loss:.6f}")
    _status(ratio < 2.0, f"Val/train ratio:  {ratio:.2f} (< 2.0 = no severe overfit)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all training validation checks."""
    print("=" * 60)
    print("  Training Validation (04)")
    print("=" * 60)

    models_dir, figure_dir, typecodes = _load_config()

    all_ok = True

    for acft in typecodes:
        model_dir = models_dir / f"opensky_2025_{acft}"
        print(f"\n{'─' * 60}")
        print(f"  Typecode: {acft} — {model_dir}")
        print(f"{'─' * 60}")

        if not model_dir.exists():
            _status(False, f"Model directory not found: {model_dir}")
            all_ok = False
            continue

        print("\n--- T1: Model artifacts ---")
        t1_ok, pt_files = validate_artifacts(model_dir)

        print("\n--- T2: Metadata ---")
        t2_ok, _meta = validate_metadata(model_dir)

        print("\n--- T3: Weight health ---")
        if pt_files:
            t3_ok, total_params = validate_weights(model_dir, pt_files)
        else:
            t3_ok = _status(False, "No .pt files — skipping weight check")
            total_params = 0

        print("\n--- T4: Forward pass ---")
        if t1_ok and t2_ok:
            t4_ok = validate_forward_pass(model_dir)
        else:
            t4_ok = _status(False, "Skipped — artifacts or metadata invalid")

        print("\n--- Plot: Convergence curve ---")
        plot_convergence(model_dir, figure_dir)

        print("\n--- Metrics ---")
        print_metrics(model_dir, total_params)

        all_ok = all_ok and t1_ok and t2_ok and t3_ok and t4_ok

    # Summary
    print("\n" + "=" * 60)
    icon = "✅" if all_ok else "❌"
    print(f"  {icon} Overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
