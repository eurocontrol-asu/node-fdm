#!/usr/bin/env python3
"""Diagnostic: inspect training data produced by the dataloader.

Loads training batches through the exact same pipeline as ``fdm train``
and produces distribution plots for every x/u/e/dx column, plus sample
sequence visualisations.  Designed to compare against node-fdm legacy
and detect dataloader issues (wrong units, flat derivatives, etc.).

Run:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/validate_04_bis_training_data.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm-pipeline" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm-data" / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(ok: bool, msg: str) -> bool:
    icon = "✅" if ok else "❌"
    print(f"  {icon} {msg}")
    return ok


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Data loading (same path as fdm train)
# ---------------------------------------------------------------------------


def load_training_data(
    typecode: str = "A320",
) -> tuple[list[str], list[str], list[str], list[str], Any, Any, dict[str, dict[str, float]]]:
    """Load train/val datasets exactly as run_training does.

    Returns:
        (x_cols, u_cols, e_cols, dx_col_names, train_ds, val_ds, stats_dict)
    """
    from node_fdm.dataset import compute_stats
    from node_fdm.loader import get_train_val_data
    from node_fdm_pipeline.config import PipelineConfig
    from node_fdm_pipeline.resolver import resolve_architecture

    cfg = PipelineConfig.from_yaml(PROJECT_ROOT / "config.yaml")
    info = resolve_architecture("opensky")
    importlib.import_module(info.architecture_import)

    process_dir = cfg.paths.resolve("process_dir")
    split_csv = process_dir / "dataset_split.csv"
    split_df = pl.read_csv(split_csv)
    data_df = split_df.filter(pl.col("aircraft_type") == typecode)

    dx_col_names = [col for _, col in info.dx_cols]

    train_ds, val_ds = get_train_val_data(
        data_df=data_df,
        x_cols=info.x_cols,
        u_cols=info.u_cols,
        e_cols=info.e0_cols,
        dx_cols=dx_col_names,
        seq_len=60,
        shift=60,
        preprocessing_fn=info.preprocessing_fn,
        segment_filter_fn=info.segment_filter_fn,
        train_limit=5000,
        val_limit=5000,
    )

    stats_dict = compute_stats(
        list(train_ds),
        x_cols=info.x_cols,
        u_cols=info.u_cols,
        e_cols=info.e0_cols,
        dx_cols=dx_col_names,
    )

    return info.x_cols, info.u_cols, info.e0_cols, dx_col_names, train_ds, val_ds, stats_dict


# ---------------------------------------------------------------------------
# Console diagnostics
# ---------------------------------------------------------------------------


def print_column_stats(
    label: str,
    cols: list[str],
    data: np.ndarray,
) -> None:
    """Print per-column statistics table.

    Args:
        label: Group label (x/u/e/dx).
        cols: Column names.
        data: Array of shape (n_samples * seq_len, n_cols).
    """
    _section(f"{label} columns — distribution")
    hdr = f"  {'column':<20s} {'mean':>12s} {'std':>12s} {'min':>12s}"
    hdr += f" {'p01':>12s} {'p50':>12s} {'p99':>12s} {'max':>12s} {'%zero':>8s}"
    print(hdr)
    sep = f"  {'─' * 20} {'─' * 12} {'─' * 12} {'─' * 12}"
    sep += f" {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 8}"
    print(sep)
    for i, col in enumerate(cols):
        v = data[:, i]
        pct_zero = 100.0 * np.sum(v == 0.0) / len(v) if len(v) > 0 else 0.0
        print(
            f"  {col:<20s}"
            f" {v.mean():>12.4f}"
            f" {v.std():>12.4f}"
            f" {v.min():>12.4f}"
            f" {np.percentile(v, 1):>12.4f}"
            f" {np.percentile(v, 50):>12.4f}"
            f" {np.percentile(v, 99):>12.4f}"
            f" {v.max():>12.4f}"
            f" {pct_zero:>7.1f}%"
        )


def print_normalization_stats(
    cols: list[str],
    stats_dict: dict[str, dict[str, float]],
    label: str,
) -> None:
    """Print normalization stats used by the model."""
    _section(f"{label} — normalization stats (from compute_stats)")
    print(f"  {'column':<20s} {'mean':>12s} {'std':>12s} {'max(p99.5)':>12s}")
    print(f"  {'─' * 20} {'─' * 12} {'─' * 12} {'─' * 12}")
    for col in cols:
        s = stats_dict.get(col, {})
        print(
            f"  {col:<20s}"
            f" {s.get('mean', 0):>12.4f}"
            f" {s.get('std', 0):>12.4f}"
            f" {s.get('max', 0):>12.4f}"
        )


def check_anomalies(
    cols: list[str],
    data: np.ndarray,
    label: str,
) -> bool:
    """Flag columns with suspicious distributions."""
    _section(f"{label} — anomaly checks")
    ok = True

    for i, col in enumerate(cols):
        v = data[:, i]
        pct_zero = 100.0 * np.sum(v == 0.0) / len(v) if len(v) > 0 else 0.0
        std = v.std()

        if std < 1e-10:
            ok = _status(False, f"{col}: constant (std={std:.2e})") and ok
        elif pct_zero > 50.0:
            ok = _status(False, f"{col}: {pct_zero:.1f}% zeros") and ok
        elif np.isnan(v).any():
            ok = _status(False, f"{col}: contains NaN") and ok
        else:
            _status(True, f"{col}: OK (std={std:.4f}, {pct_zero:.1f}% zeros)")

    return ok


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_distributions(
    all_cols: list[str],
    data: np.ndarray,
    groups: list[tuple[str, list[str]]],
    figure_dir: Path,
) -> None:
    """Generate faceted histogram for all training columns."""
    import altair as alt

    records = []
    for i, col in enumerate(all_cols):
        v = data[:, i]
        # Subsample for plot performance
        idx = np.random.default_rng(42).choice(len(v), size=min(5000, len(v)), replace=False)
        for val in v[idx]:
            # Find group
            group = "?"
            for g_name, g_cols in groups:
                if col in g_cols:
                    group = g_name
                    break
            records.append({"column": col, "value": float(val), "group": group})

    df = pl.DataFrame(records)

    chart = (
        alt.Chart(df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=50)),
            y=alt.Y("count():Q"),
            color="group:N",
        )
        .facet(facet="column:N", columns=4)
        .resolve_scale(x="independent", y="independent")
        .properties(title="Training data distributions")
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "04_bis_distributions.html"
    chart.save(output)
    _status(True, f"Distribution chart saved to {output}")


def plot_sample_sequences(
    x_cols: list[str],
    u_cols: list[str],
    dx_cols: list[str],
    train_ds: Any,
    figure_dir: Path,
    n_samples: int = 3,
) -> None:
    """Plot a few sample sequences (x, u, dx over timesteps)."""
    import altair as alt

    records = []
    for s_idx in range(min(n_samples, len(train_ds))):
        sample = train_ds[s_idx]
        seq_len = sample.x.shape[0]

        for t in range(seq_len):
            for j, col in enumerate(x_cols):
                records.append(
                    {
                        "sample": s_idx,
                        "t": t,
                        "column": col,
                        "value": float(sample.x[t, j]),
                        "group": "x",
                    }
                )
            for j, col in enumerate(u_cols):
                records.append(
                    {
                        "sample": s_idx,
                        "t": t,
                        "column": col,
                        "value": float(sample.u[t, j]),
                        "group": "u",
                    }
                )
            for j, col in enumerate(dx_cols):
                records.append(
                    {
                        "sample": s_idx,
                        "t": t,
                        "column": col,
                        "value": float(sample.dx[t, j]),
                        "group": "dx",
                    }
                )

    df = pl.DataFrame(records)

    chart = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Timestep"),
            y=alt.Y("value:Q"),
            color="sample:N",
        )
        .facet(facet="column:N", columns=4)
        .resolve_scale(y="independent")
        .properties(title="Sample training sequences")
    )

    output = figure_dir / "04_bis_sample_sequences.html"
    chart.save(output)
    _status(True, f"Sample sequences chart saved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  Training Data Diagnostic (04 bis)")
    print("=" * 60)

    x_cols, u_cols, e_cols, dx_cols, train_ds, val_ds, stats_dict = load_training_data()

    print(f"\n  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    print(f"  Seq length:    {train_ds[0].x.shape[0]}")

    # Flatten all training samples into (N * seq_len, n_cols) arrays
    import torch

    x_all = torch.cat([s.x for s in train_ds], dim=0).numpy()
    u_all = torch.cat([s.u for s in train_ds], dim=0).numpy()
    e_all = torch.cat([s.e for s in train_ds], dim=0).numpy()
    dx_all = torch.cat([s.dx for s in train_ds], dim=0).numpy()

    # Print distributions
    print_column_stats("x (state)", x_cols, x_all)
    print_column_stats("u (control)", u_cols, u_all)
    print_column_stats("e (environment)", e_cols, e_all)
    print_column_stats("dx (derivatives)", dx_cols, dx_all)

    # Normalization stats
    all_cols = x_cols + u_cols + e_cols + dx_cols
    print_normalization_stats(all_cols, stats_dict, "all")

    # Anomaly checks
    ok = True
    ok = check_anomalies(x_cols, x_all, "x") and ok
    ok = check_anomalies(u_cols, u_all, "u") and ok
    ok = check_anomalies(e_cols, e_all, "e") and ok
    ok = check_anomalies(dx_cols, dx_all, "dx") and ok

    # Plots
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(PROJECT_ROOT / "config.yaml")
    figure_dir = cfg.paths.data_dir / "figures"

    _section("Plots")
    data_all = np.concatenate([x_all, u_all, e_all, dx_all], axis=1)
    groups = [("x", x_cols), ("u", u_cols), ("e", e_cols), ("dx", dx_cols)]
    plot_distributions(all_cols, data_all, groups, figure_dir)
    plot_sample_sequences(x_cols, u_cols, dx_cols, train_ds, figure_dir)

    # Summary
    print("\n" + "=" * 60)
    icon = "✅" if ok else "⚠️"
    print(f"  {icon} Diagnostic complete: {'no anomalies' if ok else 'anomalies detected'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
