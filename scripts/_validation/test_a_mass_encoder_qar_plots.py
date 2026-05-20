"""Visualizations for Test A — MassEncoder QAR validation.

Generates two PNG figures next to ``test_a_mass_encoder_qar.md``:

1. ``test_a_scatter.png`` — scatter ``m_predicted vs m_truth`` per model
   with the identity line and OEW/MTOW bounds. Reveals whether predictions
   cluster on the identity line (good) or on horizontal bands (sigmoid
   saturation).

2. ``test_a_histograms.png`` — three histograms side-by-side: ``m_truth``,
   ``m_v2``, ``m_v3``. Confirms the bimodal pattern (saturation at OEW
   and MTOW) flagged in PHASE_1.5_LESSONS §13.2.

Re-runs the predictions from scratch — no dependency on the previous
markdown output. Self-contained so it can be re-run after Round 4-prime
to compare before/after.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from node_fdm_data.schemas.adsb_hybrid import A320_MTOW_KG, A320_OEW_KG

# Sibling module — add the script directory to sys.path so we can import it
# without making scripts/ a package.
sys.path.insert(0, str(Path(__file__).parent))
from test_a_mass_encoder_qar import (  # noqa: E402
    _dedup_qar_files,
    _find_stable_cruise_idx,
    compute_qar_features,
    load_mass_encoder,
    predict_mass,
)


def _collect(qar_dir: Path, models_dir: Path, model_names: list[str]) -> dict:
    encoders = {n: load_mass_encoder(models_dir / n) for n in model_names}
    files = _dedup_qar_files(sorted(qar_dir.glob("*.parquet")))

    truth: list[float] = []
    preds: dict[str, list[float]] = {n: [] for n in model_names}

    for f in files:
        df = pl.read_parquet(f)
        gw = df["SYS__GW"].to_numpy()
        gw_valid = np.where(np.isfinite(gw) & (gw > 0))[0]
        if len(gw_valid) == 0:
            continue
        cruise_idx = _find_stable_cruise_idx(df)
        seg_candidates = gw_valid[gw_valid >= cruise_idx]
        seg_idx = int(seg_candidates[0]) if len(seg_candidates) > 0 else int(gw_valid[0])
        try:
            features = compute_qar_features(df, seg_t0_idx=seg_idx)
        except ValueError:
            continue
        truth.append(float(gw[seg_idx]))
        for n in model_names:
            preds[n].append(predict_mass(encoders[n], features))

    return {"truth": np.asarray(truth), "preds": {n: np.asarray(v) for n, v in preds.items()}}


def _plot_scatter(data: dict, model_names: list[str], out_path: Path) -> None:
    truth = data["truth"]
    fig, axes = plt.subplots(1, len(model_names), figsize=(5.5 * len(model_names), 5.5), sharex=True, sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names, strict=True):
        pred = data["preds"][name]
        err = pred - truth
        mae = float(np.mean(np.abs(err)))
        corr = float(np.corrcoef(truth, pred)[0, 1]) if len(truth) >= 3 else math.nan

        lo = min(A320_OEW_KG, truth.min(), pred.min()) - 1000
        hi = max(A320_MTOW_KG, truth.max(), pred.max()) + 1000

        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="identity")
        ax.axhline(A320_OEW_KG, color="grey", lw=0.6, ls=":", alpha=0.6, label="OEW")
        ax.axhline(A320_MTOW_KG, color="grey", lw=0.6, ls=":", alpha=0.6, label="MTOW")
        ax.scatter(truth, pred, c="C0", s=60, alpha=0.75, edgecolors="black", linewidths=0.5)

        for i, (t, p) in enumerate(zip(truth, pred, strict=True)):
            ax.plot([t, t], [t, p], color="C3", lw=0.6, alpha=0.3)  # residual line to identity

        ax.set_xlabel("m_truth = SYS__GW (kg)")
        ax.set_ylabel("m_predicted by MassEncoder (kg)")
        ax.set_title(f"{name}\nMAE = {mae:.0f} kg | corr = {corr:.2f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Test A — MassEncoder prediction vs QAR ground truth\n"
        "Points on the identity line = correct prediction. Horizontal bands = sigmoid saturation.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_histograms(data: dict, model_names: list[str], out_path: Path) -> None:
    truth = data["truth"]
    bins = np.linspace(40_000, 80_000, 17)

    fig, axes = plt.subplots(1, 1 + len(model_names), figsize=(4.5 * (1 + len(model_names)), 4.0), sharey=True)

    axes[0].hist(truth, bins=bins, color="C2", alpha=0.7, edgecolor="black")
    axes[0].axvline(A320_OEW_KG, color="grey", ls=":", lw=0.8, label="OEW")
    axes[0].axvline(A320_MTOW_KG, color="grey", ls=":", lw=0.8, label="MTOW")
    axes[0].set_title(f"QAR ground truth (n={len(truth)})\nrange [{truth.min():.0f}, {truth.max():.0f}]")
    axes[0].set_xlabel("mass (kg)")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for ax, name in zip(axes[1:], model_names, strict=True):
        pred = data["preds"][name]
        ax.hist(pred, bins=bins, color="C0", alpha=0.7, edgecolor="black")
        ax.axvline(A320_OEW_KG, color="grey", ls=":", lw=0.8)
        ax.axvline(A320_MTOW_KG, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{name}\nrange [{pred.min():.0f}, {pred.max():.0f}]")
        ax.set_xlabel("mass (kg)")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "MassEncoder distribution vs QAR truth — saturation diagnostic\n"
        "Truth concentrated in [55k, 72k]. Predictions polarize at OEW/MTOW.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qar-dir", default="/Users/gabriel/Downloads/QAR3")
    parser.add_argument("--models", nargs="+", default=["full_hybrid_v2", "full_hybrid_v3"])
    parser.add_argument("--models-dir", default="data/models")
    args = parser.parse_args()

    qar_dir = Path(args.qar_dir)
    models_dir = Path(args.models_dir)
    out_dir = models_dir / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting predictions...")
    data = _collect(qar_dir, models_dir, args.models)
    print(f"  truth range: [{data['truth'].min():.0f}, {data['truth'].max():.0f}] kg (n={len(data['truth'])})")
    for n in args.models:
        p = data["preds"][n]
        print(f"  {n}: pred range [{p.min():.0f}, {p.max():.0f}] kg")

    scatter_path = out_dir / "test_a_scatter.png"
    hist_path = out_dir / "test_a_histograms.png"
    _plot_scatter(data, args.models, scatter_path)
    _plot_histograms(data, args.models, hist_path)
    print(f"Wrote {scatter_path}")
    print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
