"""Oracle MLP ceiling — non-linear upper bound for the 9 ADS-B features.

Validation-only diagnostic. Complements
``test_a_oracle_feature_ceiling.py`` (linear OLS) with a small MLP
trained in direct supervision against ``SYS__GW`` on the same 9
features, evaluated via K-fold cross-validation.

Reports the held-out corr / MAE of an unconstrained MLP, giving the
**non-linear upper bound** on what any 9-feature mass estimator can
achieve. The gap (MLP_ceiling − OLS_ceiling) quantifies how much of
the achievable signal lives in non-linear feature interactions.

If MLP corr >> OLS corr → features carry rich non-linear signal that
a better-trained encoder could exploit (architecture-bound).
If MLP corr ≈ OLS corr → linear combination already extracts the
signal (feature-bound, not architecture-bound).

**This is NOT a training run for the system.** The MLP is fit directly
on QAR for diagnostic purposes only — its weights are never persisted
or used at inference. Same status as the linear oracle.

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/_validation/test_a_oracle_mlp_ceiling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from test_a_mass_encoder_qar import (  # noqa: E402
    _dedup_qar_files,
    _find_stable_cruise_idx,
    compute_qar_features,
)

_FEATURES_9 = [
    "dist_total_flight",
    "dist_adep_at_t0",
    "cruise_alt_max_flight",
    "wind_long_mean_flight",
    "temp_isa_dev_mean_flight",
    "mach_cruise_planned",
    "climb_rate_mean_climb",
    "accel_mean_climb",
    "time_to_fl240_s",
]


class _OracleMLP(nn.Module):
    """Plain unconstrained 2-layer MLP — only used for diagnostic ceiling."""

    def __init__(self, n_in: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_one_fold(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    epochs: int = 800,
    lr: float = 5e-3,
    weight_decay: float = 1e-3,
    hidden: int = 32,
    seed: int = 0,
) -> np.ndarray:
    """Fit the MLP on (Xtr, ytr) and return predictions on Xte."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_mean = ytr.mean()
    y_std = ytr.std() + 1e-9

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor((ytr - y_mean) / y_std, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    model = _OracleMLP(Xtr.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = loss_fn(pred, ytr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_te = model(Xte_t).numpy()
    return pred_te * y_std + y_mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qar-dir", default="/Users/gabriel/Downloads/QAR3")
    parser.add_argument("--features", nargs="+", default=_FEATURES_9)
    parser.add_argument(
        "--report",
        default="data/models/_comparison/oracle_mlp_ceiling.md",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=800)
    args = parser.parse_args()

    qar_files = _dedup_qar_files(sorted(Path(args.qar_dir).glob("*.parquet")))
    records: list[tuple[str, np.ndarray, float]] = []

    for f in qar_files:
        try:
            df = pl.read_parquet(f)
        except Exception:
            continue
        gw = df["SYS__GW"].to_numpy()
        gw_valid = np.where(np.isfinite(gw) & (gw > 40_000) & (gw < 80_000))[0]
        if len(gw_valid) == 0:
            continue
        cruise_idx = _find_stable_cruise_idx(df)
        seg_candidates = gw_valid[gw_valid >= cruise_idx]
        seg_idx = int(seg_candidates[0]) if len(seg_candidates) > 0 else int(gw_valid[0])
        try:
            features = compute_qar_features(df, seg_t0_idx=seg_idx)
        except ValueError:
            continue
        x = np.array([features[c] for c in args.features], dtype=np.float64)
        records.append((f.name, x, float(gw[seg_idx])))

    if not records:
        print("No records collected", file=sys.stderr)
        return 1

    n = len(records)
    n_feat = len(args.features)
    print(f"Collected n={n} flights, {n_feat} features")

    X = np.stack([r[1] for r in records])
    y = np.array([r[2] for r in records])

    # Z-score features once
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    Xz = ((X - X_mean) / X_std).astype(np.float32)

    # Build deterministic K-folds.
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(n)
    fold_size = n // args.n_folds
    folds = [perm[i * fold_size : (i + 1) * fold_size] for i in range(args.n_folds)]
    folds[-1] = perm[(args.n_folds - 1) * fold_size :]  # absorb remainder

    # K-fold CV averaged over n_seeds for stability
    all_preds = np.zeros((args.n_seeds, n))
    for seed in range(args.n_seeds):
        preds = np.empty(n)
        for fold_idx, te_idx in enumerate(folds):
            tr_mask = np.ones(n, dtype=bool)
            tr_mask[te_idx] = False
            pred_te = _train_one_fold(
                Xz[tr_mask],
                y[tr_mask],
                Xz[te_idx],
                epochs=args.epochs,
                hidden=args.hidden,
                seed=seed * 1000 + fold_idx,
            )
            preds[te_idx] = pred_te
            print(
                f"  seed {seed} fold {fold_idx + 1}/{args.n_folds} "
                f"n_test={len(te_idx)}"
            )
        all_preds[seed] = preds

    mean_preds = all_preds.mean(axis=0)

    corr_per_seed = [float(np.corrcoef(y, all_preds[s])[0, 1]) for s in range(args.n_seeds)]
    mae_per_seed = [float(np.mean(np.abs(all_preds[s] - y))) for s in range(args.n_seeds)]
    corr_mean = float(np.corrcoef(y, mean_preds)[0, 1])
    mae_mean = float(np.mean(np.abs(mean_preds - y)))

    # Also report a full-batch in-sample baseline for context (over-fits intentionally).
    in_sample_pred = _train_one_fold(Xz, y, Xz, epochs=args.epochs, hidden=args.hidden, seed=0)
    corr_in_sample = float(np.corrcoef(y, in_sample_pred)[0, 1])
    mae_in_sample = float(np.mean(np.abs(in_sample_pred - y)))

    lines = [
        f"# Oracle MLP ceiling on the 9 ADS-B features — n={n}\n",
        f"> Validation set: **{n} QAR flights** (held out from any training).",
        f"> Method: 2-layer SiLU MLP ({args.hidden} hidden, {args.epochs} epochs),",
        f"> {args.n_folds}-fold CV averaged over {args.n_seeds} seeds.",
        "",
        "## Headline",
        "",
        "| Estimator | corr | MAE (kg) |",
        "|---|---:|---:|",
        f"| Full-batch MLP (in-sample, over-fit) | {corr_in_sample:.3f} | {mae_in_sample:.0f} |",
        f"| **{args.n_folds}-fold MLP (honest non-linear ceiling)** | **{corr_mean:.3f}** | **{mae_mean:.0f}** |",
        f"| OLS LOOCV (linear ceiling — see oracle_feature_ceiling) | 0.577 | 3760 |",
        f"| best trained encoder — mass_v8_lean_t15 | 0.359 | 4600 |",
        f"| baseline encoder — full_hybrid_v2 | 0.083 | 8698 |",
        "",
        "## Per-seed variability",
        "",
        "| Seed | corr | MAE (kg) |",
        "|---:|---:|---:|",
    ]
    for s in range(args.n_seeds):
        lines.append(f"| {s} | {corr_per_seed[s]:.3f} | {mae_per_seed[s]:.0f} |")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- **MLP > OLS by a margin** → the 9 features carry non-linear "
        "interactions that the linear oracle misses. A better-trained "
        "encoder (one that closes the indirect-supervision gap and uses "
        "non-linear combinations) could reach this higher ceiling.\n"
        "- **MLP ≈ OLS** → the linear oracle already saturates the "
        "feature information content. The remaining gap to corr 1.0 "
        "is feature-bound (need richer features: sequence encoder over "
        "trajectory, self-supervised physics labels, …).\n"
        "- The full-batch in-sample number is given **only** to read "
        "expressivity headroom — it over-fits and should be ignored as "
        "a generalisation estimate."
    )

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")
    print(f"MLP {args.n_folds}-fold corr = {corr_mean:.3f}, MAE = {mae_mean:.0f} kg")
    print(f"OLS LOOCV corr = 0.577 (for comparison)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
