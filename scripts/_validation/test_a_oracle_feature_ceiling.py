"""Oracle linear-regression ceiling on the 9 ADS-B MassEncoder features.

Validation-only diagnostic. Fits a plain ordinary-least-squares regression
``m_truth ~ β·features`` on the 18 QAR validation flights, using the same
9 features the v4-v6 architectures consume. Reports leave-one-flight-out
cross-validated correlation and MAE, then compares against the best
trained MassEncoder.

This is **not** a training script — it never touches the ADS-B Delta
table, and the regression weights are never persisted into a checkpoint.
The output is informative: it tells us whether the corr ceiling we are
approaching is **architectural** (encoder under-uses available signal)
or **fundamental** (no linear combination of these features can do
better, given only 18 flights).

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/_validation/test_a_oracle_feature_ceiling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qar-dir", default="/Users/gabriel/Downloads/QAR3")
    parser.add_argument("--features", nargs="+", default=_FEATURES_9)
    parser.add_argument("--report", default="data/models/_comparison/oracle_feature_ceiling.md")
    args = parser.parse_args()

    qar_files = _dedup_qar_files(sorted(Path(args.qar_dir).glob("*.parquet")))
    records: list[tuple[str, np.ndarray, float]] = []

    for f in qar_files:
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
        x = np.array([features[c] for c in args.features], dtype=np.float64)
        records.append((f.name.split("_")[1] if "_" in f.name else f.stem, x, float(gw[seg_idx])))

    if not records:
        print("No records collected", file=sys.stderr)
        return 1

    n = len(records)
    n_feat = len(args.features)
    X = np.stack([r[1] for r in records])  # (n, n_feat)
    y = np.array([r[2] for r in records])  # (n,)

    # Centre and scale features so the LOOCV regression isn't dominated
    # by features with large absolute values (e.g. distance in metres).
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    Xz = (X - X_mean) / X_std

    # Fit full-batch OLS for reference.
    Xz_aug = np.column_stack([Xz, np.ones(n)])  # add intercept
    beta_full, *_ = np.linalg.lstsq(Xz_aug, y, rcond=None)
    y_pred_full = Xz_aug @ beta_full
    corr_full = float(np.corrcoef(y, y_pred_full)[0, 1])
    mae_full = float(np.mean(np.abs(y_pred_full - y)))

    # Leave-one-out cross-validation — guards against the 18 vs 10
    # over-fitting that pure full-batch OLS would exhibit.
    loo_preds = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta_i, *_ = np.linalg.lstsq(Xz_aug[mask], y[mask], rcond=None)
        loo_preds[i] = Xz_aug[i] @ beta_i

    corr_loo = float(np.corrcoef(y, loo_preds)[0, 1])
    mae_loo = float(np.mean(np.abs(loo_preds - y)))
    rmse_loo = float(np.sqrt(np.mean((loo_preds - y) ** 2)))

    # Top-K feature subsets via greedy forward selection (LOOCV).
    def loocv_corr(idx: list[int]) -> tuple[float, float]:
        Xsub = Xz_aug[:, idx + [n_feat]]  # selected features + intercept
        preds = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            beta, *_ = np.linalg.lstsq(Xsub[mask], y[mask], rcond=None)
            preds[i] = Xsub[i] @ beta
        return (
            float(np.corrcoef(y, preds)[0, 1]),
            float(np.mean(np.abs(preds - y))),
        )

    selected: list[int] = []
    remaining = list(range(n_feat))
    forward_steps: list[tuple[str, float, float]] = []
    while remaining:
        best_corr = -np.inf
        best_idx = remaining[0]
        best_mae = np.inf
        for cand in remaining:
            c, m = loocv_corr(selected + [cand])
            if c > best_corr:
                best_corr = c
                best_idx = cand
                best_mae = m
        selected.append(best_idx)
        remaining.remove(best_idx)
        forward_steps.append((args.features[best_idx], best_corr, best_mae))

    lines = [
        "# Oracle linear-regression ceiling on the 9 ADS-B MassEncoder features\n",
        f"> Validation set: **{n} QAR flights** (held out from any training).",
        "> Method: OLS on z-scored features + intercept; LOOCV avoids 18-vs-10 over-fit.",
        "> Compares to the best trained MassEncoder (v6 corr=0.247, v5 corr=0.215).",
        "",
        "## Headline",
        "",
        "| Estimator | corr | MAE (kg) |",
        "|---|---:|---:|",
        f"| Full-batch OLS (in-sample) | {corr_full:.3f} | {mae_full:.0f} |",
        f"| **LOOCV OLS (honest ceiling)** | **{corr_loo:.3f}** | **{mae_loo:.0f}** |",
        f"| best trained — mass_v6_mlp_monotone | 0.247 | 3741 |",
        f"| best trained — mass_v5_tempered_t3 | 0.215 | 5685 |",
        f"| baseline — full_hybrid_v2 | 0.182 | 9207 |",
        "",
        f"RMSE LOOCV = {rmse_loo:.0f} kg.",
        "",
        "## Greedy forward selection (LOOCV corr & MAE per addition)",
        "",
        "| Step | Feature added | LOOCV corr | LOOCV MAE (kg) |",
        "|---:|---|---:|---:|",
    ]
    for k, (name, c, m) in enumerate(forward_steps, start=1):
        lines.append(f"| {k} | `{name}` | {c:.3f} | {m:.0f} |")

    lines.append("")
    lines.append("## Interpretation key")
    lines.append("")
    lines.append(
        "- LOOCV corr is the **honest ceiling** for any linear combination of "
        "the listed features on this validation set.\n"
        "- If LOOCV corr ≥ 0.5 → the 9 ADS-B features carry enough mass signal "
        "to reach the project target; the bottleneck is **encoder expressivity** "
        "(better encoder needed).\n"
        "- If LOOCV corr ∈ [0.25, 0.5] → there is some headroom but reaching "
        "0.5 with these features alone is uncertain.\n"
        "- If LOOCV corr < 0.25 → the features themselves are the bottleneck; "
        "needs richer features (sequence encoder over trajectory, or "
        "self-supervised physics constraint)."
    )

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    print(f"LOOCV corr = {corr_loo:.3f}, MAE = {mae_loo:.0f} kg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
