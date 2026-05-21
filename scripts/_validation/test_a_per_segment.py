"""Per-segment Test A — validation contract aligned with training.

The trained MassEncoder is invoked **once per 60-second segment** during
training, predicting ``m_0`` at the segment's first timestep. The
existing Test A evaluates only at a single fixed point per flight (first
stable-cruise sample), so it never measures the encoder's intra-flight
behaviour — does it correctly track the ~2-3 t fuel burn between TOC
and TOD, or does it return the same value for every segment?

This script answers that by:

1. Sliding a (seq_len=60s, shift=60s) window across every QAR flight,
   mirroring the training loader.
2. At each segment start ``t_0``, computing the flight features the
   encoder consumes (with ``dist_adep_at_t0`` varying segment-by-segment).
3. Reading the ground-truth ``SYS__GW(t_0)`` at the same point.
4. Comparing per-segment predictions to per-segment truth (Pearson corr
   + MAE on the pooled dataset and decomposed between/within flight).

Also runs an OLS and a small MLP oracle on the same per-segment pool so
the indirect-supervision gap can be measured on the same contract the
trainer actually optimises against.

Validation-only, never touches the ADS-B Delta table; the MLP oracle is
fit purely diagnostically.

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/_validation/test_a_per_segment.py \\
        --models mass_v8_lean_t15 mass_v5_tempered_t3 full_hybrid_v2 \\
        --seq-len 60 --shift 60
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from test_a_mass_encoder_qar import (  # noqa: E402
    _R_EARTH_M,
    _FT_TO_M,
    _MACH_CRUISE_FALLBACK,
    _MACH_CRUISE_MAX,
    _MACH_CRUISE_MIN,
    _dedup_qar_files,
    load_mass_encoder,
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

_CLIMB_BAND_LOW_M = 1500.0
_CLIMB_BAND_HIGH_M = 4500.0
_FL240_M = 7300.0
_GROUND_OFFSET_M = 1000.0
_QAR_NOMINAL_DT_S = 1.0
_KT_TO_MS = 0.514444


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _R_EARTH_M * math.asin(math.sqrt(a))


def _flight_aggregates(df: pl.DataFrame) -> dict[str, float]:
    """Compute the per-flight (constant) aggregates: cruise_alt, wind, temp, mach, dynamics.

    These are look-ahead aggregates — they match what the training loader
    computes and broadcasts to every segment of the flight. This script
    uses them as-is to mirror the trained encoder's input distribution.
    """
    alt_ft = df["ALT__STD"].to_numpy().astype(np.float64)
    alt_m = alt_ft * _FT_TO_M
    cruise_alt_max_m = float(np.nanmax(alt_m))
    wind_long_mean = float(np.nanmean(df["WIND__LONG"].to_numpy()))
    temp_isa_dev_mean = float(np.nanmean(df["TEMP__DELTA_ISA"].to_numpy()))

    mach_sel = df["SPD__MACH_SEL"].to_numpy()
    in_range = np.isfinite(mach_sel) & (mach_sel >= _MACH_CRUISE_MIN) & (mach_sel <= _MACH_CRUISE_MAX)
    mach_cruise = (
        float(np.max(mach_sel[in_range])) if in_range.any() else _MACH_CRUISE_FALLBACK
    )

    band_mask = (
        np.isfinite(alt_m) & (alt_m >= _CLIMB_BAND_LOW_M) & (alt_m <= _CLIMB_BAND_HIGH_M)
    )
    if "ATT__VV" in df.columns and band_mask.any():
        vv = df["ATT__VV"].to_numpy().astype(np.float64)
        m = band_mask & np.isfinite(vv)
        climb_rate = float(np.nanmean(vv[m])) if m.any() else 0.0
    else:
        climb_rate = 0.0

    if "SPD__TAS" in df.columns:
        tas_ms = df["SPD__TAS"].to_numpy().astype(np.float64) * _KT_TO_MS
        finite = np.isfinite(tas_ms)
        accel = 0.0
        if finite.sum() >= 2:
            accel_arr = np.gradient(tas_ms) / _QAR_NOMINAL_DT_S
            m = band_mask & np.isfinite(accel_arr)
            if m.any():
                accel = float(np.nanmean(accel_arr[m]))
    else:
        accel = 0.0

    above_ground = np.isfinite(alt_m) & (alt_m >= _GROUND_OFFSET_M)
    above_fl240 = np.isfinite(alt_m) & (alt_m >= _FL240_M)
    if above_ground.any() and above_fl240.any():
        idx_start = int(np.argmax(above_ground))
        idx_top = int(np.argmax(above_fl240))
        dt = float(max(idx_top - idx_start, 0)) * _QAR_NOMINAL_DT_S
    else:
        dt = 0.0

    return {
        "cruise_alt_max_flight": cruise_alt_max_m,
        "wind_long_mean_flight": wind_long_mean,
        "temp_isa_dev_mean_flight": temp_isa_dev_mean,
        "mach_cruise_planned": mach_cruise,
        "climb_rate_mean_climb": climb_rate,
        "accel_mean_climb": accel,
        "time_to_fl240_s": dt,
    }


def _build_segments(
    df: pl.DataFrame,
    aggregates: dict[str, float],
    *,
    seq_len: int,
    shift: int,
    min_alt_m: float = 500.0,
) -> list[tuple[int, dict[str, float], float]]:
    """Slide (seq_len, shift) windows and emit (t_idx, features, m_truth)."""
    lat = df["NAV__LAT"].to_numpy()
    lon = df["NAV__LONG"].to_numpy()
    alt_ft = df["ALT__STD"].to_numpy().astype(np.float64)
    alt_m = alt_ft * _FT_TO_M
    gw = df["SYS__GW"].to_numpy()

    valid_pos = np.isfinite(lat) & np.isfinite(lon) & (lat != 0.0) & (lon != 0.0)
    if not valid_pos.any():
        return []
    first_valid = int(np.argmax(valid_pos))
    last_valid = int(np.where(valid_pos)[0][-1])
    lat0, lon0 = float(lat[first_valid]), float(lon[first_valid])
    lat_end, lon_end = float(lat[last_valid]), float(lon[last_valid])

    n = df.shape[0]
    segments: list[tuple[int, dict[str, float], float]] = []
    for start in range(0, n - seq_len + 1, shift):
        if not valid_pos[start]:
            continue
        if not np.isfinite(alt_m[start]) or alt_m[start] < min_alt_m:
            continue
        m_truth = float(gw[start])
        if not (np.isfinite(m_truth) and 40_000 < m_truth < 80_000):
            continue

        lat_t = float(lat[start])
        lon_t = float(lon[start])
        adep_at_t0 = _haversine_m(lat0, lon0, lat_t, lon_t)
        ades_at_t0 = _haversine_m(lat_t, lon_t, lat_end, lon_end)
        feats = {
            "dist_total_flight": adep_at_t0 + ades_at_t0,
            "dist_adep_at_t0": adep_at_t0,
            **aggregates,
        }
        segments.append((start, feats, m_truth))
    return segments


class _OracleMLP(nn.Module):
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


def _flight_aware_kfold_mlp(
    X: np.ndarray,
    y: np.ndarray,
    flight_ids: np.ndarray,
    *,
    n_folds: int,
    n_seeds: int,
    hidden: int,
    epochs: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[float, float]:
    """K-fold CV with **flight-level splits** so a flight's segments stay together.

    Robust against the ~40k-segment per-segment pool: mini-batch SGD with
    gradient clipping and feature-NaN cleaning (rare ``temp_isa_dev`` /
    ``mach_cruise_planned`` fallbacks can leave NaN that contaminates the
    normaliser otherwise).
    """
    X = X.copy()
    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not finite_mask.all():
        X = X[finite_mask]
        y = y[finite_mask]
        flight_ids = flight_ids[finite_mask]
    # Z-score X once with robust feature-wise scaling
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    Xz = ((X - X_mean) / X_std).astype(np.float32)

    unique_flights = np.array(sorted(set(flight_ids.tolist())))
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(unique_flights)
    folds = np.array_split(perm, n_folds)

    all_preds = np.zeros((n_seeds, len(y)))
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        preds = np.empty(len(y))
        for fold_idx, te_flights in enumerate(folds):
            te_mask = np.isin(flight_ids, te_flights)
            tr_mask = ~te_mask
            ytr = y[tr_mask]
            y_mean = float(ytr.mean())
            y_std = float(ytr.std()) + 1e-6
            Xtr_t = torch.tensor(Xz[tr_mask], dtype=torch.float32)
            ytr_t = torch.tensor((ytr - y_mean) / y_std, dtype=torch.float32)
            Xte_t = torch.tensor(Xz[te_mask], dtype=torch.float32)

            model = _OracleMLP(Xz.shape[1], hidden=hidden)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = nn.SmoothL1Loss()
            n_tr = len(ytr)
            for _ in range(epochs):
                idx = torch.randperm(n_tr)
                for s in range(0, n_tr, batch_size):
                    sub = idx[s : s + batch_size]
                    opt.zero_grad()
                    loss = loss_fn(model(Xtr_t[sub]), ytr_t[sub])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    opt.step()
            model.eval()
            with torch.no_grad():
                preds[te_mask] = model(Xte_t).numpy() * y_std + y_mean
        all_preds[seed] = preds

    mean_preds = all_preds.mean(axis=0)
    return (
        float(np.corrcoef(y, mean_preds)[0, 1]),
        float(np.mean(np.abs(mean_preds - y))),
    )


def _flight_aware_loocv_ols(
    X: np.ndarray,
    y: np.ndarray,
    flight_ids: np.ndarray,
) -> tuple[float, float]:
    """Leave-one-flight-out OLS regression on z-scored features."""
    unique_flights = sorted(set(flight_ids.tolist()))
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    Xz_aug = np.column_stack([Xz, np.ones(len(y))])
    preds = np.empty(len(y))
    for fid in unique_flights:
        te_mask = flight_ids == fid
        tr_mask = ~te_mask
        beta, *_ = np.linalg.lstsq(Xz_aug[tr_mask], y[tr_mask], rcond=None)
        preds[te_mask] = Xz_aug[te_mask] @ beta
    return (
        float(np.corrcoef(y, preds)[0, 1]),
        float(np.mean(np.abs(preds - y))),
    )


def _decompose_between_within(
    y: np.ndarray, preds: np.ndarray, flight_ids: np.ndarray
) -> dict[str, float]:
    """Between-flight vs within-flight correlation decomposition."""
    unique = sorted(set(flight_ids.tolist()))
    y_means = np.zeros_like(y, dtype=np.float64)
    p_means = np.zeros_like(preds, dtype=np.float64)
    n_per_flight: list[int] = []
    for fid in unique:
        mask = flight_ids == fid
        y_means[mask] = y[mask].mean()
        p_means[mask] = preds[mask].mean()
        n_per_flight.append(int(mask.sum()))

    between_corr = float(np.corrcoef(y_means, p_means)[0, 1])
    multi_mask = np.array(
        [n_per_flight[unique.index(fid)] >= 2 for fid in flight_ids], dtype=bool
    )
    within_corr = float("nan")
    if multi_mask.any():
        y_within = (y - y_means)[multi_mask]
        p_within = (preds - p_means)[multi_mask]
        if np.std(y_within) > 1e-6 and np.std(p_within) > 1e-6:
            within_corr = float(np.corrcoef(y_within, p_within)[0, 1])
    return {
        "between_flight_corr": between_corr,
        "within_flight_corr": within_corr,
        "n_segments_total": int(len(y)),
        "n_flights_with_multi_segments": int(sum(1 for n in n_per_flight if n >= 2)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qar-dir", default="/Users/gabriel/Downloads/QAR3")
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mass_v8_lean_t15", "mass_v5_tempered_t3", "full_hybrid_v2"],
    )
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--shift", type=int, default=60)
    parser.add_argument(
        "--report", default="data/models/_comparison/test_a_per_segment.md"
    )
    parser.add_argument("--mlp-folds", type=int, default=5)
    parser.add_argument("--mlp-seeds", type=int, default=3)
    parser.add_argument("--mlp-epochs", type=int, default=400)
    parser.add_argument("--mlp-hidden", type=int, default=32)
    args = parser.parse_args()

    qar_files = _dedup_qar_files(sorted(Path(args.qar_dir).glob("*.parquet")))
    print(f"Found {len(qar_files)} unique QAR files")

    flight_ids: list[str] = []
    segments_t: list[int] = []
    features_list: list[np.ndarray] = []
    truths: list[float] = []

    for f in qar_files:
        try:
            df = pl.read_parquet(f)
        except Exception:
            continue
        agg = _flight_aggregates(df)
        flight_id = f.name.split("_")[1] if "_" in f.name else f.stem
        # add file stem as suffix to keep flights unique even when flight_id
        # collides across days
        unique_fid = f"{flight_id}__{f.stem}"
        segs = _build_segments(df, agg, seq_len=args.seq_len, shift=args.shift)
        for t_idx, feats, m_truth in segs:
            flight_ids.append(unique_fid)
            segments_t.append(t_idx)
            features_list.append(np.array([feats[c] for c in _FEATURES_9], dtype=np.float64))
            truths.append(m_truth)

    if not features_list:
        print("No segments collected", file=sys.stderr)
        return 1

    X = np.stack(features_list)
    y = np.array(truths)
    flight_arr = np.array(flight_ids)
    n_seg = len(y)
    n_flights = len(set(flight_ids))
    avg_per_flight = n_seg / n_flights
    print(f"Collected {n_seg} segments across {n_flights} flights (avg {avg_per_flight:.1f} segs/flight)")

    # ------------------------------------------------------------------
    # Trained model predictions per segment
    # ------------------------------------------------------------------
    encoders = {
        name: load_mass_encoder(Path(args.models_dir) / name) for name in args.models
    }

    model_metrics: dict[str, dict[str, float]] = {}
    feature_names = _FEATURES_9
    for name in args.models:
        enc, feature_cols = encoders[name]
        idx = [feature_names.index(c) for c in feature_cols]
        with torch.no_grad():
            x_t = torch.tensor(X[:, idx], dtype=torch.float32)
            preds = enc(x_t).numpy()
        err = preds - y
        corr = float(np.corrcoef(y, preds)[0, 1])
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        bias = float(err.mean())
        bias_pct = float(np.mean(err / y) * 100)
        decomp = _decompose_between_within(y, preds, flight_arr)
        model_metrics[name] = {
            "corr": corr,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "bias_pct": bias_pct,
            **decomp,
            "pred_min": float(preds.min()),
            "pred_max": float(preds.max()),
            "pred_intra_flight_std_mean": float(
                np.mean(
                    [
                        preds[flight_arr == fid].std()
                        for fid in set(flight_ids)
                        if (flight_arr == fid).sum() >= 2
                    ]
                )
            ),
        }
        print(
            f"  {name:30s} corr={corr:+.3f}  MAE={mae:.0f}  "
            f"between={decomp['between_flight_corr']:+.3f}  within={decomp['within_flight_corr']:+.3f}  "
            f"intra-flight pred std={model_metrics[name]['pred_intra_flight_std_mean']:.0f} kg"
        )

    # ------------------------------------------------------------------
    # Oracles on the same per-segment pool (flight-aware splits)
    # ------------------------------------------------------------------
    print("\nFitting OLS oracle (flight-aware LOOCV) ...")
    ols_corr, ols_mae = _flight_aware_loocv_ols(X, y, flight_arr)
    print(f"  OLS corr={ols_corr:+.3f}  MAE={ols_mae:.0f}")

    print(f"Fitting MLP oracle ({args.mlp_folds}-fold flight-aware × {args.mlp_seeds} seeds) ...")
    mlp_corr, mlp_mae = _flight_aware_kfold_mlp(
        X,
        y,
        flight_arr,
        n_folds=args.mlp_folds,
        n_seeds=args.mlp_seeds,
        hidden=args.mlp_hidden,
        epochs=args.mlp_epochs,
    )
    print(f"  MLP corr={mlp_corr:+.3f}  MAE={mlp_mae:.0f}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    lines = [
        f"# Test A per-segment — n={n_seg} segments across {n_flights} flights",
        "",
        f"> seq_len={args.seq_len}s, shift={args.shift}s (mirrors training loader).",
        f"> Avg {avg_per_flight:.1f} segments/flight.",
        "> Ground truth = `SYS__GW` at the segment start.",
        "> Oracles use **flight-aware** splits (entire flight held out per fold).",
        "",
        "## Per-segment correlation",
        "",
        "| Model | corr (pooled) | MAE (kg) | bias (kg) | between-flight corr | within-flight corr | intra-flight pred std (kg) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in model_metrics.items():
        wcorr_str = (
            f"{m['within_flight_corr']:.3f}"
            if not (isinstance(m["within_flight_corr"], float) and math.isnan(m["within_flight_corr"]))
            else "n/a"
        )
        lines.append(
            f"| `{name}` | {m['corr']:+.3f} | {m['mae']:.0f} | {m['bias']:+.0f} | "
            f"{m['between_flight_corr']:+.3f} | {wcorr_str} | "
            f"{m['pred_intra_flight_std_mean']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Oracles (flight-aware CV on the same per-segment pool)",
            "",
            "| Estimator | corr | MAE (kg) | Comment |",
            "|---|---:|---:|---|",
            f"| OLS LOOCV (leave-one-flight-out, linear) | **{ols_corr:+.3f}** | **{ols_mae:.0f}** | linear ceiling for these 9 features |",
            f"| MLP {args.mlp_folds}-fold (leave-flights-out, non-lin) | **{mlp_corr:+.3f}** | **{mlp_mae:.0f}** | non-linear ceiling |",
            "",
            "## Reading guide",
            "",
            "- **corr (pooled)** = Pearson on all segments concatenated. The headline number — what a downstream consumer would see if it queried the encoder at random points.",
            "- **between-flight corr** = correlation of flight-mean prediction vs flight-mean truth. Measures route-typical signal.",
            "- **within-flight corr** = correlation of (segment − flight_mean) prediction vs (segment − flight_mean) truth. Measures the encoder's ability to track mass *evolution within a flight* (mainly fuel burn).",
            "- **intra-flight pred std** = average across flights of the std of the encoder's predictions across that flight's segments. **If this is near zero, the encoder is blind to intra-flight variation** (constant prediction = it has no causal segment-varying feature).",
            "- The OLS / MLP oracle numbers use **leave-one-flight-out** splits so a flight's segments never leak between train and test.",
        ]
    )

    out_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
