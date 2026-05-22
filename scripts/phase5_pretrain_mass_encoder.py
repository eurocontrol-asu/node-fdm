"""Phase 5 Stage 1 — Pre-train mass encoder on QAR 414 train flights.

Trains a `MassEncoderLinearTempered` MLP (same architecture as v14)
on QAR-derived features with `SYS__GW` as supervised target, then
evaluates the corr on the 106-flight test split.

Features (3 causal, computed from QAR data with haversine on TRAJ__LAT_GPS / TRAJ__LON_GPS) :
* `dist_total_flight`     : haversine(first GPS sample, last GPS sample) in nautical miles.
* `dist_adep_at_t0`       : haversine(first GPS sample, t_0 GPS sample) where t_0 = first cruise sample (alt > 9000 m stable).
* `cruise_alt_max_flight` : max altitude reached over flight, in metres.

Target : `SYS__GW` at t_0.

R5-status : the pre-training itself uses QAR (mass target IS QAR
SYS__GW). This is the *explicit* Phase 5 R5 break — used for
demonstration, transferred via weights only to v15.

Outputs :
* `data/models/_pretrained/mass_encoder_v15_qar_pretrained.pt`
* `data/investigations/phase5_qar_joint_supervision/artifacts/stage1_pretrain_report.md`
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn, optim

# Set seed for reproducibility.
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_FDM_SRC = REPO_ROOT / "packages" / "node-fdm" / "src"
if str(NODE_FDM_SRC) not in sys.path:
    sys.path.insert(0, str(NODE_FDM_SRC))

from node_fdm.layers.mass_encoder import MassEncoderLinearTempered  # noqa: E402

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")
SPLIT_DIR = Path("data/qar_splits")
TRAIN_TXT = SPLIT_DIR / "phase5_train.txt"
TEST_TXT = SPLIT_DIR / "phase5_test.txt"

A320_OEW = 42_600.0
A320_MTOW = 78_000.0

# Earth radius in nautical miles.
EARTH_R_NM = 3440.065

FT_PER_M = 3.28084


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return float(2 * EARTH_R_NM * math.asin(math.sqrt(a)))


def extract_flight_data(parquet_path: Path) -> tuple[float, float, float, float] | None:
    """Extract (dist_total, dist_adep_at_t0, cruise_alt_max, m_qar_at_t0) for a flight.

    Returns None if cruise window unavailable.
    """
    needed = {"ALT__STD", "TRAJ__LAT_GPS", "TRAJ__LON_GPS", "SYS__GW"}
    try:
        df = pl.read_parquet(parquet_path, columns=list(needed))
    except Exception:  # noqa: BLE001
        return None
    if not needed.issubset(df.columns):
        return None
    df = df.drop_nulls(["ALT__STD", "TRAJ__LAT_GPS", "TRAJ__LON_GPS", "SYS__GW"])
    if len(df) < 100:
        return None

    alt_m = (df["ALT__STD"] / FT_PER_M).to_numpy()
    lat = df["TRAJ__LAT_GPS"].to_numpy()
    lon = df["TRAJ__LON_GPS"].to_numpy()
    sys_gw = df["SYS__GW"].to_numpy()

    # Filter out non-physical GPS samples (lat=0, lon=0 sentinel).
    valid_gps = (np.abs(lat) > 0.01) & (np.abs(lon) > 0.01) & np.isfinite(lat) & np.isfinite(lon)
    if valid_gps.sum() < 50:
        return None
    lat = lat[valid_gps]
    lon = lon[valid_gps]
    alt_m = alt_m[valid_gps]
    sys_gw = sys_gw[valid_gps]

    # Find first cruise sample : alt > 9000 m AND ascent stopped.
    # Use a simple threshold + check stability.
    cruise_mask = alt_m > 9000.0
    if cruise_mask.sum() < 20:
        return None
    first_cruise_idx = int(np.argmax(cruise_mask))  # first True
    if first_cruise_idx == 0:
        # Already starts in cruise — skip (incomplete data).
        return None

    # Features.
    cruise_alt_max_m = float(np.nanmax(alt_m))

    lat0 = float(lat[0])
    lon0 = float(lon[0])
    lat_t0 = float(lat[first_cruise_idx])
    lon_t0 = float(lon[first_cruise_idx])
    lat_end = float(lat[-1])
    lon_end = float(lon[-1])

    dist_adep_at_t0 = haversine_nm(lat0, lon0, lat_t0, lon_t0)
    # Use total straight-line distance from start to end as proxy for total route.
    # This is an under-estimate of arc length but consistent across flights.
    dist_total_flight = haversine_nm(lat0, lon0, lat_end, lon_end)

    # Filter invalid m_qar.
    m_qar = float(sys_gw[first_cruise_idx])
    if not (A320_OEW * 0.9 < m_qar < A320_MTOW * 1.1):
        return None

    return dist_total_flight, dist_adep_at_t0, cruise_alt_max_m, m_qar


def load_split_dataset(
    split_file: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features + target for all flights in a split.

    Returns (X, y, flight_names).
    """
    filenames = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    print(f"Loading {len(filenames)} flights from {split_file.name}")

    features = []
    targets = []
    used_names = []
    for i, name in enumerate(filenames):
        if i % 100 == 0:
            print(f"  [{i+1}/{len(filenames)}] {name}")
        p = QAR_DIR / name
        if not p.exists():
            continue
        data = extract_flight_data(p)
        if data is None:
            continue
        dt, dadep, cam, m = data
        features.append([dt, dadep, cam])
        targets.append(m)
        used_names.append(name)
    X = np.array(features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    print(f"  Used {len(X)} / {len(filenames)} flights")
    return X, y, used_names


def compute_feature_stats(X: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-feature mean/std for InputNormalizer."""
    cols = ["dist_total_flight", "dist_adep_at_t0", "cruise_alt_max_flight"]
    return {
        col: {"mean": float(X[:, i].mean()), "std": float(X[:, i].std() + 1e-8)}
        for i, col in enumerate(cols)
    }


def train_mass_encoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_stats: dict[str, dict[str, float]],
    epochs: int = 500,
    lr: float = 1e-2,
) -> MassEncoderLinearTempered:
    """Train MassEncoderLinearTempered on (X, y) — matching v14 architecture."""
    # v9 / v14 spec : 3 causal features, signs +, -, - (cf. adsb_hybrid_v9_causal.py).
    encoder = MassEncoderLinearTempered(
        feature_stats=feature_stats,
        feature_cols=["dist_total_flight", "dist_adep_at_t0", "cruise_alt_max_flight"],
        expected_signs=[+1.0, -1.0, -1.0],
        oew_kg=A320_OEW,
        mtow_kg=A320_MTOW,
        b0_init=0.85,
        temperature=1.5,
    )
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_train).to(torch.float32)
    y_t = torch.from_numpy(y_train).to(torch.float64)

    for epoch in range(epochs):
        m_pred = encoder(X_t)
        loss = loss_fn(m_pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                m_np = m_pred.detach().numpy()
                corr = float(np.corrcoef(m_np, y_train)[0, 1])
                print(f"  epoch {epoch+1:4d}  loss={loss.item():.2e}  train_corr={corr:+.4f}")
    return encoder


def evaluate_encoder(
    encoder: MassEncoderLinearTempered,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute corr, bias, MAE on (X, y)."""
    encoder.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(torch.float32)
        m_pred = encoder(X_t).detach().numpy()
    corr = float(np.corrcoef(m_pred, y)[0, 1])
    bias = float(np.median(m_pred / y))
    mae = float(np.mean(np.abs(m_pred - y)))
    rmse = float(np.sqrt(np.mean((m_pred - y) ** 2)))
    abs_rel = np.abs((m_pred - y) / y)
    return {
        "n": len(y),
        "corr": corr,
        "bias": bias,
        "mae_kg": mae,
        "rmse_kg": rmse,
        "median_abs_rel": float(np.median(abs_rel)),
    }


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--out-weights",
        type=Path,
        default=Path("data/models/_pretrained/mass_encoder_v15_qar_pretrained.pt"),
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("data/investigations/phase5_qar_joint_supervision/artifacts/stage1_pretrain_report.md"),
    )
    args = parser.parse_args()

    # Load splits.
    X_train, y_train, names_train = load_split_dataset(TRAIN_TXT)
    print()
    X_test, y_test, names_test = load_split_dataset(TEST_TXT)
    print()

    if len(X_train) < 100 or len(X_test) < 50:
        print("Not enough valid flights — abort")
        return 1

    # Stats fit on train only.
    feature_stats = compute_feature_stats(X_train)
    print(f"Feature stats (train) : {feature_stats}")
    print()

    # Train encoder.
    print(f"Training mass encoder on n={len(X_train)} flights, {args.epochs} epochs ...")
    encoder = train_mass_encoder(X_train, y_train, feature_stats, args.epochs, args.lr)
    print()

    # Evaluate.
    train_stats = evaluate_encoder(encoder, X_train, y_train)
    test_stats = evaluate_encoder(encoder, X_test, y_test)

    print(f"=== Train stats ({train_stats['n']} flights) ===")
    print(f"  corr  : {train_stats['corr']:+.4f}")
    print(f"  bias  : {train_stats['bias']:.4f}")
    print(f"  MAE   : {train_stats['mae_kg']:.1f} kg")
    print(f"  RMSE  : {train_stats['rmse_kg']:.1f} kg")
    print(f"  med%  : {train_stats['median_abs_rel']*100:.2f} %")
    print()
    print(f"=== Test stats ({test_stats['n']} flights) ===")
    print(f"  corr  : {test_stats['corr']:+.4f}")
    print(f"  bias  : {test_stats['bias']:.4f}")
    print(f"  MAE   : {test_stats['mae_kg']:.1f} kg")
    print(f"  RMSE  : {test_stats['rmse_kg']:.1f} kg")
    print(f"  med%  : {test_stats['median_abs_rel']*100:.2f} %")
    print()

    # v14 baseline on QAR (Phase 1.8a Exp 1 + 4 reported test corr ~ +0.337 *overall* on per-segment).
    # Our Stage 1 here is *per-flight* (1 sample per flight), so comparable to between-flight corr.
    print("=== Baseline (v14 mass encoder, per-segment Test A on QAR) ===")
    print("  Phase 1.8a Exp 1 per-segment within +0.761, between +0.276, overall +0.337")
    print("  Stage 1 is per-flight (one prediction per flight) → comparable to between-flight corr")

    # Save weights.
    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": encoder.state_dict(),
        "feature_stats": feature_stats,
        "feature_cols": encoder.feature_cols,
        "expected_signs": encoder.signs.tolist(),
        "b0_init": float(encoder.b0.item()),
        "temperature": float(encoder.temperature.item()),
        "oew_kg": A320_OEW,
        "mtow_kg": A320_MTOW,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "trained_on": "qar_phase5_train_414_flights",
    }, args.out_weights)
    print(f"\nWeights saved : {args.out_weights}")

    # Markdown report.
    lines = [
        "# Phase 5 Stage 1 — QAR pre-training mass encoder",
        "",
        f"> Trained on {train_stats['n']} QAR Air France flights (train split, 2018).",
        f"> Evaluated on {test_stats['n']} held-out QAR flights (test split).",
        "> R5-status : QAR is the *training signal*. Weights transferable to v14 architecture (Stage 2).",
        "",
        "## Architecture",
        "",
        "- `MassEncoderLinearTempered` (identical to v14, temperature=1.5).",
        "- 3 causal features : `dist_total_flight`, `dist_adep_at_t0`, `cruise_alt_max_flight` (signs +/-/-).",
        "- Bounds : OEW=42 600 kg, MTOW=78 000 kg.",
        "- Optimizer : Adam, lr=1e-2, 500 epochs.",
        "",
        "## Feature extraction (QAR-side)",
        "",
        "- `dist_total_flight` : haversine(first GPS, last GPS) nautical miles.",
        "- `dist_adep_at_t0` : haversine(first GPS, t_0 GPS where t_0 = first sample alt > 9000 m).",
        "- `cruise_alt_max_flight` : max altitude over flight (metres).",
        "- target `m_qar` : SYS__GW at t_0 (kg).",
        "",
        "## Results",
        "",
        "| Metric | Train | Test |",
        "|---|---:|---:|",
        f"| n_flights | {train_stats['n']} | {test_stats['n']} |",
        f"| **corr Pearson** | **{train_stats['corr']:+.4f}** | **{test_stats['corr']:+.4f}** |",
        f"| bias | {train_stats['bias']:.4f} | {test_stats['bias']:.4f} |",
        f"| MAE (kg) | {train_stats['mae_kg']:.1f} | {test_stats['mae_kg']:.1f} |",
        f"| RMSE (kg) | {train_stats['rmse_kg']:.1f} | {test_stats['rmse_kg']:.1f} |",
        f"| median \\|Δm\\|/m_qar | {train_stats['median_abs_rel']*100:.2f} % | {test_stats['median_abs_rel']*100:.2f} % |",
        "",
        "## Verdict",
        "",
    ]
    if test_stats["corr"] >= 0.55:
        lines.append("✅ **Stage 1 PASS — P2 broken by QAR supervision** (test corr ≥ +0.55).")
        verdict = "PASS"
    elif test_stats["corr"] >= 0.40:
        lines.append("⚠️ **Stage 1 partial — improvement above baseline but below target** (+0.40 ≤ corr < +0.55).")
        verdict = "partial"
    else:
        lines.append("❌ **Stage 1 fail — QAR pre-training does not break P2** (corr < +0.40 on test).")
        verdict = "FAIL"
    lines.append("")
    lines.append("**Baselines for comparison** :")
    lines.append("- v14 mass encoder per-segment on QAR : within +0.761, **overall +0.337** (Phase 1.8a Exp 4 baseline).")
    lines.append("- Phase 1.8a P&S inversion ADS-B-only : max +0.143 (Strategy C v2).")
    lines.append("")
    lines.append("## Compliance")
    lines.append("")
    lines.append("- R1 (no QAR in v14 training) : ✅ preserved (this script does *standalone* QAR training, not v14).")
    lines.append("- R3 (train/test segregation) : ✅ test set untouched during training.")
    lines.append("- R11 (gate before fine-tuning) : ")
    lines.append(f"  - corr ≥ +0.55 → proceed to Stage 2 (transfer + ADS-B fine-tune).")
    lines.append(f"  - +0.40 ≤ corr < +0.55 → procced cautiously, Stage 2 may marginally improve.")
    lines.append(f"  - corr < +0.40 → escalate, P2 even with QAR-direct supervision is hard.")

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(lines))
    print(f"Report saved : {args.out_report}")
    print(f"\nVerdict : {verdict}")

    return 0 if test_stats["corr"] >= 0.40 else 1


if __name__ == "__main__":
    sys.exit(main())
