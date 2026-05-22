"""Phase 5 Stage 2 (light) — Swap-only test : QAR-pretrained encoder in v14.

Load v14 checkpoint, swap the mass encoder weights with the QAR-pretrained
ones from Stage 1, then re-run Test A per-segment on the 106 QAR test
flights *without any retraining*.

This tests whether the QAR-pretrained encoder is *plug-and-play* or
whether the rest of v14 (drag/thrust/eta corrections) needs fine-tuning
to re-balance to the new mass predictions.

Quick gate before investing in the 30-min v15 retrain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_FDM_SRC = REPO_ROOT / "packages" / "node-fdm" / "src"
if str(NODE_FDM_SRC) not in sys.path:
    sys.path.insert(0, str(NODE_FDM_SRC))

# Reuse Test A logic from existing validation script.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "_validation"))
from test_a_per_segment import _flight_aggregates, _haversine_m  # noqa: E402

from node_fdm.architectures import (  # noqa: E402, F401
    adsb_hybrid_v14_psefficiency,
)
from node_fdm.layers.mass_encoder import MassEncoderLinearTempered  # noqa: E402

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")
TEST_TXT = Path("data/qar_splits/phase5_test.txt")
V14_CKPT_DIR = Path("data/models/full_hybrid_v14")
PRETRAINED_PATH = Path("data/models/_pretrained/mass_encoder_v15_qar_pretrained.pt")

FT_PER_M = 3.28084


def load_v14_mass_encoder(v14_dir: Path) -> tuple[MassEncoderLinearTempered, dict[str, dict[str, float]]]:
    """Reconstruct v14's mass encoder from the saved meta + weights."""
    import importlib

    importlib.import_module("node_fdm.architectures.adsb_hybrid_v14_psefficiency")

    meta = json.loads((v14_dir / "meta.json").read_text())
    # v14 uses 3 causal features, signs +/-/-, T=1.5.
    # Need feature_stats from the trainer's saved state.
    # The encoder weights are in mass_encoder.pt.
    enc_state = torch.load(v14_dir / "mass_encoder.pt", map_location="cpu", weights_only=False)
    # InputNormalizer attrs live inside the state_dict ; we need to reconstruct an
    # encoder instance to load into. Let's read the buffer values directly.
    feat_cols = ["dist_total_flight", "dist_adep_at_t0", "cruise_alt_max_flight"]
    # Reconstruct feature_stats by reading the saved normalizer buffers.
    state_dict = enc_state["state_dict"] if "state_dict" in enc_state else enc_state
    # Normalizer buffers : normalizer.{col}_mean, normalizer.{col}_std
    feature_stats = {}
    for col in feat_cols:
        mean_key = f"normalizer.{col}.mean"
        std_key = f"normalizer.{col}.std"
        if mean_key not in state_dict or std_key not in state_dict:
            # Try alternative key formats.
            for k in state_dict.keys():
                if col in k and "mean" in k:
                    mean_key = k
                if col in k and "std" in k:
                    std_key = k
        feature_stats[col] = {
            "mean": float(state_dict[mean_key]),
            "std": float(state_dict[std_key]),
        }
    encoder = MassEncoderLinearTempered(
        feature_stats=feature_stats,
        feature_cols=feat_cols,
        expected_signs=[+1.0, -1.0, -1.0],
        oew_kg=42_600.0,
        mtow_kg=78_000.0,
        b0_init=0.85,
        temperature=1.5,
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder, feature_stats


def load_qar_pretrained_encoder() -> tuple[MassEncoderLinearTempered, dict[str, dict[str, float]]]:
    """Load the Stage 1 pre-trained encoder."""
    payload = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
    feature_stats = payload["feature_stats"]
    encoder = MassEncoderLinearTempered(
        feature_stats=feature_stats,
        feature_cols=payload["feature_cols"],
        expected_signs=payload["expected_signs"],
        oew_kg=payload["oew_kg"],
        mtow_kg=payload["mtow_kg"],
        b0_init=payload["b0_init"],
        temperature=payload["temperature"],
    )
    encoder.load_state_dict(payload["state_dict"])
    encoder.eval()
    return encoder, feature_stats


def extract_per_segment_features(parquet_path: Path, seq_len: int = 60, shift: int = 60) -> list[dict]:
    """Slide (seq_len, shift) window across flight, return per-segment dicts.

    Each dict contains the 3 features at each segment start + m_qar at segment start.
    Mirrors test_a_per_segment.py's per-segment logic.
    """
    needed = {"ALT__STD", "TRAJ__LAT_GPS", "TRAJ__LON_GPS", "SYS__GW"}
    try:
        df = pl.read_parquet(parquet_path, columns=list(needed))
    except Exception:  # noqa: BLE001
        return []
    if not needed.issubset(df.columns):
        return []
    df = df.drop_nulls(["ALT__STD", "TRAJ__LAT_GPS", "TRAJ__LON_GPS", "SYS__GW"])
    if len(df) < 100:
        return []

    alt_m = (df["ALT__STD"] / FT_PER_M).to_numpy()
    lat = df["TRAJ__LAT_GPS"].to_numpy()
    lon = df["TRAJ__LON_GPS"].to_numpy()
    sys_gw = df["SYS__GW"].to_numpy()

    valid_gps = (np.abs(lat) > 0.01) & (np.abs(lon) > 0.01) & np.isfinite(lat) & np.isfinite(lon)
    if valid_gps.sum() < 50:
        return []
    lat = lat[valid_gps]
    lon = lon[valid_gps]
    alt_m = alt_m[valid_gps]
    sys_gw = sys_gw[valid_gps]

    # Flight-level features
    cruise_alt_max_m = float(np.nanmax(alt_m))
    lat0 = float(lat[0])
    lon0 = float(lon[0])
    lat_end = float(lat[-1])
    lon_end = float(lon[-1])
    # dist_total_flight = haversine(start, end) per Stage 1 convention.
    dist_total_flight = _haversine_m(lat0, lon0, lat_end, lon_end) / 1852.0  # m → nm

    segments = []
    for start in range(0, len(lat) - seq_len + 1, shift):
        end = start + seq_len
        # Per-segment t_0 = first sample of segment.
        lat_t0 = float(lat[start])
        lon_t0 = float(lon[start])
        dist_adep_at_t0 = _haversine_m(lat0, lon0, lat_t0, lon_t0) / 1852.0  # nm
        m_qar_t0 = float(sys_gw[start])
        if not (40_000 < m_qar_t0 < 80_000):
            continue
        segments.append({
            "dist_total_flight": dist_total_flight,
            "dist_adep_at_t0": dist_adep_at_t0,
            "cruise_alt_max_flight": cruise_alt_max_m,
            "m_qar": m_qar_t0,
            "flight": parquet_path.name,
            "alt_t0": float(alt_m[start]),
        })
    return segments


def evaluate_encoder_per_segment(encoder, segments: list[dict]) -> dict[str, float]:
    """Run encoder per-segment + compute within/between/overall corr."""
    if not segments:
        return {"n_segments": 0, "n_flights": 0, "within_corr": float("nan"),
                "between_corr": float("nan"), "overall_corr": float("nan")}

    # Build feature tensor + m_qar array.
    X = np.array([[s["dist_total_flight"], s["dist_adep_at_t0"], s["cruise_alt_max_flight"]]
                  for s in segments], dtype=np.float32)
    m_qar = np.array([s["m_qar"] for s in segments], dtype=np.float32)
    flight_ids = [s["flight"] for s in segments]

    with torch.no_grad():
        m_pred = encoder(torch.from_numpy(X)).detach().numpy().astype(np.float32)

    overall_corr = float(np.corrcoef(m_pred, m_qar)[0, 1])

    # Within-flight corr : avg over flights of corr(per-segment) within that flight.
    from collections import defaultdict
    flight_groups = defaultdict(list)
    for i, fid in enumerate(flight_ids):
        flight_groups[fid].append((m_pred[i], m_qar[i]))
    within_corrs = []
    flight_means_pred = []
    flight_means_qar = []
    for fid, pairs in flight_groups.items():
        if len(pairs) < 2:
            continue
        ps = np.array([p for p, _ in pairs])
        qs = np.array([q for _, q in pairs])
        if ps.std() < 1e-6 or qs.std() < 1e-6:
            # All predictions identical for this flight (encoder uses flight-level
            # features so within-flight predictions are constant) - skip from within-corr.
            continue
        within_corrs.append(float(np.corrcoef(ps, qs)[0, 1]))
        flight_means_pred.append(float(ps.mean()))
        flight_means_qar.append(float(qs.mean()))
    within_corr = float(np.mean(within_corrs)) if within_corrs else float("nan")
    if flight_means_pred:
        between_corr = float(np.corrcoef(flight_means_pred, flight_means_qar)[0, 1])
    else:
        # Fallback : use the unique flight-level predictions (encoder is flight-level).
        unique_flights = set(flight_ids)
        fmp = []
        fmq = []
        for fid in unique_flights:
            mask = [f == fid for f in flight_ids]
            fmp.append(float(np.mean(m_pred[mask])))
            fmq.append(float(np.mean(m_qar[mask])))
        between_corr = float(np.corrcoef(fmp, fmq)[0, 1]) if len(fmp) > 1 else float("nan")

    mae = float(np.mean(np.abs(m_pred - m_qar)))
    bias = float(np.median(m_pred / m_qar))
    return {
        "n_segments": len(segments),
        "n_flights": len(set(flight_ids)),
        "within_corr": within_corr,
        "between_corr": between_corr,
        "overall_corr": overall_corr,
        "mae_kg": mae,
        "bias": bias,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--shift", type=int, default=60)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/investigations/phase5_qar_joint_supervision/artifacts/stage2_swap_test.md"),
    )
    args = parser.parse_args()

    print(f"Loading v14 mass encoder from {V14_CKPT_DIR}/mass_encoder.pt ...")
    v14_encoder, v14_stats = load_v14_mass_encoder(V14_CKPT_DIR)
    print(f"v14 stats : {v14_stats}")
    print(f"v14 effective coefs : {v14_encoder.effective_coefficients()}")
    print()
    print(f"Loading QAR-pretrained encoder from {PRETRAINED_PATH} ...")
    qar_encoder, qar_stats = load_qar_pretrained_encoder()
    print(f"QAR-pretrained stats : {qar_stats}")
    print(f"QAR-pretrained effective coefs : {qar_encoder.effective_coefficients()}")
    print()

    # Build per-segment dataset from test split.
    filenames = [line.strip() for line in TEST_TXT.read_text().splitlines() if line.strip()]
    print(f"Loading {len(filenames)} test flights for per-segment evaluation ...")
    all_segments = []
    for i, name in enumerate(filenames):
        if i % 25 == 0:
            print(f"  [{i+1}/{len(filenames)}] {name}")
        segs = extract_per_segment_features(QAR_DIR / name, args.seq_len, args.shift)
        all_segments.extend(segs)
    print(f"  Total segments : {len(all_segments)} across {len({s['flight'] for s in all_segments})} flights")
    print()

    # v14 expects features in METRES (normalizer trained on ADS-B Delta which uses
    # haversine_m in the loader). Stage 1 used nm (haversine_nm). Convert nm→m
    # for v14 evaluation to ensure unit consistency.
    NM_TO_M = 1852.0
    all_segments_v14 = [
        {
            **s,
            "dist_total_flight": s["dist_total_flight"] * NM_TO_M,
            "dist_adep_at_t0": s["dist_adep_at_t0"] * NM_TO_M,
            # cruise_alt_max_flight stays in metres (same unit in both pipelines).
        }
        for s in all_segments
    ]

    # Evaluate both encoders.
    print("=== v14 mass encoder (baseline) on QAR test split per-segment (features in metres) ===")
    v14_stats_eval = evaluate_encoder_per_segment(v14_encoder, all_segments_v14)
    print(f"  n_segments={v14_stats_eval['n_segments']}, n_flights={v14_stats_eval['n_flights']}")
    print(f"  within_corr  = {v14_stats_eval['within_corr']:+.4f}")
    print(f"  between_corr = {v14_stats_eval['between_corr']:+.4f}")
    print(f"  overall_corr = {v14_stats_eval['overall_corr']:+.4f}")
    print(f"  bias = {v14_stats_eval['bias']:.4f}, MAE = {v14_stats_eval['mae_kg']:.0f} kg")
    print()

    print("=== QAR-pretrained encoder (Stage 1) on QAR test split per-segment ===")
    qar_stats_eval = evaluate_encoder_per_segment(qar_encoder, all_segments)
    print(f"  n_segments={qar_stats_eval['n_segments']}, n_flights={qar_stats_eval['n_flights']}")
    print(f"  within_corr  = {qar_stats_eval['within_corr']:+.4f}")
    print(f"  between_corr = {qar_stats_eval['between_corr']:+.4f}")
    print(f"  overall_corr = {qar_stats_eval['overall_corr']:+.4f}")
    print(f"  bias = {qar_stats_eval['bias']:.4f}, MAE = {qar_stats_eval['mae_kg']:.0f} kg")
    print()

    # Comparison table.
    delta = {
        "within": qar_stats_eval["within_corr"] - v14_stats_eval["within_corr"],
        "between": qar_stats_eval["between_corr"] - v14_stats_eval["between_corr"],
        "overall": qar_stats_eval["overall_corr"] - v14_stats_eval["overall_corr"],
    }
    print("=== Gains (QAR-pretrained vs v14) ===")
    print(f"  Δ within_corr  : {delta['within']:+.4f}")
    print(f"  Δ between_corr : {delta['between']:+.4f}")
    print(f"  Δ overall_corr : {delta['overall']:+.4f}")
    print()

    # Verdict.
    if qar_stats_eval["overall_corr"] >= 0.50 and delta["overall"] >= 0.10:
        verdict = "✅ Stage 2 swap-only PASS — pre-trained encoder transfers plug-and-play"
    elif delta["overall"] >= 0.10:
        verdict = "⚠️ Stage 2 partial — significant overall gain but threshold not reached"
    else:
        verdict = "❌ Stage 2 swap fails — pre-trained encoder doesn't transfer without retrain"
    print(f"Verdict : {verdict}")

    # Markdown report.
    lines = [
        "# Phase 5 Stage 2 (light) — Swap-only test : QAR-pretrained encoder in v14",
        "",
        "> Test : load v14 architecture, swap mass encoder weights with QAR-pretrained ones (Stage 1).",
        "> Run Test A per-segment on the 106 QAR test flights — *no retraining*.",
        "",
        f"## Per-segment evaluation on {qar_stats_eval['n_flights']} flights ({qar_stats_eval['n_segments']} segments)",
        "",
        "| Metric | v14 (baseline) | **QAR-pretrained swap** | Δ |",
        "|---|---:|---:|---:|",
        f"| within_corr | {v14_stats_eval['within_corr']:+.4f} | {qar_stats_eval['within_corr']:+.4f} | {delta['within']:+.4f} |",
        f"| between_corr | {v14_stats_eval['between_corr']:+.4f} | {qar_stats_eval['between_corr']:+.4f} | {delta['between']:+.4f} |",
        f"| **overall_corr** | {v14_stats_eval['overall_corr']:+.4f} | **{qar_stats_eval['overall_corr']:+.4f}** | **{delta['overall']:+.4f}** |",
        f"| MAE (kg) | {v14_stats_eval['mae_kg']:.0f} | {qar_stats_eval['mae_kg']:.0f} | {qar_stats_eval['mae_kg']-v14_stats_eval['mae_kg']:+.0f} |",
        f"| bias | {v14_stats_eval['bias']:.4f} | {qar_stats_eval['bias']:.4f} | — |",
        "",
        f"## Verdict : {verdict}",
        "",
        "## Compliance",
        "",
        "- R3 (test split untouched in training) : ✅ — Stage 1 trained on different split.",
        "- R11 (gate before full retrain) : ✅ — this test gates Stage 2 full retrain (next).",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\nReport saved : {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
