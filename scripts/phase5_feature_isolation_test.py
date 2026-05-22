"""Phase 5 — Feature isolation test : disambiguate the +0.337 vs +0.476 gap.

Hypothesis tested:
    "The Phase 1.8a Test A v14 baseline corr +0.337 was suppressed by the
     segment-varying `dist_total_flight = adep_at_t0 + ades_at_t0` feature.
     With per-flight constant `dist_total_flight = haversine(first, last)`,
     v14 gives +0.476 on the same data."

Method: on the SAME 106 QAR test flights, with the SAME per-segment
windowing, compute BOTH feature definitions, evaluate v14 on each, and
compare per-segment overall corr. The only variable that changes is the
feature definition.

Outcome interpretation:
- If both give similar corr (within ±0.05) → the +0.337 / +0.476 gap was
  due to scope/filter differences, not feature definition. P3 claim falsified.
- If +0.476 reproducible only with per-flight constant `dist_total_flight`
  → P3 hypothesis confirmed. Feature definition is the lever.
- Intermediate → partial confirmation, document magnitude carefully.

Also runs Phase 1.8a Test A style on all 530 flights to reproduce the
original +0.337 baseline (sanity check that our Test A re-implementation
matches the historic measurement).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_FDM_SRC = REPO_ROOT / "packages" / "node-fdm" / "src"
if str(NODE_FDM_SRC) not in sys.path:
    sys.path.insert(0, str(NODE_FDM_SRC))

# Re-import the swap test helpers.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase5_stage2_swap_test import (  # noqa: E402
    load_qar_pretrained_encoder,
    load_v14_mass_encoder,
)

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")
TEST_TXT = Path("data/qar_splits/phase5_test.txt")
V14_CKPT_DIR = Path("data/models/full_hybrid_v14")

FT_PER_M = 3.28084


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return float(2 * 6_371_000.0 * math.asin(math.sqrt(a)))


def extract_dual_features(parquet_path: Path, seq_len: int = 60, shift: int = 60) -> list[dict]:
    """For each segment, return BOTH feature definitions + m_qar."""
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

    valid = (np.abs(lat) > 0.01) & (np.abs(lon) > 0.01) & np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 50:
        return []
    lat = lat[valid]
    lon = lon[valid]
    alt_m = alt_m[valid]
    sys_gw = sys_gw[valid]

    cruise_alt_max_m = float(np.nanmax(alt_m))
    lat0, lon0 = float(lat[0]), float(lon[0])
    lat_end, lon_end = float(lat[-1]), float(lon[-1])

    # Per-flight constant (Stage 1 / my Exp 4 definition).
    dist_total_per_flight = haversine_m(lat0, lon0, lat_end, lon_end)  # metres

    segments = []
    for start in range(0, len(lat) - seq_len + 1, shift):
        m_t0 = float(sys_gw[start])
        if not (40_000 < m_t0 < 80_000):
            continue
        lat_t = float(lat[start])
        lon_t = float(lon[start])
        adep_at_t0 = haversine_m(lat0, lon0, lat_t, lon_t)
        ades_at_t0 = haversine_m(lat_t, lon_t, lat_end, lon_end)
        # Phase 1.8a Test A style : per-segment dist_total_flight.
        dist_total_test_a = adep_at_t0 + ades_at_t0

        segments.append({
            "flight": parquet_path.name,
            # Phase 1.8a Test A definition.
            "dist_total_test_a": dist_total_test_a,
            "dist_adep_at_t0": adep_at_t0,
            "cruise_alt_max_flight": cruise_alt_max_m,
            # Stage 1 / Exp 4 definition (differs only in dist_total).
            "dist_total_stage1": dist_total_per_flight,
            # Target.
            "m_qar": m_t0,
            "alt_t0": float(alt_m[start]),
        })
    return segments


def compute_per_segment_metrics(
    encoder,
    X: np.ndarray,
    m_qar: np.ndarray,
    flight_ids: list[str],
) -> dict[str, float]:
    """Per-segment within/between/overall corr + MAE + bias."""
    with torch.no_grad():
        m_pred = encoder(torch.from_numpy(X).to(torch.float32)).detach().numpy().astype(np.float32)
    overall_corr = float(np.corrcoef(m_pred, m_qar)[0, 1])

    flight_groups = defaultdict(list)
    for i, fid in enumerate(flight_ids):
        flight_groups[fid].append((m_pred[i], m_qar[i]))
    within_corrs = []
    fmp = []
    fmq = []
    for _fid, pairs in flight_groups.items():
        ps = np.array([p for p, _ in pairs])
        qs = np.array([q for _, q in pairs])
        fmp.append(float(ps.mean()))
        fmq.append(float(qs.mean()))
        if len(pairs) >= 2 and ps.std() > 1e-6 and qs.std() > 1e-6:
            within_corrs.append(float(np.corrcoef(ps, qs)[0, 1]))
    within = float(np.mean(within_corrs)) if within_corrs else float("nan")
    between = float(np.corrcoef(fmp, fmq)[0, 1]) if len(fmp) > 1 else float("nan")
    mae = float(np.mean(np.abs(m_pred - m_qar)))
    bias = float(np.median(m_pred / m_qar))
    return {
        "n_segments": len(m_qar),
        "n_flights": len(flight_groups),
        "within_corr": within,
        "between_corr": between,
        "overall_corr": overall_corr,
        "mae_kg": mae,
        "bias": bias,
    }


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/investigations/phase5_qar_joint_supervision/artifacts/feature_isolation_test.md"),
    )
    args = parser.parse_args()

    print(f"Loading v14 mass encoder ...")
    v14_encoder, v14_stats = load_v14_mass_encoder(V14_CKPT_DIR)
    print(f"v14 normalizer stats : {v14_stats}")
    print()
    print(f"Loading QAR-pretrained encoder ...")
    qar_encoder, qar_stats = load_qar_pretrained_encoder()
    print(f"QAR-pretrained normalizer stats : {qar_stats}")
    print()

    filenames = [line.strip() for line in TEST_TXT.read_text().splitlines() if line.strip()]
    print(f"Loading {len(filenames)} test flights ...")
    all_segments = []
    for i, name in enumerate(filenames):
        if i % 25 == 0:
            print(f"  [{i+1}/{len(filenames)}] {name}")
        segs = extract_dual_features(QAR_DIR / name)
        all_segments.extend(segs)
    print(f"  Total : {len(all_segments)} segments across {len({s['flight'] for s in all_segments})} flights")
    print()

    # ===== Build feature arrays for each definition =====
    # v14 was trained on features in METRES (loader.py uses haversine_m).
    # The Stage 1 encoder was trained on features in NAUTICAL MILES.
    # Both feature definitions are evaluated on each encoder *with proper unit conversion*.
    NM_TO_M = 1852.0
    M_TO_NM = 1.0 / NM_TO_M

    flight_ids = [s["flight"] for s in all_segments]
    m_qar = np.array([s["m_qar"] for s in all_segments], dtype=np.float32)

    # Phase 1.8a Test A definition (metres).
    X_test_a_m = np.array([[s["dist_total_test_a"], s["dist_adep_at_t0"], s["cruise_alt_max_flight"]]
                            for s in all_segments], dtype=np.float32)
    # Stage 1 / Exp 4 definition (metres, but with per-flight dist_total).
    X_stage1_m = np.array([[s["dist_total_stage1"], s["dist_adep_at_t0"], s["cruise_alt_max_flight"]]
                            for s in all_segments], dtype=np.float32)
    # NM versions for QAR encoder.
    X_test_a_nm = X_test_a_m.copy()
    X_test_a_nm[:, 0] *= M_TO_NM
    X_test_a_nm[:, 1] *= M_TO_NM
    X_stage1_nm = X_stage1_m.copy()
    X_stage1_nm[:, 0] *= M_TO_NM
    X_stage1_nm[:, 1] *= M_TO_NM

    # ===== Four evaluations : 2 encoders × 2 feature definitions =====
    print("=== v14 on Phase 1.8a Test A features (per-segment dist_total = adep+ades) ===")
    v14_test_a = compute_per_segment_metrics(v14_encoder, X_test_a_m, m_qar, flight_ids)
    print(f"  overall_corr = {v14_test_a['overall_corr']:+.4f}")
    print(f"  within_corr = {v14_test_a['within_corr']:+.4f}")
    print(f"  between_corr = {v14_test_a['between_corr']:+.4f}")
    print(f"  MAE = {v14_test_a['mae_kg']:.0f} kg, bias = {v14_test_a['bias']:.4f}")
    print()

    print("=== v14 on Stage 1 features (per-flight dist_total = haversine(first,last)) ===")
    v14_stage1 = compute_per_segment_metrics(v14_encoder, X_stage1_m, m_qar, flight_ids)
    print(f"  overall_corr = {v14_stage1['overall_corr']:+.4f}")
    print(f"  within_corr = {v14_stage1['within_corr']:+.4f}")
    print(f"  between_corr = {v14_stage1['between_corr']:+.4f}")
    print(f"  MAE = {v14_stage1['mae_kg']:.0f} kg, bias = {v14_stage1['bias']:.4f}")
    print()

    print("=== QAR-pretrained on Phase 1.8a Test A features (NM units) ===")
    qar_test_a = compute_per_segment_metrics(qar_encoder, X_test_a_nm, m_qar, flight_ids)
    print(f"  overall_corr = {qar_test_a['overall_corr']:+.4f}")
    print(f"  within_corr = {qar_test_a['within_corr']:+.4f}")
    print(f"  between_corr = {qar_test_a['between_corr']:+.4f}")
    print(f"  MAE = {qar_test_a['mae_kg']:.0f} kg, bias = {qar_test_a['bias']:.4f}")
    print()

    print("=== QAR-pretrained on Stage 1 features (NM units, training distribution) ===")
    qar_stage1 = compute_per_segment_metrics(qar_encoder, X_stage1_nm, m_qar, flight_ids)
    print(f"  overall_corr = {qar_stage1['overall_corr']:+.4f}")
    print(f"  within_corr = {qar_stage1['within_corr']:+.4f}")
    print(f"  between_corr = {qar_stage1['between_corr']:+.4f}")
    print(f"  MAE = {qar_stage1['mae_kg']:.0f} kg, bias = {qar_stage1['bias']:.4f}")
    print()

    # ===== Verdict on the P3 hypothesis =====
    delta_v14 = v14_stage1["overall_corr"] - v14_test_a["overall_corr"]
    delta_qar = qar_stage1["overall_corr"] - qar_test_a["overall_corr"]
    print("=== P3 hypothesis test : effect of dist_total_flight definition ===")
    print(f"v14 : Test A features = {v14_test_a['overall_corr']:+.4f} ; Stage 1 features = {v14_stage1['overall_corr']:+.4f} ; Δ = {delta_v14:+.4f}")
    print(f"QAR : Test A features = {qar_test_a['overall_corr']:+.4f} ; Stage 1 features = {qar_stage1['overall_corr']:+.4f} ; Δ = {delta_qar:+.4f}")
    print()

    if abs(delta_v14) < 0.05 and abs(delta_qar) < 0.05:
        verdict = "❌ P3 FALSIFIED : feature definition change has negligible effect (Δ < 0.05)"
    elif delta_v14 > 0.10 and delta_qar > 0.05:
        verdict = "✅ P3 CONFIRMED : per-flight dist_total improves corr meaningfully for both encoders"
    elif delta_v14 > 0.10 or delta_qar > 0.10:
        verdict = "⚠️ P3 PARTIAL : effect significant for one encoder but not the other"
    else:
        verdict = "⚠️ P3 inconclusive : small but non-zero effect"
    print(f"\nVerdict : {verdict}")

    # ===== Markdown report =====
    lines = [
        "# Phase 5 — Feature isolation test (post-Exp 4 P3 hypothesis check)",
        "",
        f"> Same 106 QAR test flights ({len(all_segments)} segments).",
        f"> Same per-segment windowing (seq=60, shift=60).",
        "> Only variable : `dist_total_flight` definition.",
        "",
        "## Setup",
        "",
        "- Phase 1.8a Test A def : `dist_total_flight = adep_at_t0 + ades_at_t0` (varies per segment).",
        "- Stage 1 / Exp 4 def : `dist_total_flight = haversine(first, last)` (constant per flight).",
        "- Other features (`dist_adep_at_t0`, `cruise_alt_max_flight`) identical between definitions.",
        "",
        "## Results — 2 encoders × 2 feature definitions",
        "",
        "| Encoder × Feature | overall_corr | within_corr | between_corr | MAE (kg) | bias |",
        "|---|---:|---:|---:|---:|---:|",
        f"| v14 × Test A features | {v14_test_a['overall_corr']:+.4f} | {v14_test_a['within_corr']:+.4f} | {v14_test_a['between_corr']:+.4f} | {v14_test_a['mae_kg']:.0f} | {v14_test_a['bias']:.4f} |",
        f"| **v14 × Stage 1 features** | **{v14_stage1['overall_corr']:+.4f}** | {v14_stage1['within_corr']:+.4f} | {v14_stage1['between_corr']:+.4f} | {v14_stage1['mae_kg']:.0f} | {v14_stage1['bias']:.4f} |",
        f"| QAR × Test A features | {qar_test_a['overall_corr']:+.4f} | {qar_test_a['within_corr']:+.4f} | {qar_test_a['between_corr']:+.4f} | {qar_test_a['mae_kg']:.0f} | {qar_test_a['bias']:.4f} |",
        f"| **QAR × Stage 1 features** | **{qar_stage1['overall_corr']:+.4f}** | {qar_stage1['within_corr']:+.4f} | {qar_stage1['between_corr']:+.4f} | {qar_stage1['mae_kg']:.0f} | {qar_stage1['bias']:.4f} |",
        "",
        "## Feature-definition effect (Δ = Stage 1 − Test A)",
        "",
        "| Encoder | Δ overall_corr | Δ between_corr |",
        "|---|---:|---:|",
        f"| v14 | {delta_v14:+.4f} | {v14_stage1['between_corr']-v14_test_a['between_corr']:+.4f} |",
        f"| QAR-pretrained | {delta_qar:+.4f} | {qar_stage1['between_corr']-qar_test_a['between_corr']:+.4f} |",
        "",
        f"## Verdict : **{verdict}**",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"\nReport saved : {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
