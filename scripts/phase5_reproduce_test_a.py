"""Phase 5 — Reproduce Phase 1.8a Test A baseline +0.337 exactly.

Goal : identify which combination of (flight scope, filters, position
source) produces v14 overall_corr +0.337 vs my Exp 4's +0.476.

Runs Test A logic *identical* to `scripts/_validation/test_a_per_segment.py`
on 4 scope variations:
* All 530 valid flights × alt >= 500 m × NAV__LAT/LONG (Test A native)
* 106 test split × alt >= 500 m × NAV__LAT/LONG
* All 530 × alt >= 0 (no alt filter) × TRAJ__LAT_GPS/LONG (Exp 4 native)
* 106 test split × alt >= 0 × TRAJ__LAT_GPS/LONG (Exp 4 native)

If +0.337 reproduces only with Test A filters → the gap is filtering.
If +0.337 reproduces only on all 530 flights → the gap is population.
"""

from __future__ import annotations

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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase5_stage2_swap_test import load_v14_mass_encoder  # noqa: E402

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")
TRAIN_TXT = Path("data/qar_splits/phase5_train.txt")
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


def extract_segments_test_a_style(
    parquet_path: Path,
    *,
    use_nav_pos: bool = True,
    min_alt_m: float = 500.0,
    seq_len: int = 60,
    shift: int = 60,
) -> list[dict]:
    """Replicate Phase 1.8a Test A's `_build_segments` exactly when use_nav_pos=True, min_alt=500."""
    if use_nav_pos:
        pos_cols = ["NAV__LAT", "NAV__LONG"]
    else:
        pos_cols = ["TRAJ__LAT_GPS", "TRAJ__LON_GPS"]
    needed = {"ALT__STD", "SYS__GW", *pos_cols}
    try:
        df = pl.read_parquet(parquet_path, columns=list(needed))
    except Exception:  # noqa: BLE001
        return []
    if not needed.issubset(df.columns):
        return []
    df = df.drop_nulls(["ALT__STD", "SYS__GW", *pos_cols])
    if len(df) < seq_len + 1:
        return []

    lat = df[pos_cols[0]].to_numpy()
    lon = df[pos_cols[1]].to_numpy()
    alt_m = (df["ALT__STD"] / FT_PER_M).to_numpy()
    gw = df["SYS__GW"].to_numpy()

    valid_pos = np.isfinite(lat) & np.isfinite(lon) & (lat != 0.0) & (lon != 0.0)
    if not valid_pos.any():
        return []
    first_valid = int(np.argmax(valid_pos))
    last_valid = int(np.where(valid_pos)[0][-1])
    lat0, lon0 = float(lat[first_valid]), float(lon[first_valid])
    lat_end, lon_end = float(lat[last_valid]), float(lon[last_valid])

    cruise_alt_max_m = float(np.nanmax(alt_m))

    n = len(df)
    segments = []
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
        adep_at_t0 = haversine_m(lat0, lon0, lat_t, lon_t)
        ades_at_t0 = haversine_m(lat_t, lon_t, lat_end, lon_end)
        segments.append({
            "flight": parquet_path.name,
            "dist_total_flight": adep_at_t0 + ades_at_t0,
            "dist_adep_at_t0": adep_at_t0,
            "cruise_alt_max_flight": cruise_alt_max_m,
            "m_qar": m_truth,
        })
    return segments


def evaluate(encoder, segments: list[dict]) -> dict[str, float]:
    if not segments:
        return {"n_segments": 0, "n_flights": 0, "overall_corr": float("nan")}
    X = np.array([[s["dist_total_flight"], s["dist_adep_at_t0"], s["cruise_alt_max_flight"]]
                  for s in segments], dtype=np.float32)
    m_qar = np.array([s["m_qar"] for s in segments], dtype=np.float32)
    flight_ids = [s["flight"] for s in segments]
    with torch.no_grad():
        m_pred = encoder(torch.from_numpy(X).to(torch.float32)).detach().numpy().astype(np.float32)
    overall = float(np.corrcoef(m_pred, m_qar)[0, 1])

    flight_groups = defaultdict(list)
    for i, fid in enumerate(flight_ids):
        flight_groups[fid].append((m_pred[i], m_qar[i]))
    within = []
    fmp, fmq = [], []
    for _fid, pairs in flight_groups.items():
        ps = np.array([p for p, _ in pairs])
        qs = np.array([q for _, q in pairs])
        fmp.append(float(ps.mean()))
        fmq.append(float(qs.mean()))
        if len(pairs) >= 2 and ps.std() > 1e-6 and qs.std() > 1e-6:
            within.append(float(np.corrcoef(ps, qs)[0, 1]))
    within_corr = float(np.mean(within)) if within else float("nan")
    between_corr = float(np.corrcoef(fmp, fmq)[0, 1]) if len(fmp) > 1 else float("nan")
    return {
        "n_segments": len(segments),
        "n_flights": len(flight_groups),
        "overall_corr": overall,
        "within_corr": within_corr,
        "between_corr": between_corr,
    }


def main() -> int:  # noqa: PLR0915
    print("Loading v14 ...")
    v14_encoder, _ = load_v14_mass_encoder(V14_CKPT_DIR)
    print()

    # Collect 3 scopes : all 520 (valid), 414 train, 106 test.
    train_names = set(TRAIN_TXT.read_text().splitlines())
    test_names = set(TEST_TXT.read_text().splitlines())
    train_names.discard("")
    test_names.discard("")
    all_names = sorted({p.name for p in QAR_DIR.glob("*A320*.parquet")})

    print(f"All A320 parquet files : {len(all_names)}")
    print(f"Train split : {len(train_names)} flights")
    print(f"Test split  : {len(test_names)} flights")
    print()

    # 4 conditions × 3 scopes = 12 evaluations (only run informative ones).
    cases = [
        # (label, use_nav_pos, min_alt_m, scope_filter)
        ("Test A native (NAV pos, alt>500m) × all valid", True, 500.0, all_names),
        ("Test A native × train split (414)", True, 500.0, sorted(train_names)),
        ("Test A native × test split (106)", True, 500.0, sorted(test_names)),
        ("Exp 4 style (TRAJ_GPS pos, no alt filter) × all valid", False, 0.0, all_names),
        ("Exp 4 style × test split (106)", False, 0.0, sorted(test_names)),
        ("TRAJ_GPS pos + alt>500m × test split", False, 500.0, sorted(test_names)),
        ("NAV pos + no alt filter × test split", True, 0.0, sorted(test_names)),
    ]

    results = []
    for label, use_nav, min_alt, names in cases:
        print(f"=== {label} ===")
        print(f"  scope : {len(names)} flights, use_nav_pos={use_nav}, min_alt_m={min_alt}")
        all_segs = []
        for i, name in enumerate(names):
            if i % 100 == 0:
                print(f"    [{i+1}/{len(names)}] {name}", flush=True)
            segs = extract_segments_test_a_style(
                QAR_DIR / name, use_nav_pos=use_nav, min_alt_m=min_alt,
            )
            all_segs.extend(segs)
        if not all_segs:
            print("  no segments")
            continue
        stats = evaluate(v14_encoder, all_segs)
        print(f"  n_segments={stats['n_segments']}, n_flights={stats['n_flights']}")
        print(f"  overall_corr  = {stats['overall_corr']:+.4f}")
        print(f"  within_corr   = {stats['within_corr']:+.4f}")
        print(f"  between_corr  = {stats['between_corr']:+.4f}")
        print()
        results.append((label, stats))

    # Summary table
    print("\n=== Summary : v14 overall_corr by scope/filter ===")
    print(f"{'Condition':<60s} {'n_seg':>8s} {'n_fli':>6s} {'overall':>8s} {'within':>8s} {'between':>8s}")
    for label, s in results:
        print(f"{label:<60s} {s['n_segments']:>8d} {s['n_flights']:>6d} {s['overall_corr']:>+8.4f} {s['within_corr']:>+8.4f} {s['between_corr']:>+8.4f}")

    # Write markdown.
    out = Path("data/investigations/phase5_qar_joint_supervision/artifacts/test_a_reproduction.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5 — Test A reproduction : identify the +0.337 source",
        "",
        "> Run Test A logic with various (scope, filter, position source) combinations to identify",
        "> which variable causes v14 overall_corr +0.337 (Phase 1.8a baseline) vs +0.476 (my Exp 4).",
        "",
        "| Condition | n_seg | n_fli | overall | within | between |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, s in results:
        lines.append(
            f"| {label} | {s['n_segments']} | {s['n_flights']} | "
            f"{s['overall_corr']:+.4f} | {s['within_corr']:+.4f} | {s['between_corr']:+.4f} |"
        )
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"\nReport saved : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
