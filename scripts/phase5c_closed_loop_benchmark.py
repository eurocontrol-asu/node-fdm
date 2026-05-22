"""Phase 5c — v14 vs Pure P&S vs QAR-supervised closed-loop benchmark.

Closed-loop mass-swap ablation on AC7 (fuel-flow prediction vs QAR observed).

Same kinematic AC7 chain (T_obs from kinematics + drag_PS, η_PS analytical,
Eq 19) but with **different mass inputs** :

* **Pure P&S baseline** : m = 60 000 kg constant (fleet-average).
* **QAR oracle** : m = SYS__GW (Phase 4 AC7 baseline, theoretical ceiling).
* **v14 mass encoder** : m = v14's MLP(3 causal features) prediction.
* **QAR-pretrained encoder** : m = Phase 5 Stage 1's MLP prediction.

The ablation isolates *the value added by the mass encoder* over a naive
constant-mass baseline (representing pure P&S without learned m), and
quantifies how much of the AC7 ceiling is due to mass uncertainty.

On the 106 QAR test flights (Phase 5 R3 segregation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
NODE_FDM_SRC = REPO_ROOT / "packages" / "node-fdm" / "src"
if str(NODE_FDM_SRC) not in sys.path:
    sys.path.insert(0, str(NODE_FDM_SRC))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase4_fuelflow_qar_validation import (  # noqa: E402
    C_T_DO,
    G,
    LCV_KEROSENE,
    S_REF,
    c_d_ps,
    eta_o_np,
    isa_temp_pressure,
)
from phase5_stage2_swap_test import (  # noqa: E402
    load_qar_pretrained_encoder,
    load_v14_mass_encoder,
)

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")
TEST_TXT = Path("data/qar_splits/phase5_test.txt")
V14_CKPT_DIR = Path("data/models/full_hybrid_v14")

FT_PER_M = 3.28084
KTS_PER_MS = 1.9438445
R_GAS = 287.05
A320_OEW = 42_600.0
A320_MTOW = 78_000.0
A320_FLEET_AVG = 60_000.0
NM_TO_M = 1852.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return float(2 * 6_371_000.0 * math.asin(math.sqrt(a)))


def extract_cruise_with_features(parquet_path: Path) -> dict | None:
    """Extract cruise-stable samples + per-flight features for one flight."""
    needed = {"ALT__STD", "SPD__TAS", "SPD__MACH", "SYS__GW",
              "TRAJ__LAT_GPS", "TRAJ__LON_GPS",
              "TEMP__SAT", "FUEL__FF_LEFT", "FUEL__FF_RIGHT"}
    try:
        df = pl.read_parquet(parquet_path, columns=list(needed))
    except Exception:  # noqa: BLE001
        return None
    if not needed.issubset(df.columns):
        return None
    df = df.with_columns([
        (pl.col("ALT__STD") / FT_PER_M).alias("alt_m"),
        (pl.col("SPD__TAS") / KTS_PER_MS).alias("tas_ms"),
        (pl.col("TEMP__SAT") + 273.15).alias("temp_k"),
    ])
    df = df.with_columns([
        pl.col("alt_m").diff().alias("dalt_m_s"),
        pl.col("tas_ms").diff().alias("dtas_ms2"),
    ])
    df = df.drop_nulls(["alt_m", "tas_ms", "dtas_ms2", "dalt_m_s", "SPD__MACH",
                        "TRAJ__LAT_GPS", "TRAJ__LON_GPS",
                        "FUEL__FF_LEFT", "FUEL__FF_RIGHT"])

    # Per-flight features (extracted from GPS — Phase 5 standard).
    lat = df["TRAJ__LAT_GPS"].to_numpy()
    lon = df["TRAJ__LON_GPS"].to_numpy()
    valid_gps = (np.abs(lat) > 0.01) & (np.abs(lon) > 0.01) & np.isfinite(lat) & np.isfinite(lon)
    if valid_gps.sum() < 50:
        return None
    lat = lat[valid_gps]
    lon = lon[valid_gps]
    alt_full_m = df["alt_m"].to_numpy()[valid_gps]
    cruise_alt_max_m = float(np.nanmax(alt_full_m))
    lat0, lon0 = float(lat[0]), float(lon[0])
    lat_end, lon_end = float(lat[-1]), float(lon[-1])
    dist_total_m = haversine_m(lat0, lon0, lat_end, lon_end)

    df_cruise = df.filter(
        (pl.col("alt_m") > 8500.0)
        & (pl.col("alt_m") < 12500.0)
        & (pl.col("dalt_m_s").abs() < 0.5)
        & (pl.col("dtas_ms2").abs() < 0.3)
        & (pl.col("SPD__MACH") > 0.6)
        & (pl.col("SPD__MACH") < 0.85)
        & (pl.col("SYS__GW") > 40000)
        & (pl.col("SYS__GW") < 80000)
        & (pl.col("FUEL__FF_LEFT") > 0)
        & (pl.col("FUEL__FF_RIGHT") > 0)
    )
    if len(df_cruise) < 30:
        return None

    alt_m = df_cruise["alt_m"].to_numpy()
    tas_ms = df_cruise["tas_ms"].to_numpy()
    mach = df_cruise["SPD__MACH"].to_numpy()
    mass_qar = df_cruise["SYS__GW"].to_numpy()
    temp_k = df_cruise["temp_k"].to_numpy()
    dalt = df_cruise["dalt_m_s"].to_numpy()
    dtas = df_cruise["dtas_ms2"].to_numpy()
    ff_l = df_cruise["FUEL__FF_LEFT"].to_numpy().astype(np.float64)
    ff_r = df_cruise["FUEL__FF_RIGHT"].to_numpy().astype(np.float64)
    lat_cruise = df_cruise["TRAJ__LAT_GPS"].to_numpy()
    lon_cruise = df_cruise["TRAJ__LON_GPS"].to_numpy()

    gamma = np.arcsin(np.clip(dalt / np.maximum(tas_ms, 50.0), -0.3, 0.3))
    _, p_pa = isa_temp_pressure(alt_m)
    rho = p_pa / (R_GAS * temp_k)
    q_pa = 0.5 * rho * tas_ms * tas_ms
    mdot_f_qar_kg_s = (ff_l + ff_r) / 3600.0

    # Per-segment dist_adep_at_t0 — haversine from first GPS to this segment's GPS.
    dist_adep_per_seg = np.array(
        [haversine_m(lat0, lon0, float(lat_cruise[i]), float(lon_cruise[i]))
         for i in range(len(lat_cruise))],
        dtype=np.float64,
    )

    return {
        "flight": parquet_path.name,
        "n": len(alt_m),
        # Per-sample arrays (cruise-stable filter).
        "alt_m": alt_m,
        "tas_ms": tas_ms,
        "mach": mach,
        "mass_qar": mass_qar,
        "temp_k": temp_k,
        "dtas": dtas,
        "gamma": gamma,
        "q_pa": q_pa,
        "mdot_f_qar_kg_s": mdot_f_qar_kg_s,
        # Per-segment feature : dist_adep_at_t0 (each cruise sample).
        "dist_adep_per_seg_m": dist_adep_per_seg,
        # Per-flight constants.
        "dist_total_m": dist_total_m,
        "cruise_alt_max_m": cruise_alt_max_m,
    }


def compute_mdot_f_with_mass(
    mass_per_sample: np.ndarray,
    flight_data: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mdot_f_pred using a given mass-per-sample array.

    Returns (mdot_f_pred_kg_s, valid_mask).
    """
    q_pa = flight_data["q_pa"]
    mach = flight_data["mach"]
    temp_k = flight_data["temp_k"]
    tas_ms = flight_data["tas_ms"]
    dtas = flight_data["dtas"]
    gamma = flight_data["gamma"]
    mdot_qar = flight_data["mdot_f_qar_kg_s"]

    # Drag (depends on mass via C_L).
    c_l = mass_per_sample * G / np.maximum(q_pa * S_REF, 1.0)
    cd = c_d_ps(c_l, mach, temp_k, q_pa, tas_ms)
    drag_n = q_pa * S_REF * cd

    # T_obs from kinematics (depends on mass).
    t_obs = mass_per_sample * dtas + drag_n + mass_per_sample * G * np.sin(gamma)

    # C_T → η_PS → mdot_f.
    c_t_inst = t_obs / np.maximum(q_pa * S_REF, 1.0)
    eta_ps = eta_o_np(c_t_inst, mach)
    mdot_pred = t_obs * tas_ms / np.maximum(eta_ps * LCV_KEROSENE, 1e-6)

    valid = (t_obs > 1000.0) & (eta_ps > 0.05) & (eta_ps < 0.55) \
            & (mdot_qar > 0.1) & (mdot_qar < 5.0) & np.isfinite(mdot_pred)
    return mdot_pred, valid


def run_encoder_predictions(
    encoder,
    flights: list[dict],
    features_in_nm: bool,
) -> np.ndarray:
    """Run the mass encoder on per-flight features, return m_pred per FLIGHT."""
    # Features per flight : [dist_total, dist_adep_at_t0_at_first_cruise_sample, cruise_alt_max].
    # The mass encoder gives one prediction per (3 features) input.
    # For the closed-loop test, we need m per CRUISE SAMPLE — but the encoder
    # is per-flight (or per-segment with varying dist_adep). We use
    # per-cruise-sample dist_adep (so prediction varies within the flight).
    UNIT = (1.0 / NM_TO_M) if features_in_nm else 1.0
    preds_per_flight = []
    for f in flights:
        n = f["n"]
        feats = np.stack([
            np.full(n, f["dist_total_m"] * UNIT, dtype=np.float32),
            (f["dist_adep_per_seg_m"] * UNIT).astype(np.float32),
            np.full(n, f["cruise_alt_max_m"], dtype=np.float32),  # always in metres
        ], axis=-1)
        with torch.no_grad():
            m_pred = encoder(torch.from_numpy(feats)).detach().numpy().astype(np.float32)
        preds_per_flight.append(m_pred)
    return np.concatenate(preds_per_flight)


def compute_stats(
    label: str,
    mdot_pred: np.ndarray,
    mdot_qar: np.ndarray,
    valid: np.ndarray,
    mass_used: np.ndarray,
    mass_qar: np.ndarray,
) -> dict:
    if int(valid.sum()) < 100:
        return {"label": label, "n": 0, "corr": float("nan")}
    mp = mdot_pred[valid]
    mq = mdot_qar[valid]
    abs_rel = np.abs((mp - mq) / mq)
    return {
        "label": label,
        "n": int(valid.sum()),
        "corr": float(np.corrcoef(mp, mq)[0, 1]),
        "bias_mdot": float(np.median(mp / mq)),
        "median_abs_rel_err_pct": float(np.median(abs_rel) * 100),
        "p99_abs_rel_err_pct": float(np.percentile(abs_rel, 99) * 100),
        "mdot_pred_median_kg_s": float(np.median(mp)),
        "mdot_qar_median_kg_s": float(np.median(mq)),
        # Mass diagnostics (where valid).
        "mass_used_median_kg": float(np.median(mass_used[valid])),
        "mass_corr_vs_qar": float(np.corrcoef(mass_used[valid], mass_qar[valid])[0, 1])
            if mass_used[valid].std() > 1e-3 else 0.0,
    }


def main() -> int:  # noqa: PLR0915
    print("Loading encoders ...")
    v14_encoder, _ = load_v14_mass_encoder(V14_CKPT_DIR)
    qar_encoder, _ = load_qar_pretrained_encoder()
    print()

    filenames = [line.strip() for line in TEST_TXT.read_text().splitlines() if line.strip()]
    print(f"Loading {len(filenames)} test flights ...")
    flights = []
    for i, name in enumerate(filenames):
        if i % 25 == 0:
            print(f"  [{i+1}/{len(filenames)}] {name}")
        d = extract_cruise_with_features(QAR_DIR / name)
        if d is not None:
            flights.append(d)
    print(f"  Used {len(flights)} flights.")
    print()

    # Concatenate per-sample arrays across flights.
    mass_qar = np.concatenate([f["mass_qar"] for f in flights])
    mdot_qar = np.concatenate([f["mdot_f_qar_kg_s"] for f in flights])
    n_total = len(mass_qar)
    print(f"Total cruise-stable samples : {n_total}")

    # Per-sample mass for each variant.
    mass_pure_ps = np.full(n_total, A320_FLEET_AVG, dtype=np.float32)
    mass_oracle = mass_qar.astype(np.float32)
    # v14 expects features in metres ; QAR-pretrained expects NM.
    mass_v14 = run_encoder_predictions(v14_encoder, flights, features_in_nm=False)
    mass_qar_pre = run_encoder_predictions(qar_encoder, flights, features_in_nm=True)

    # === Compute mdot_f for each mass variant ===
    variants = [
        ("Pure P&S (m=60t constant)", mass_pure_ps),
        ("v14 mass encoder", mass_v14),
        ("QAR-pretrained encoder", mass_qar_pre),
        ("QAR oracle (theoretical ceiling)", mass_oracle),
    ]
    results = []
    for label, mass in variants:
        mdot_preds = []
        valids = []
        offset = 0
        for f in flights:
            n = f["n"]
            m_slice = mass[offset:offset + n]
            mp, vd = compute_mdot_f_with_mass(m_slice, f)
            mdot_preds.append(mp)
            valids.append(vd)
            offset += n
        mdot_pred = np.concatenate(mdot_preds)
        valid = np.concatenate(valids)
        stats = compute_stats(label, mdot_pred, mdot_qar, valid, mass, mass_qar)
        results.append(stats)
        print(f"\n=== {label} ===")
        print(f"  n_valid : {stats.get('n', 0)}")
        if stats.get("n", 0):
            print(f"  AC7 corr Pearson : {stats['corr']:+.4f}")
            print(f"  bias mdot_pred/mdot_qar : {stats['bias_mdot']:.4f}")
            print(f"  median |Δmdot|/mdot_qar : {stats['median_abs_rel_err_pct']:.2f} %")
            print(f"  p99 |Δmdot|/mdot_qar : {stats['p99_abs_rel_err_pct']:.2f} %")
            print(f"  mass used : median {stats['mass_used_median_kg']:.0f} kg, corr_vs_qar {stats['mass_corr_vs_qar']:+.4f}")

    # === Summary table ===
    print("\n\n========================= AC7 BENCHMARK TABLE =========================")
    print(f"{'Variant':<40s} {'AC7 corr':>9s} {'bias':>7s} {'med|Δ|':>8s} {'mass_corr':>10s}")
    for r in results:
        if r.get("n", 0):
            print(f"{r['label']:<40s} {r['corr']:>+9.4f} {r['bias_mdot']:>7.4f} "
                  f"{r['median_abs_rel_err_pct']:>7.2f}% {r['mass_corr_vs_qar']:>+10.4f}")

    # Δ vs Pure P&S baseline.
    print("\n========================= GAIN OVER PURE P&S =========================")
    ps_corr = results[0]["corr"] if results[0].get("n") else float("nan")
    for r in results[1:]:
        if r.get("n", 0):
            d = r["corr"] - ps_corr
            print(f"{r['label']:<40s} Δ AC7 corr = {d:+.4f}")

    # === Markdown report ===
    out = Path("data/investigations/phase5_qar_joint_supervision/artifacts/closed_loop_benchmark.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 5c — Closed-loop AC7 benchmark : v14 vs Pure P&S vs QAR-supervised",
        "",
        f"> n_flights = {len(flights)}, n_cruise_samples = {n_total}.",
        "> Same kinematic AC7 chain (T_obs reconstruction + η_PS + Eq 19) with different mass inputs.",
        "",
        "## Variants tested",
        "",
        "- **Pure P&S** : m = 60 000 kg constant (fleet-average baseline, no learned mass).",
        "- **v14 mass encoder** : m predicted per-segment from `MassEncoderLinearTempered`(3 features) — *trained on ADS-B Delta JetBlue/AAL/etc. 2025*.",
        "- **QAR-pretrained encoder** : m predicted from Phase 5 Stage 1 encoder (trained on 414 QAR train flights).",
        "- **QAR oracle** : m = SYS__GW observed (theoretical ceiling — what AC7 with perfect mass would give).",
        "",
        "## Results",
        "",
        "| Variant | AC7 corr | bias | median \\|Δmdot\\|/mdot_qar | p99 | mass corr vs QAR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r.get("n", 0):
            lines.append(
                f"| {r['label']} | {r['corr']:+.4f} | {r['bias_mdot']:.4f} | "
                f"{r['median_abs_rel_err_pct']:.2f}% | {r['p99_abs_rel_err_pct']:.2f}% | "
                f"{r['mass_corr_vs_qar']:+.4f} |"
            )
    lines.extend([
        "",
        "## Gain over Pure P&S baseline",
        "",
        "| Variant | Δ AC7 corr | Interpretation |",
        "|---|---:|---|",
    ])
    if results[0].get("n"):
        for r in results[1:]:
            if r.get("n", 0):
                d = r["corr"] - ps_corr
                lines.append(f"| {r['label']} | {d:+.4f} | |")
    lines.extend([
        "",
        "## Key questions answered",
        "",
        "1. **Does v14 mass encoder add value over Pure P&S (m=fleet avg)?**",
        f"   Δ corr = {results[1]['corr'] - ps_corr:+.4f}  ",
        "2. **Does QAR-supervised mass encoder beat v14?**",
        f"   Δ corr = {results[2]['corr'] - results[1]['corr']:+.4f}  ",
        "3. **What's the theoretical ceiling (perfect mass)?**",
        f"   Oracle corr = {results[3]['corr']:+.4f}  ",
        f"   Gap v14 to oracle = {results[3]['corr'] - results[1]['corr']:+.4f}  ",
        "",
        "## R5 compliance",
        "",
        "- v14 trained R5-strict on ADS-B (no QAR in training loop).",
        "- QAR-pretrained encoder = explicit R5 break (Phase 5 demonstration).",
        "- Pure P&S = no training at all.",
        "- QAR oracle = bound test, not a deployable model.",
    ])
    out.write_text("\n".join(lines))
    print(f"\nReport saved : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
