"""Test A — MassEncoder QAR validation.

Validate the MassEncoder of trained models (v2 Newton, v3 CL mass-aware)
against ground-truth mass (SYS__GW) from real QAR data.

For each QAR flight, computes the 6 vol-niveau features the same way as the
ADS-B training pipeline, passes them through each model's MassEncoder, and
compares the predicted m_0 to SYS__GW at the segment start (default: first
stable cruise sample with ALT__STD > 25 000 ft).

Usage::

    uv run python scripts/_validation/test_a_mass_encoder_qar.py \
        --qar-dir /Users/gabriel/Downloads/QAR3 \
        --models full_hybrid_v2 full_hybrid_v3

Writes ``data/models/_comparison/test_a_mass_encoder_qar.md``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from node_fdm.architectures import (  # noqa: F401  (auto-register)
    adsb_hybrid_v2,
    adsb_hybrid_v3,
    adsb_hybrid_v4,
    adsb_hybrid_v5_tempered,
    adsb_hybrid_v6_mlp,
    adsb_hybrid_v7_lean,
    adsb_hybrid_v8_lean_t15,
    adsb_hybrid_v9_causal,
    adsb_hybrid_v10_ps_auxloss,
    adsb_hybrid_v11_ps_residual,
    adsb_hybrid_v12_psdrag,
    adsb_hybrid_v13_psthrust,
    adsb_hybrid_v13b_psthrust_w10,
    adsb_hybrid_v13c_psthrust_tet,
    adsb_hybrid_v13d_psthrust_parallel,
    adsb_hybrid_v14_psefficiency,
)
from node_fdm.architectures.registry import get as get_arch_spec
from node_fdm.layers.mass_encoder import MassEncoderLinear, MassEncoderLinearTempered
from node_fdm_data.schemas.adsb_hybrid import (
    A320_MTOW_KG,
    A320_OEW_KG,
    FLIGHT_FEATURE_COLS_6,
    FLIGHT_FEATURE_SIGNS_6,
)

_R_EARTH_M = 6_371_000.0
_FT_TO_M = 0.3048
_KT_TO_MS = 0.514444
_MACH_CRUISE_MIN = 0.70
_MACH_CRUISE_MAX = 0.86
_MACH_CRUISE_FALLBACK = 0.78
_STABLE_CRUISE_ALT_FT = 25_000.0
_CLIMB_BAND_LOW_M = 1500.0
_CLIMB_BAND_HIGH_M = 4500.0
_FL240_M = 7300.0
_GROUND_OFFSET_M = 1000.0
_QAR_NOMINAL_DT_S = 1.0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two (lat, lon) points in degrees."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _R_EARTH_M * math.asin(math.sqrt(a))


def _find_stable_cruise_idx(df: pl.DataFrame, min_alt_ft: float = _STABLE_CRUISE_ALT_FT) -> int:
    """Return index of the first sample with ALT__STD above ``min_alt_ft`` (fall-back: 0)."""
    alt = df["ALT__STD"].to_numpy()
    above = np.isfinite(alt) & (alt > min_alt_ft)
    if not above.any():
        return 0
    return int(np.argmax(above))


def _resolve_feature_cols(meta: dict) -> tuple[list[str], list[float]]:
    """Look up flight-feature columns and signs from the meta.json arch name.

    Falls back to FLIGHT_FEATURE_COLS_6 / SIGNS_6 when the arch is not
    registered (e.g. legacy checkpoints) so older models stay readable.
    """
    arch_name = meta.get("architecture_name", "")
    try:
        spec = get_arch_spec(arch_name)
    except KeyError:
        return list(FLIGHT_FEATURE_COLS_6), list(FLIGHT_FEATURE_SIGNS_6)
    cols = list(getattr(spec, "flight_feature_cols", None) or FLIGHT_FEATURE_COLS_6)
    signs = list(getattr(spec, "flight_feature_signs", None) or FLIGHT_FEATURE_SIGNS_6)
    return cols, signs


def _dynamic_qar_aggregates(df: pl.DataFrame, alt_m: np.ndarray) -> dict[str, float]:
    """Compute the 3 Experiment-01 dynamic mass-signature aggregates on QAR.

    Mirrors the ADS-B loader logic:
      - climb_rate_mean_climb : mean ATT__VV (m/s) on the low-climb altitude band.
      - accel_mean_climb : mean of finite-difference d(TAS)/dt on the same band.
      - time_to_fl240_s : row-index delta × QAR nominal step (1 s).
    """
    band_mask = np.isfinite(alt_m) & (alt_m >= _CLIMB_BAND_LOW_M) & (alt_m <= _CLIMB_BAND_HIGH_M)

    if "ATT__VV" in df.columns and band_mask.any():
        vv = df["ATT__VV"].to_numpy().astype(np.float64)
        m = band_mask & np.isfinite(vv)
        climb_rate = float(np.nanmean(vv[m])) if m.any() else 0.0
    else:
        climb_rate = 0.0

    if "SPD__TAS" in df.columns:
        tas_kt = df["SPD__TAS"].to_numpy().astype(np.float64)
        tas_ms = tas_kt * _KT_TO_MS
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
        "climb_rate_mean_climb": climb_rate,
        "accel_mean_climb": accel,
        "time_to_fl240_s": dt,
    }


def compute_qar_features(df: pl.DataFrame, seg_t0_idx: int) -> dict[str, float]:
    """Compute every MassEncoder vol-niveau feature from a QAR flight DataFrame.

    Returns all 9 candidate features (6 baseline + 3 dynamic). Consumers
    filter the dict by the architecture's ``flight_feature_cols``.

    Mirrors the ADS-B pipeline: distances via great-circle from departure point,
    cruise alt max, wind/temp/mach aggregates over the whole flight, and the
    Experiment-01 dynamic aggregates over the climb altitude band.
    """
    lat = df["NAV__LAT"].to_numpy()
    lon = df["NAV__LONG"].to_numpy()
    valid = np.isfinite(lat) & np.isfinite(lon) & (lat != 0.0) & (lon != 0.0)
    if not valid.any():
        raise ValueError("No valid NAV__LAT/LONG samples")
    first_idx = int(np.argmax(valid))
    last_idx = int(np.where(valid)[0][-1])
    seg_idx = seg_t0_idx if valid[seg_t0_idx] else first_idx

    lat0, lon0 = float(lat[first_idx]), float(lon[first_idx])
    lat_end, lon_end = float(lat[last_idx]), float(lon[last_idx])
    lat_t, lon_t = float(lat[seg_idx]), float(lon[seg_idx])

    adep_at_t0 = _haversine_m(lat0, lon0, lat_t, lon_t)
    ades_at_t0 = _haversine_m(lat_t, lon_t, lat_end, lon_end)
    dist_total = adep_at_t0 + ades_at_t0

    alt_ft = df["ALT__STD"].to_numpy().astype(np.float64)
    alt_m = alt_ft * _FT_TO_M
    cruise_alt_max_m = float(np.nanmax(alt_m))
    wind_long_mean = float(np.nanmean(df["WIND__LONG"].to_numpy()))
    temp_isa_dev_mean = float(np.nanmean(df["TEMP__DELTA_ISA"].to_numpy()))

    mach_sel = df["SPD__MACH_SEL"].to_numpy()
    in_range = (
        np.isfinite(mach_sel) & (mach_sel >= _MACH_CRUISE_MIN) & (mach_sel <= _MACH_CRUISE_MAX)
    )
    mach_cruise = float(np.max(mach_sel[in_range])) if in_range.any() else _MACH_CRUISE_FALLBACK

    base = {
        "dist_total_flight": dist_total,
        "dist_adep_at_t0": adep_at_t0,
        "cruise_alt_max_flight": cruise_alt_max_m,
        "wind_long_mean_flight": wind_long_mean,
        "temp_isa_dev_mean_flight": temp_isa_dev_mean,
        "mach_cruise_planned": mach_cruise,
    }
    base.update(_dynamic_qar_aggregates(df, alt_m))
    return base


def load_mass_encoder(model_dir: Path, device: str = "cpu") -> tuple[torch.nn.Module, list[str]]:
    """Re-instantiate a trained MassEncoder from its model directory.

    Selection priority:

    1. If the architecture spec declares ``mass_encoder_class_path``,
       instantiate that custom encoder with ``mass_encoder_kwargs``.
    2. Else if ``mass_encoder_temperature > 1``, use
       :class:`MassEncoderLinearTempered`.
    3. Else, fall back to :class:`MassEncoderLinear`.

    Returns the encoder plus the resolved ``feature_cols`` so the caller
    can build feature vectors in the right column order.
    """
    meta = json.loads((model_dir / "meta.json").read_text())
    feature_cols, feature_signs = _resolve_feature_cols(meta)
    temperature = float(meta.get("mass_encoder_temperature", 1.0) or 1.0)

    arch_spec = None
    arch_name = meta.get("architecture_name", "")
    try:
        arch_spec = get_arch_spec(arch_name)
    except KeyError:
        pass

    common_kwargs = {
        "feature_stats": meta["stats_dict"],
        "feature_cols": feature_cols,
        "expected_signs": feature_signs,
        "oew_kg": A320_OEW_KG,
        "mtow_kg": A320_MTOW_KG,
    }

    encoder: torch.nn.Module
    class_path = getattr(arch_spec, "mass_encoder_class_path", None) if arch_spec else None
    extra_kwargs = dict(getattr(arch_spec, "mass_encoder_kwargs", {}) or {}) if arch_spec else {}
    if class_path:
        module_path, class_name = class_path.rsplit(".", 1)
        encoder_cls = getattr(importlib.import_module(module_path), class_name)
        encoder = encoder_cls(**common_kwargs, **extra_kwargs).to(device)
    elif temperature == 1.0:
        encoder = MassEncoderLinear(**common_kwargs).to(device)
    else:
        encoder = MassEncoderLinearTempered(**common_kwargs, temperature=temperature).to(device)

    state = torch.load(model_dir / "mass_encoder.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(state, strict=False)
    encoder.eval()
    return encoder, feature_cols


def predict_mass(
    encoder: torch.nn.Module,
    features: dict[str, float],
    feature_cols: list[str],
    device: str = "cpu",
) -> float:
    """Pass a single feature row through the encoder and return scalar mass in kg."""
    vec = torch.tensor(
        [features[c] for c in feature_cols],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        return float(encoder(vec).item())


def _dedup_qar_files(files: list[Path]) -> list[Path]:
    """Remove duplicates by content fingerprint (flight_id, n_samples, GW_init)."""
    seen: set[tuple] = set()
    out: list[Path] = []
    for f in files:
        try:
            df = pl.read_parquet(f, columns=["SYS__GW"])
        except Exception:
            continue
        gw = df["SYS__GW"].drop_nulls()
        first_val = gw.first() if gw.len() > 0 else 0.0
        gw_init = float(first_val) if first_val is not None else 0.0  # type: ignore[arg-type]
        flight_id = f.name.split("_")[1] if "_" in f.name else f.stem
        key = (flight_id, df.shape[0], round(gw_init, 1))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qar-dir", default="/Users/gabriel/Downloads/QAR3")
    parser.add_argument("--models", nargs="+", default=["full_hybrid_v2", "full_hybrid_v3"])
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument(
        "--segment",
        choices=["t0", "cruise"],
        default="cruise",
        help="Segment start point: takeoff (t0) or first stable cruise (ALT > 25k ft)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    qar_dir = Path(args.qar_dir)
    models_dir = Path(args.models_dir)

    encoders: dict[str, tuple[torch.nn.Module, list[str]]] = {
        name: load_mass_encoder(models_dir / name, args.device) for name in args.models
    }

    qar_files = _dedup_qar_files(sorted(qar_dir.glob("*.parquet")))
    print(f"Processing {len(qar_files)} unique QAR files...")

    records: list[dict] = []
    for f in qar_files:
        try:
            df = pl.read_parquet(f)
        except Exception as exc:
            print(f"  SKIP {f.name}: {exc}")
            continue

        parts = f.name.split("_")
        flight_id = parts[1] if len(parts) > 1 else f.stem
        tail = parts[6] if len(parts) > 6 else "?"

        gw = df["SYS__GW"].to_numpy()
        gw_valid = np.where(np.isfinite(gw) & (gw > 0))[0]
        if len(gw_valid) == 0:
            print(f"  SKIP {flight_id}: no valid SYS__GW")
            continue

        if args.segment == "t0":
            seg_idx = int(gw_valid[0])
        else:
            cruise_idx = _find_stable_cruise_idx(df)
            seg_candidates = gw_valid[gw_valid >= cruise_idx]
            seg_idx = int(seg_candidates[0]) if len(seg_candidates) > 0 else int(gw_valid[0])

        m_truth = float(gw[seg_idx])

        try:
            features = compute_qar_features(df, seg_t0_idx=seg_idx)
        except ValueError as exc:
            print(f"  SKIP {flight_id}: {exc}")
            continue

        preds = {
            name: predict_mass(enc, features, feature_cols, args.device)
            for name, (enc, feature_cols) in encoders.items()
        }

        rec: dict = {
            "flight_id": flight_id,
            "tail": tail,
            "seg_idx": seg_idx,
            "n_samples": len(df),
            "m_truth_kg": m_truth,
            **features,
            **{f"m_{name}": preds[name] for name in args.models},
            **{f"err_{name}": preds[name] - m_truth for name in args.models},
        }
        records.append(rec)
        pred_summary = " ".join(f"{n}={preds[n]:.0f}" for n in args.models)
        print(f"  {flight_id}/{tail}: truth={m_truth:.0f}  {pred_summary}")

    out_dir = models_dir / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test_a_mass_encoder_qar.md"
    _write_report(records, args.models, args.segment, out_path)
    print(f"\nWrote {out_path}")


def _write_report(records: list[dict], models: list[str], segment: str, out_path: Path) -> None:
    if not records:
        out_path.write_text("# Test A — No valid records\n")
        return

    lines: list[str] = [
        "# Test A — MassEncoder QAR validation\n",
        f"> **Models compared**: {', '.join(models)}",
        f"> **Segment start**: `{segment}` "
        + (
            "(first stable cruise sample with `ALT__STD > 25 000 ft`)"
            if segment == "cruise"
            else "(takeoff, first valid `SYS__GW` sample)"
        ),
        "> **Ground truth**: `SYS__GW` (mass-and-balance derived gross weight) at segment start.",
        "",
        "## Per-flight predictions",
        "",
    ]

    headers = ["flight", "tail", "m_truth (kg)"]
    for m in models:
        headers += [f"m_{m} (kg)", f"err {m} (kg)", f"err {m} (%)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---:"] * len(headers)) + "|")
    for r in records:
        row = [r["flight_id"], r["tail"], f"{r['m_truth_kg']:.0f}"]
        for m in models:
            err = r[f"err_{m}"]
            row += [
                f"{r[f'm_{m}']:.0f}",
                f"{err:+.0f}",
                f"{err / r['m_truth_kg'] * 100:+.2f}",
            ]
        lines.append("| " + " | ".join(row) + " |")

    m_truth = np.array([r["m_truth_kg"] for r in records])
    lines += [
        "",
        "## Aggregated statistics",
        "",
        "| Model | n | bias (kg) | MAE (kg) | RMSE (kg) | bias (%) | MAE (%) | corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stats: dict[str, dict[str, float]] = {}
    for m in models:
        m_pred = np.array([r[f"m_{m}"] for r in records])
        err = m_pred - m_truth
        bias = float(err.mean())
        mae = float(np.abs(err).mean())
        rmse = float(np.sqrt((err**2).mean()))
        bias_pct = float((err / m_truth * 100).mean())
        mae_pct = float((np.abs(err) / m_truth * 100).mean())
        corr = float(np.corrcoef(m_truth, m_pred)[0, 1]) if len(m_truth) >= 3 else float("nan")
        stats[m] = {
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "corr": corr,
            "bias_pct": bias_pct,
            "mae_pct": mae_pct,
        }
        lines.append(
            f"| {m} | {len(records)} | {bias:+.0f} | {mae:.0f} | {rmse:.0f} | "
            f"{bias_pct:+.2f} | {mae_pct:.2f} | {corr:.3f} |"
        )

    lines += ["", "## Verdict", ""]
    if len(models) >= 2:
        ranked = sorted(models, key=lambda m: stats[m]["mae"])
        best = ranked[0]
        worst = ranked[-1]
        ratio = (
            stats[worst]["mae"] / stats[best]["mae"] if stats[best]["mae"] > 0 else float("inf")
        )
        if ratio < 1.1:
            verdict = (
                f"- **Both models predict similarly** (MAE {stats[best]['mae']:.0f} vs "
                f"{stats[worst]['mae']:.0f} kg, ratio {ratio:.2f}× < 1.1× → indistinguishable on this set)."
            )
        else:
            verdict = (
                f"- **{best} predicts better** than {worst} on QAR ground truth: "
                f"MAE {stats[best]['mae']:.0f} vs {stats[worst]['mae']:.0f} kg "
                f"({ratio:.2f}× ratio)."
            )
        lines.append(verdict)

        bias_signs = {m: ("+" if stats[m]["bias"] > 0 else "-") for m in models}
        lines.append(
            "- Bias sign: "
            + ", ".join(f"{m}={bias_signs[m]}{abs(stats[m]['bias_pct']):.2f}%" for m in models)
        )
        for m in models:
            sign = "overestimates" if stats[m]["bias"] > 0 else "underestimates"
            lines.append(
                f"  - {m} {sign} m_0 by {abs(stats[m]['bias']):.0f} kg "
                f"({abs(stats[m]['bias_pct']):.2f} %) on average."
            )

        corr_text = ", ".join(f"{m}={stats[m]['corr']:.3f}" for m in models)
        lines.append(f"- Correlation with truth: {corr_text}")
        lines.append(
            "- A correlation < 0.5 means the MassEncoder is largely guessing the mean; "
            "> 0.8 means features genuinely inform mass prediction."
        )

    lines.append("")
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
