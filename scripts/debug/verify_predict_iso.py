"""Verify that ``fdm predict`` produces the same arrays as ``check_inference.py``.

Runs both data-loading / preprocessing paths on the same flight and
compares:
  1. The arrays fed to the predictor (x0, u, e).
  2. The raw prediction dict returned by ``predict_flight``.

Usage:
    uv run python -m scripts.debug.verify_predict_iso [flight_id] [model_name]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

from node_fdm.predictor import NodeFDMPredictor
from node_fdm_pipeline.resolver import resolve_architecture

ARCH = "adsb"
MODEL_DIR = Path("data/models")
DELTA_PATH = Path("data/flights.delta")

info = resolve_architecture(ARCH)

cli_model_name = (
    sys.argv[2] if len(sys.argv) > 2
    else os.environ.get("CHECK_INFERENCE_MODEL", f"{info.name}_A320")
)
model_path = MODEL_DIR / cli_model_name
if not model_path.exists():
    raise SystemExit(f"Model not found at {model_path}")

predictor = NodeFDMPredictor(model_path=model_path, device="cpu")

# ── PATH A: check_inference.py logic (reference) ──────────────────────
df_a = pl.read_delta(str(DELTA_PATH)).sort("meta_flight_id", "raw_timestamp")

sel_cols = [
    c for c in df_a.columns
    if c.startswith("fdm_") and "_sel" in c and df_a.schema[c].is_numeric()
]
if sel_cols:
    df_a = df_a.with_columns([pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols])

df_a = df_a.with_columns([
    pl.col("fdm_alt_target_m").fill_null(strategy="backward").fill_null(strategy="forward"),
    pl.col("fdm_gamma_target_rad").fill_nan(0.0).fill_null(0.0),
    pl.col("fdm_tas_target_ms").fill_nan(0.0).fill_null(0.0),
    pl.col("fdm_heading_target_rad")
    .fill_nan(None).fill_null(strategy="forward").over("meta_flight_id")
    .fill_null(strategy="backward").over("meta_flight_id"),
    pl.col("fdm_gamma_target_known").fill_nan(0.0).fill_null(0.0),
    pl.col("fdm_tas_target_known").fill_null(False),
    pl.lit(True).alias("fdm_heading_target_known"),
])

_state_exo_cols = [
    c for c in df_a.columns
    if (c.startswith("raw_") or c.startswith("era_") or c.startswith("fdm_"))
    and df_a.schema[c].is_numeric()
]
df_a = df_a.with_columns([
    pl.col(c).fill_nan(None)
    .fill_null(strategy="forward").over("meta_flight_id")
    .fill_null(strategy="backward").over("meta_flight_id")
    for c in _state_exo_cols
])

val_df = df_a.filter(
    pl.col("meta_split").is_in(["val", "test"]) & pl.col("fdm_flag_valid")
)
flight_ids = val_df["meta_flight_id"].unique().sort().to_list()
if not flight_ids:
    raise SystemExit("No validation flights found")

test_flight_ids = (
    df_a.filter(pl.col("meta_split").eq("test") & pl.col("fdm_flag_valid"))
    ["meta_flight_id"].unique().sort().to_list()
)

cli_fid = sys.argv[1] if len(sys.argv) > 1 else None
if cli_fid is not None:
    if cli_fid not in flight_ids:
        raise SystemExit(f"Flight {cli_fid!r} not in val/test split")
    fid = cli_fid
else:
    if not test_flight_ids:
        raise SystemExit("No test-split flights found")
    fid = test_flight_ids[0]

flight_full_a = df_a.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
crop_start_a = int(flight_full_a["fdm_flag_crop_start"][0])
crop_end_a = int(flight_full_a["fdm_flag_crop_end"][0])
flight_a = flight_full_a.slice(crop_start_a, crop_end_a - crop_start_a + 1)

x_a = flight_a.select(info.x_cols).to_numpy().astype(np.float32)
u_a = flight_a.select(info.u_cols).to_numpy().astype(np.float32)
e_a = flight_a.select(info.e0_cols).to_numpy().astype(np.float32)
x0_a = x_a[0]

pred_a = predictor.predict_flight(x0_a, u_a, e_a)

# ── PATH B: predict.py pipeline logic ─────────────────────────────────
from node_fdm_pipeline.commands.predict import _load_test_df

df_b = _load_test_df(DELTA_PATH)

flight_full_b = df_b.filter(pl.col("meta_flight_id") == fid).sort("raw_timestamp")
if len(flight_full_b) == 0:
    # fid might be in val split only — check_inference allows val+test,
    # _load_test_df only keeps test.  Re-check.
    raise SystemExit(
        f"Flight {fid!r} not found in _load_test_df output "
        "(it may be in 'val' split only — pick a 'test' flight)."
    )

crop_start_b = int(flight_full_b["fdm_flag_crop_start"][0])
crop_end_b = int(flight_full_b["fdm_flag_crop_end"][0])
flight_b = flight_full_b.slice(crop_start_b, crop_end_b - crop_start_b + 1)

x_b = flight_b.select(info.x_cols).to_numpy().astype(np.float32)
u_b = flight_b.select(info.u_cols).to_numpy().astype(np.float32)
e_b = flight_b.select(info.e0_cols).to_numpy().astype(np.float32)
x0_b = x_b[0]

pred_b = predictor.predict_flight(x0_b, u_b, e_b)

# ── Compare ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Flight: {fid}   Model: {cli_model_name}")
print(f"{'='*60}")

print(f"\nCrop indices  A: [{crop_start_a}, {crop_end_a}]  B: [{crop_start_b}, {crop_end_b}]")
assert crop_start_a == crop_start_b, "crop_start mismatch"
assert crop_end_a == crop_end_b, "crop_end mismatch"

print(f"Cropped shape A: {flight_a.height}  B: {flight_b.height}")
assert flight_a.height == flight_b.height, "cropped height mismatch"

ok = True

def compare(name: str, a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape {a.shape} vs {b.shape}")
        return False
    if np.array_equal(a, b, equal_nan=True):
        print(f"  OK   {name}: identical")
        return True
    diff = np.abs(a - b)
    max_diff = np.nanmax(diff)
    mean_diff = np.nanmean(diff)
    n_diff = int((diff > 0).sum())
    print(f"  DIFF {name}: max={max_diff:.2e}  mean={mean_diff:.2e}  n_diff={n_diff}/{a.size}")
    return max_diff < 1e-6

print("\n── Input arrays ──")
ok &= compare("x0", x0_a, x0_b)
ok &= compare("x_arr", x_a, x_b)
ok &= compare("u_arr", u_a, u_b)
ok &= compare("e_arr", e_a, e_b)

print("\n── Predictions ──")
keys_a = sorted(pred_a.keys())
keys_b = sorted(pred_b.keys())
if keys_a != keys_b:
    print(f"  FAIL prediction keys differ: {keys_a} vs {keys_b}")
    ok = False
else:
    for k in keys_a:
        va = np.asarray(pred_a[k], dtype=np.float64)
        vb = np.asarray(pred_b[k], dtype=np.float64)
        ok &= compare(f"pred[{k}]", va, vb)

print(f"\n{'='*60}")
if ok:
    print("RESULT: PASS — predict.py is iso with check_inference.py")
else:
    print("RESULT: FAIL — divergences detected (see above)")
print(f"{'='*60}")
sys.exit(0 if ok else 1)
