"""Quick inference check: predict one flight and plot state variables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from node_fdm.predictor import NodeFDMPredictor
from node_fdm_pipeline.resolver import resolve_architecture

# --- Config ---
ARCH = "adsb"
MODEL_DIR = Path("data/models")
DELTA_PATH = Path("data/flights.delta")
STEP_S = 4.0  # grid step in seconds

info = resolve_architecture(ARCH)

# --- Load model ---
model_path = MODEL_DIR / f"{info.name}_A320"
if not model_path.exists():
    raise SystemExit(f"Model not found at {model_path}")

predictor = NodeFDMPredictor(model_path=model_path, device="cpu")

# --- Load a test flight ---
df = pl.read_delta(str(DELTA_PATH))
df = df.filter(pl.col("fdm_flag_valid"))

# Fill NaN on _sel columns (match training)
sel_cols = [c for c in df.columns if c.startswith("fdm_") and "_sel" in c]
if sel_cols:
    df = df.with_columns([pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols])

# Pick a flight from val split (we have few test flights)
val_df = df.filter(pl.col("meta_split") == "val")
flight_ids = val_df["meta_flight_id"].unique().sort().to_list()

if not flight_ids:
    raise SystemExit("No validation flights found")

# Pick the longest flight
best_fid = None
best_len = 0
for fid in flight_ids[:20]:  # check first 20
    n = val_df.filter(pl.col("meta_flight_id") == fid).shape[0]
    if n > best_len:
        best_len = n
        best_fid = fid

print(f"Flight: {best_fid} ({best_len} timesteps, {best_len * STEP_S / 60:.0f} min)")

flight_df = val_df.filter(pl.col("meta_flight_id") == best_fid).sort("raw_timestamp")

# --- Extract arrays ---
x_arr = flight_df.select(info.x_cols).to_numpy().astype(np.float32)
u_arr = flight_df.select(info.u_cols).to_numpy().astype(np.float32)
e_arr = flight_df.select(info.e0_cols).to_numpy().astype(np.float32)

# Filter to finite rows
finite_mask = (
    np.isfinite(x_arr).all(axis=1)
    & np.isfinite(u_arr).all(axis=1)
    & np.isfinite(e_arr).all(axis=1)
)
print(f"Finite rows: {finite_mask.sum()}/{len(finite_mask)}")

x_arr = x_arr[finite_mask]
u_arr = u_arr[finite_mask]
e_arr = e_arr[finite_mask]

x0 = x_arr[0]

# --- Predict ---
predictions = predictor.predict_flight(x0, u_arr, e_arr)
print(f"Prediction length: {len(list(predictions.values())[0])} steps")

# --- Plot ---
time_true = np.arange(len(x_arr)) * STEP_S / 60  # minutes
time_pred = np.arange(len(list(predictions.values())[0])) * STEP_S / 60

n_plots = len(info.x_cols) + 1  # extra subplot for alt - alt_target
fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
if n_plots == 1:
    axes = [axes]

labels = {
    "raw_alt_m": ("Altitude", "m"),
    "fdm_gamma_rad": ("Flight Path Angle", "rad"),
    "era_tas_ms": ("True Airspeed", "m/s"),
}

for i, col in enumerate(info.x_cols):
    ax = axes[i]
    label, unit = labels.get(col, (col, ""))

    # True trajectory
    ax.plot(time_true, x_arr[:, i], "k-", lw=1.5, label="True", alpha=0.8)

    # Predicted trajectory
    pred_vals = predictions[col]
    ax.plot(time_pred, pred_vals, "r--", lw=1.2, label="Predicted (Neural ODE)", alpha=0.8)

    # Y-axis limits based on true trajectory ± 10% margin
    true_vals = x_arr[:, i]
    ymin, ymax = true_vals.min(), true_vals.max()
    margin = (ymax - ymin) * 0.10 if ymax != ymin else abs(ymax) * 0.10 + 1.0
    ax.set_ylim(ymin - margin, ymax + margin)

    ax.set_ylabel(f"{label} [{unit}]")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

# --- Alt - Alt_target subplot ---
alt_idx = info.x_cols.index("raw_alt_m")
alt_target_idx = info.u_cols.index("fdm_alt_target_m")

alt_true = x_arr[:, alt_idx]
alt_target = u_arr[:, alt_target_idx]
diff_true = alt_true - alt_target

alt_pred = predictions["raw_alt_m"]
diff_pred = alt_pred - alt_target[: len(alt_pred)]

ax_diff = axes[len(info.x_cols)]
ax_diff.plot(time_true, diff_true, "k-", lw=1.5, label="True", alpha=0.8)
ax_diff.plot(time_pred, diff_pred, "r--", lw=1.2, label="Predicted (Neural ODE)", alpha=0.8)
ax_diff.axhline(0, color="gray", ls=":", lw=0.8)
ymin, ymax = diff_true.min(), diff_true.max()
margin = (ymax - ymin) * 0.10 if ymax != ymin else abs(ymax) * 0.10 + 1.0
ax_diff.set_ylim(ymin - margin, ymax + margin)
ax_diff.set_ylabel("Alt − Alt_target [m]")
ax_diff.legend(loc="best")
ax_diff.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time [min]")
fig.suptitle(f"Neural ODE Inference — {best_fid}", fontsize=14)
fig.tight_layout()

out_path = Path("data/figures/inference_check.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
