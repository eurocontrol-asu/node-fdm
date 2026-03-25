"""Quick inference check: predict one flight and plot state variables.

Loads a trained Neural-ODE model and a validation flight, runs forward
prediction, and produces a 3-column figure:

* Column 1 (Alt): altitude + target overlay, then alt_diff below.
* Column 2 (TAS): TAS + target overlay, then tas_diff below.
* Column 3 (FPA): flight-path angle + gamma_target + GammaDefaultNet output,
  then gamma_diff below.

Output: ``data/figures/inference_check.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
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

# Pick a flight from val split
val_df = df.filter(pl.col("meta_split") == "val")
flight_ids = val_df["meta_flight_id"].unique().sort().to_list()

if not flight_ids:
    raise SystemExit("No validation flights found")

# Pick the longest flight
best_fid = None
best_len = 0
for fid in flight_ids[:20]:
    n = val_df.filter(pl.col("meta_flight_id") == fid).shape[0]
    if n > best_len:
        best_len = n
        best_fid = fid

print(f"Flight: {best_fid} ({best_len} timesteps, {best_len * STEP_S / 60:.0f} min)")

flight_df = val_df.filter(pl.col("meta_flight_id") == best_fid).sort("raw_timestamp")

# --- Extract arrays ---
x_arr = flight_df.select(info.x_cols).to_numpy().astype(np.float32)
u_arr_raw = flight_df.select(info.u_cols).to_numpy().astype(np.float32)
e_arr = flight_df.select(info.e0_cols).to_numpy().astype(np.float32)

# Extract known mask before any transformation
gamma_known_idx = info.u_cols.index("fdm_gamma_target_known")
gamma_known = u_arr_raw[:, gamma_known_idx].copy()

# u_arr is already NaN-free (segments.py fills NaN→0, known mask carries the info)
u_arr = u_arr_raw

finite_mask = (
    np.isfinite(x_arr).all(axis=1)
    & np.isfinite(e_arr).all(axis=1)
)
print(f"Finite rows: {finite_mask.sum()}/{len(finite_mask)}")

x_arr = x_arr[finite_mask]
u_arr = u_arr[finite_mask]
e_arr = e_arr[finite_mask]
gamma_known = gamma_known[finite_mask]

x0 = x_arr[0]

# --- Load GammaDefaultNet from trajectory checkpoint ---
traj_layer = predictor.model.layers_dict["trajectory"]
gamma_net = traj_layer.gamma_default_net

traj_ckpt_path = model_path / "trajectory.pt"
if traj_ckpt_path.exists():
    traj_ckpt = torch.load(traj_ckpt_path, weights_only=True, map_location="cpu")
    traj_layer.load_state_dict(traj_ckpt["layer_state"])
    print("Loaded GammaDefaultNet weights from trajectory.pt")
else:
    print("No trajectory.pt found, using initial (zero-init) GammaDefaultNet")

# --- Predict ---
predictions = predictor.predict_flight(x0, u_arr, e_arr)
print(f"Prediction length: {len(list(predictions.values())[0])} steps")

# --- Time axes ---
time_true = np.arange(len(x_arr)) * STEP_S / 60  # minutes
time_pred = np.arange(len(list(predictions.values())[0])) * STEP_S / 60

# --- Extract state variables ---
alt_idx = info.x_cols.index("raw_alt_m")
tas_idx = info.x_cols.index("era_tas_ms")
gamma_idx = info.x_cols.index("fdm_gamma_rad")

alt_true = x_arr[:, alt_idx]
tas_true = x_arr[:, tas_idx]
gamma_true = x_arr[:, gamma_idx]

alt_pred = predictions["raw_alt_m"]
tas_pred = predictions["era_tas_ms"]
gamma_pred = predictions["fdm_gamma_rad"]

# --- Extract targets from U_COLS ---
alt_target_idx = info.u_cols.index("fdm_alt_target_m")
tas_target_idx = info.u_cols.index("fdm_tas_target_ms")
gamma_target_idx = info.u_cols.index("fdm_gamma_target_rad")

alt_target = u_arr[:, alt_target_idx]
tas_target = u_arr[:, tas_target_idx]
# Compute per-timestep gamma_default from the net on TRUE trajectory
gamma_target_raw = u_arr[:, gamma_target_idx]
vz_true = tas_true * np.sin(gamma_true)
with torch.no_grad():
    gamma_default_true = gamma_net(
        torch.from_numpy(alt_true),
        torch.from_numpy(tas_true),
        torch.from_numpy(vz_true),
    ).numpy()

# Effective gamma target: real target where known, net output where unknown
gamma_target_effective = np.where(gamma_known == 1.0, gamma_target_raw, gamma_default_true)
# For display: show real target (blue) and net default (cyan) separately
gamma_target = np.where(gamma_known == 1.0, gamma_target_raw, np.nan)  # gaps where unknown

# Also compute net output on PREDICTED trajectory (for the diff subplot)
n_pred = len(gamma_pred)
vz_pred = tas_pred * np.sin(gamma_pred)
with torch.no_grad():
    gamma_default_pred = gamma_net(
        torch.from_numpy(alt_pred.astype(np.float32)),
        torch.from_numpy(tas_pred.astype(np.float32)),
        torch.from_numpy(vz_pred.astype(np.float32)),
    ).numpy()

# Stats for display
gd_mean = np.degrees(gamma_default_true.mean())
gd_std = np.degrees(gamma_default_true.std())
print(f"GammaDefaultNet output (on true traj): mean={gd_mean:.2f}°, std={gd_std:.2f}°")


# --- Helper ---
def _set_ylim(ax, true_vals):
    valid = true_vals[np.isfinite(true_vals)]
    if len(valid) == 0:
        return
    ymin, ymax = valid.min(), valid.max()
    margin = (ymax - ymin) * 0.10 if ymax != ymin else abs(ymax) * 0.10 + 1.0
    ax.set_ylim(ymin - margin, ymax + margin)


# --- Figure: 3 columns × 2 rows ---
fig, axes = plt.subplots(2, 3, figsize=(20, 8), sharex=True)

# ── Col 1, Row 0: Altitude + target ──
ax = axes[0, 0]
ax.plot(time_true, alt_true, "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, alt_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, alt_target, "b-", lw=2.0, label="Target", alpha=0.4)
_set_ylim(ax, alt_true)
ax.set_ylabel("Altitude [m]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Col 1, Row 1: Alt diff ──
ax = axes[1, 0]
diff_alt_true = alt_target - alt_true
diff_alt_pred = alt_target[: len(alt_pred)] - alt_pred
ax.plot(time_true, diff_alt_true, "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, diff_alt_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.axhline(0, color="gray", ls=":", lw=0.8)
_set_ylim(ax, diff_alt_true)
ax.set_ylabel("Alt_target − Alt [m]")
ax.set_xlabel("Time [min]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Col 2, Row 0: TAS + target ──
ax = axes[0, 1]
ax.plot(time_true, tas_true, "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, tas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, tas_target, "b-", lw=2.0, label="Target", alpha=0.4)
_set_ylim(ax, tas_true)
ax.set_ylabel("TAS [m/s]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Col 2, Row 1: TAS diff ──
ax = axes[1, 1]
diff_tas_true = tas_target - tas_true
diff_tas_pred = tas_target[: len(tas_pred)] - tas_pred
ax.plot(time_true, diff_tas_true, "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, diff_tas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.axhline(0, color="gray", ls=":", lw=0.8)
_set_ylim(ax, diff_tas_true)
ax.set_ylabel("TAS_target − TAS [m/s]")
ax.set_xlabel("Time [min]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Col 3, Row 0: Flight-path angle + gamma target + GammaDefaultNet ──
ax = axes[0, 2]
ax.plot(time_true, np.degrees(gamma_true), "k-", lw=0.8, label="True", alpha=0.5)
ax.plot(time_pred, np.degrees(gamma_pred), "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, np.degrees(gamma_target), "b-", lw=3.0, label="γ target (known)", alpha=0.9)
# Show GammaDefaultNet output where unknown (per-timestep, not flat)
gamma_default_line = np.where(gamma_known == 0.0, gamma_default_true, np.nan)
ax.plot(
    time_true,
    np.degrees(gamma_default_line),
    "c-",
    lw=1.5,
    label=f"γ net default (μ={gd_mean:.1f}°)",
    alpha=0.7,
)
_set_ylim(ax, np.degrees(gamma_true))
ax.set_ylabel("FPA [°]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Col 3, Row 1: Gamma diff ──
ax = axes[1, 2]
# True diff uses net output on true trajectory where unknown
diff_gamma_true = gamma_target_effective - gamma_true
# Predicted diff uses net output on predicted trajectory where unknown
gamma_target_eff_pred = np.where(
    gamma_known[:n_pred] == 1.0,
    gamma_target_raw[:n_pred],
    gamma_default_pred,
)
diff_gamma_pred = gamma_target_eff_pred - gamma_pred
ax.plot(time_true, np.degrees(diff_gamma_true), "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, np.degrees(diff_gamma_pred), "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.axhline(0, color="gray", ls=":", lw=0.8)
_set_ylim(ax, np.degrees(diff_gamma_true))
ax.set_ylabel("γ_target − γ [°]")
ax.set_xlabel("Time [min]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(f"Neural ODE Inference — {best_fid}", fontsize=14)
fig.tight_layout()

out_path = Path("data/figures/inference_check.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
