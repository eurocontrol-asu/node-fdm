"""Quick inference check: predict one flight and plot state variables.

Loads a trained Neural-ODE model and a validation flight, runs forward
prediction, and produces a 3-column figure:

* Column 1 (Alt): altitude + target overlay, then alt_diff below.
* Column 2 (TAS): TAS + target overlay, then tas_diff below.
* Column 3 (FPA): flight-path angle + gamma_target + GammaDefaultNet output,
  then gamma_diff below.
* Row 3: Mach, CAS, VZ — true + predicted + sel target, with gray shading
  on rows where the target/segment is absent.

Output: ``data/figures/inference_check.png``.
"""

from __future__ import annotations

import sys
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

# Pick a flight from val split
val_df = df.filter(pl.col("meta_split") == "val")
flight_ids = val_df["meta_flight_id"].unique().sort().to_list()

if not flight_ids:
    raise SystemExit("No validation flights found")

# CLI override: `python check_inference.py <flight_id>` to target a specific vol
cli_fid = sys.argv[1] if len(sys.argv) > 1 else None
if cli_fid is not None:
    if cli_fid not in flight_ids:
        raise SystemExit(f"Flight {cli_fid!r} not in val split. Available: {flight_ids[:20]}")
    best_fid = cli_fid
    best_len = val_df.filter(pl.col("meta_flight_id") == best_fid).shape[0]
else:
    # Default: pick the longest flight from the first 20
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
# gamma_diff = known * (target - gamma), 0 when unknown
gamma_target_raw = u_arr[:, gamma_target_idx]
gamma_target_effective = np.where(gamma_known == 1.0, gamma_target_raw, 0.0)
gamma_target = np.where(gamma_known == 1.0, gamma_target_raw, np.nan)  # gaps where unknown
n_pred = len(gamma_pred)

# Stats for display
pct_known = gamma_known.mean() * 100
print(f"Gamma target: {pct_known:.0f}% known, {100 - pct_known:.0f}% unknown (gamma_diff=0)")

# --- Extract Mach / CAS / VZ (true + targets) from the raw flight DataFrame ---
from node_fdm_data.physics.constants import GAMMA_AIR, R
from node_fdm_data.physics.speed import tas_to_cas_real

KT_TO_MS = 0.514444
FTMIN_TO_MS = 0.00508

extra_cols = [
    "era_mach",
    "fdm_mach_sel",
    "bds_ias_ms",
    "fdm_cas_sel_kt",
    "raw_vz_ms",
    "fdm_vz_sel_ms",
    "fdm_tas_target_known",
    "era_temp_K",
]
extra = flight_df.select(extra_cols).to_numpy().astype(np.float32)[finite_mask]

mach_true = extra[:, 0]
mach_sel = extra[:, 1]  # NaN outside detected segments
cas_true = extra[:, 2]  # IAS Mode-S clean — same source as the detected plateau
cas_sel = extra[:, 3] * KT_TO_MS  # plateau detected on bds_ias_kt_clean → m/s
vz_true = extra[:, 4]
vz_sel = extra[:, 5]  # NaN outside detected segments
tas_known = extra[:, 6]  # 1.0 where Mach/CAS envelope yields a target
temp_true = extra[:, 7]  # ERA5 real temperature [K] — used for round-trip closure

# Unknown masks for shading "target absent" regions in gray
mach_unknown = np.isnan(mach_sel)
cas_unknown = np.isnan(cas_sel)
vz_unknown = np.isnan(vz_sel)
tas_unknown = tas_known == 0.0

# --- Derive predicted Mach / CAS / VZ from predicted state (alt, tas, gamma) ---
# Mach_pred = tas_pred / sqrt(γ·R·T_real)         — using real ERA5 temperature
# CAS_pred  = tas_to_cas_real(tas_pred, alt_pred, T_real)
# VZ_pred   = tas_pred * sin(gamma_pred)
# Use real temperature (era_temp_K) so the round-trip TAS→CAS via real T closes
# back onto bds_ias_ms; otherwise an ISA-only conversion biases CAS by ~2 m/s
# at FL350 because era_tas_ms itself was derived via cas_to_tas_real (real T).
mach_pred = tas_pred / np.sqrt(GAMMA_AIR * R * temp_true)
cas_pred = tas_to_cas_real(tas_pred, alt_pred, temp_true)
vz_pred = tas_pred * np.sin(gamma_pred)


# --- Helpers ---
def _set_ylim(ax, true_vals):
    valid = true_vals[np.isfinite(true_vals)]
    if len(valid) == 0:
        return
    ymin, ymax = valid.min(), valid.max()
    margin = (ymax - ymin) * 0.10 if ymax != ymin else abs(ymax) * 0.10 + 1.0
    ax.set_ylim(ymin - margin, ymax + margin)


def _shade_unknown(ax, t, mask, label):
    """Shade rows where target is absent (unknown / no detected segment)."""
    if not mask.any():
        return
    ymin, ymax = ax.get_ylim()
    ax.fill_between(t, ymin, ymax, where=mask, alpha=0.08, color="gray", label=label)
    ax.set_ylim(ymin, ymax)  # fill_between can shift ylim; clamp back


# --- Figure: 3 columns × 3 rows ---
fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True)

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
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Col 2, Row 0: TAS + target ──
ax = axes[0, 1]
ax.plot(time_true, tas_true, "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, tas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, tas_target, "b-", lw=2.0, label="Target", alpha=0.4)
_set_ylim(ax, tas_true)
_shade_unknown(ax, time_true, tas_unknown, "TAS unknown")
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
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Col 3, Row 0: Flight-path angle + gamma target + GammaDefaultNet ──
ax = axes[0, 2]
ax.plot(time_true, np.degrees(gamma_true), "k-", lw=0.8, label="True", alpha=0.5)
ax.plot(time_pred, np.degrees(gamma_pred), "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, np.degrees(gamma_target), "b-", lw=3.0, label="γ target (known)", alpha=0.9)
# Show unknown regions as shaded
unknown_mask = gamma_known == 0.0
if unknown_mask.any():
    ax.fill_between(time_true, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -10, 10,
                     where=unknown_mask, alpha=0.08, color="gray", label="γ unknown")
_set_ylim(ax, np.degrees(gamma_true))
ax.set_ylabel("FPA [°]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Col 3, Row 1: Gamma diff ──
ax = axes[1, 2]
# Only show diff where gamma target is known; 0 otherwise
diff_gamma_true = np.where(gamma_known == 1.0, gamma_target_raw - gamma_true, 0.0)
diff_gamma_pred = np.where(
    gamma_known[:n_pred] == 1.0,
    gamma_target_raw[:n_pred] - gamma_pred,
    0.0,
)
ax.plot(time_true, np.degrees(diff_gamma_true), "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, np.degrees(diff_gamma_pred), "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.axhline(0, color="gray", ls=":", lw=0.8)
_set_ylim(ax, np.degrees(diff_gamma_true))
ax.set_ylabel("γ_target − γ [°]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 2: Mach / CAS / VZ — true + predicted + target ──
ax = axes[2, 0]
ax.plot(time_true, mach_true, "k-", lw=1.5, label="True (era_mach)", alpha=0.8)
ax.plot(time_pred, mach_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, mach_sel, "b-", lw=2.0, label="Mach target (sel)", alpha=0.4)
_set_ylim(ax, mach_true)
_shade_unknown(ax, time_true, mach_unknown, "Mach unknown")
ax.set_ylabel("Mach [-]")
ax.set_xlabel("Time [min]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
ax.plot(time_true, cas_true, "k-", lw=1.5, label="True (bds_ias_ms)", alpha=0.8)
ax.plot(time_pred, cas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, cas_sel, "b-", lw=2.0, label="CAS target (sel)", alpha=0.4)
_set_ylim(ax, cas_true)
_shade_unknown(ax, time_true, cas_unknown, "CAS unknown")
ax.set_ylabel("CAS [m/s]")
ax.set_xlabel("Time [min]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[2, 2]
ax.plot(time_true, vz_true, "k-", lw=1.5, label="True (raw_vz_ms)", alpha=0.8)
ax.plot(time_pred, vz_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, vz_sel, "b-", lw=2.0, label="VZ target (sel)", alpha=0.4)
ax.axhline(0, color="gray", ls=":", lw=0.8)
_set_ylim(ax, vz_true)
_shade_unknown(ax, time_true, vz_unknown, "VZ unknown")
ax.set_ylabel("VZ [m/s]")
ax.set_xlabel("Time [min]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(f"Neural ODE Inference — {best_fid}", fontsize=14)
fig.tight_layout()

out_path = Path(f"data/figures/inference_check_{best_fid}.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
