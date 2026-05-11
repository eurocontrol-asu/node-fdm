"""Quick inference check: predict one flight and plot state variables.

Loads a trained Neural-ODE model and a validation flight, runs forward
prediction, and produces a 4-column figure:

* Column 1 (Alt): altitude + target overlay, then alt_diff below.
* Column 2 (TAS): TAS + target overlay, then tas_diff below.
* Column 3 (FPA): flight-path angle + gamma_target + GammaDefaultNet output,
  then gamma_diff below.
* Column 4 (Heading): heading + heading_target overlay, then signed
  heading_diff below.  Plotted only when the architecture exposes
  ``fdm_heading_rad`` (lateral channel).
* Row 3: Mach, CAS, VZ — true + predicted + sel target, with gray shading
  on rows where the target/segment is absent.

Output: ``data/figures/inference_check_{flight_id}.png``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
# Usage: python check_inference.py [flight_id] [model_name]
# Default model_name = f"{info.name}_A320"
cli_model_name = (
    sys.argv[2] if len(sys.argv) > 2
    else os.environ.get("CHECK_INFERENCE_MODEL", f"{info.name}_A320")
)
model_path = MODEL_DIR / cli_model_name
if not model_path.exists():
    raise SystemExit(f"Model not found at {model_path}")
print(f"Model: {cli_model_name}")

predictor = NodeFDMPredictor(model_path=model_path, device="cpu")

# --- Load a test flight ---
# NOTE: do NOT pre-filter on fdm_flag_valid — keep the full timeline so gaps and
# invalid rows remain visible. The predictor needs finite contiguous arrays, but
# the targets are bfilled (alt) / handled by *_known masks (gamma, tas) and the
# `_sel` columns are NaN→0-filled below, so the full row set works directly.
df = pl.read_delta(str(DELTA_PATH)).sort("meta_flight_id", "raw_timestamp")

# Fill NaN on _sel columns (match training)
sel_cols = [
    c
    for c in df.columns
    if c.startswith("fdm_") and "_sel" in c and df.schema[c].is_numeric()
]
if sel_cols:
    df = df.with_columns([pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols])

# Fill rules for U_COLS so the predictor never sees NaN in inputs/targets even
# on rows where fdm_flag_valid=False:
#   - fdm_alt_target_m       → backward/forward fill (continuous trajectory target)
#   - fdm_*_target_rad / _ms → fill 0 (gated off by *_known mask)
#   - fdm_*_target_known     → fill 0 / False (treat unknown as "no target")
df = df.with_columns(
    [
        pl.col("fdm_alt_target_m")
        .fill_null(strategy="backward")
        .fill_null(strategy="forward"),
        pl.col("fdm_gamma_target_rad").fill_nan(0.0).fill_null(0.0),
        pl.col("fdm_tas_target_ms").fill_nan(0.0).fill_null(0.0),
        # heading target: ffill+bfill within each flight so the lateral head
        # always has a smooth target through originally-unknown segments
        # (in-turn, head/tail). Combined with heading_target_known forced to
        # True below, the model gets a continuous correction signal and stops
        # drifting freely on those segments.
        pl.col("fdm_heading_target_rad")
        .fill_nan(None)
        .fill_null(strategy="forward")
        .over("meta_flight_id")
        .fill_null(strategy="backward")
        .over("meta_flight_id"),
        pl.col("fdm_gamma_target_known").fill_nan(0.0).fill_null(0.0),
        pl.col("fdm_tas_target_known").fill_null(False),
        # Force heading_target_known=True after the ffill above so the lateral
        # head always has a signal (otherwise heading drifts freely on unknown
        # segments — turns, head/tail — even though the ffilled target is sane).
        pl.lit(True).alias("fdm_heading_target_known"),
    ]
)

# ffill+bfill on state (raw_*, era_*, fdm_*_rad) and exo (era_*) numerical
# columns so the predictor sees a continuous timeline. We do this ONLY here in
# the inference debug script — the pipeline never modifies x_cols this way.
_state_exo_cols = [
    c
    for c in df.columns
    if (c.startswith("raw_") or c.startswith("era_") or c.startswith("fdm_"))
    and df.schema[c].is_numeric()
]
df = df.with_columns(
    [
        pl.col(c)
        .fill_nan(None)
        .fill_null(strategy="forward")
        .over("meta_flight_id")
        .fill_null(strategy="backward")
        .over("meta_flight_id")
        for c in _state_exo_cols
    ]
)

## Pick a flight from val or test split (use only valid rows to build the candidate set,
# but keep the FULL row set per flight downstream so plots show the entire timeline).
val_df = df.filter(
    pl.col("meta_split").is_in(["val", "test"]) & pl.col("fdm_flag_valid")
)
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
    # Default: pick a random flight from the val/test split.
    import random
    seed_env = os.environ.get("CHECK_INFERENCE_SEED")
    if seed_env is not None:
        random.seed(int(seed_env))
    best_fid = random.choice(flight_ids)
    best_len = val_df.filter(pl.col("meta_flight_id") == best_fid).shape[0]

print(f"Flight: {best_fid} ({best_len} timesteps, {best_len * STEP_S / 60:.0f} min)")

# Pull the FULL row set, then TRIM head/tail. fdm_flag_crop_start/end is based
# only on temporal jumps and misses leading/trailing NaN-filled rows produced by
# the resampler (e.g. when the flight starts before useful ADS-B coverage). We
# trim from each end up to the first/last row that has both a valid raw_alt_ft
# AND raw_gs_kt above min_speed_kt — equivalent to combining
# fdm_flag_crop_* with fdm_flag_min_speed at the endpoints only (mid-flight
# rows are NEVER dropped here, so anomalies stay visible).
MIN_SPEED_KT = 50.0
# Compute trim window on the RAW (pre-fill) delta so leading/trailing NaN rows
# from the resampler are visible — once we apply ffill/bfill above, those NaN
# are propagated away and we can no longer detect the original useful range.
_raw = (
    pl.read_delta(str(DELTA_PATH))
    .filter(pl.col("meta_flight_id") == best_fid)
    .sort("raw_timestamp")
)
_ok = (
    _raw["raw_alt_ft"].is_not_null()
    & _raw["raw_alt_ft"].is_not_nan()
    & _raw["raw_gs_kt"].is_not_null()
    & _raw["raw_gs_kt"].is_not_nan()
    & (_raw["raw_gs_kt"] > MIN_SPEED_KT)
).to_numpy()
_idx = np.where(_ok)[0]
if len(_idx) == 0:
    raise SystemExit(f"No usable row for flight {best_fid}")
trim_start, trim_end = int(_idx[0]), int(_idx[-1])
flight_df_full = df.filter(pl.col("meta_flight_id") == best_fid).sort("raw_timestamp")
flight_df = flight_df_full.slice(trim_start, trim_end - trim_start + 1)
print(
    f"Full rows: {flight_df_full.height}  trimmed to [{trim_start},{trim_end}] → "
    f"{flight_df.height} ({flight_df.height * STEP_S / 60:.1f} min)"
)

# --- Extract arrays for the predictor (cropped row set, fill rules applied above) ---
x_arr = flight_df.select(info.x_cols).to_numpy().astype(np.float32)
u_arr_raw = flight_df.select(info.u_cols).to_numpy().astype(np.float32)
e_arr = flight_df.select(info.e0_cols).to_numpy().astype(np.float32)

gamma_known_idx = info.u_cols.index("fdm_gamma_target_known")
gamma_known = u_arr_raw[:, gamma_known_idx].copy()
u_arr = u_arr_raw

# After ffill+bfill above, x/u/e should be NaN-free; assert.
nan_inside = int(
    (
        ~(
            np.isfinite(x_arr).all(axis=1)
            & np.isfinite(u_arr).all(axis=1)
            & np.isfinite(e_arr).all(axis=1)
        )
    ).sum()
)
print(f"NaN remaining inside crop: {nan_inside}")
if nan_inside:
    raise SystemExit(
        f"NaN still present in {nan_inside} rows after ffill/bfill — investigate."
    )

x0 = x_arr[0]

# Timestamps for predicted curves (= same grid as true/target now).
ts_full = flight_df["raw_timestamp"].to_numpy()
t0_ts = ts_full[0]
time_pred_full = (ts_full - t0_ts).astype("timedelta64[s]").astype(float) / 60.0

# --- Predict ---
predictions = predictor.predict_flight(x0, u_arr, e_arr)
print(f"Prediction length: {len(list(predictions.values())[0])} steps")

# --- Time axes ---
# `time_true` covers the FULL flight (incl. invalid rows) → reveals gaps & outliers.
# `time_pred` covers only the valid+finite subset fed to the predictor.
time_true = (ts_full - t0_ts).astype("timedelta64[s]").astype(float) / 60.0
time_pred = time_pred_full[: len(list(predictions.values())[0])]

# --- Extract TRUE state from flight_df (full timeline) ---
x_full = flight_df.select(info.x_cols).to_numpy().astype(np.float32)
u_full = flight_df.select(info.u_cols).to_numpy().astype(np.float32)

alt_idx = info.x_cols.index("raw_alt_m")
tas_idx = info.x_cols.index("era_tas_ms")
gamma_idx = info.x_cols.index("fdm_gamma_rad")

alt_true = x_full[:, alt_idx]
tas_true = x_full[:, tas_idx]
gamma_true = x_full[:, gamma_idx]

alt_pred = predictions["raw_alt_m"]
tas_pred = predictions["era_tas_ms"]
gamma_pred = predictions["fdm_gamma_rad"]

# --- Lateral channel (Phase 2B): present when fdm_heading_rad is in x_cols ---
HAS_LATERAL = "fdm_heading_rad" in info.x_cols
heading_true: np.ndarray | None = None
heading_pred: np.ndarray | None = None
heading_target: np.ndarray | None = None
heading_target_known: np.ndarray | None = None
if HAS_LATERAL:
    heading_idx = info.x_cols.index("fdm_heading_rad")
    heading_true = x_full[:, heading_idx]
    heading_pred = predictions["fdm_heading_rad"]

    h_target_idx = info.u_cols.index("fdm_heading_target_rad")
    h_known_idx = info.u_cols.index("fdm_heading_target_known")
    heading_target = u_full[:, h_target_idx]
    heading_target_known = u_full[:, h_known_idx]

# --- Extract targets from U_COLS (full timeline) ---
alt_target_idx = info.u_cols.index("fdm_alt_target_m")
tas_target_idx = info.u_cols.index("fdm_tas_target_ms")
gamma_target_idx = info.u_cols.index("fdm_gamma_target_rad")

alt_target = u_full[:, alt_target_idx]
tas_target = u_full[:, tas_target_idx]

gamma_known_full = u_full[:, info.u_cols.index("fdm_gamma_target_known")]
gamma_target_raw_full = u_full[:, gamma_target_idx]
gamma_target = np.where(gamma_known_full == 1.0, gamma_target_raw_full, np.nan)
n_pred = len(gamma_pred)

# Stats for display
pct_known = gamma_known.mean() * 100
print(f"Gamma target: {pct_known:.0f}% known, {100 - pct_known:.0f}% unknown (gamma_diff=0)")
if HAS_LATERAL and heading_target_known is not None:
    pct_h = heading_target_known.mean() * 100
    print(f"Heading target: {pct_h:.0f}% known, {100 - pct_h:.0f}% unknown")

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
if HAS_LATERAL:
    extra_cols += ["raw_lat_deg", "raw_lon_deg", "fdm_in_turn"]
extra = flight_df.select(extra_cols).to_numpy().astype(np.float32)

mach_true = extra[:, 0]
mach_sel = extra[:, 1]  # NaN outside detected segments
cas_true = extra[:, 2]  # IAS Mode-S clean — same source as the detected plateau
cas_sel = extra[:, 3] * KT_TO_MS  # plateau detected on bds_ias_kt_clean → m/s
vz_true = extra[:, 4]
vz_sel = extra[:, 5]  # NaN outside detected segments
tas_known = extra[:, 6]  # 1.0 where Mach/CAS envelope yields a target
temp_true = extra[:, 7]  # ERA5 real temperature [K] — used for round-trip closure

if HAS_LATERAL:
    lat_arr = extra[:, 8]
    lon_arr = extra[:, 9]
    in_turn_arr = extra[:, 10].astype(bool)

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
# flight_df is now cropped; temp_true and predicted curves share the same grid.
mach_pred = tas_pred / np.sqrt(GAMMA_AIR * R * temp_true)
cas_pred = tas_to_cas_real(tas_pred, alt_pred, temp_true)
vz_pred = tas_pred * np.sin(gamma_pred)

# --- Predicted ground track (lat/lon) by Euler integration of the wind triangle ---
# air_velocity = TAS_horiz * (sin(heading), cos(heading))   (east, north)
# ground_velocity = air_velocity + wind                      (wind in (u,v) = (east,north))
# Then integrate step-by-step from (lat0, lon0) ground truth.
lat_pred: np.ndarray | None = None
lon_pred: np.ndarray | None = None
if HAS_LATERAL and heading_pred is not None:
    R_EARTH_M = 6_371_000.0
    u_wind_idx = info.e0_cols.index("era_u_wind_ms")
    v_wind_idx = info.e0_cols.index("era_v_wind_ms")
    u_wind_arr = e_arr[:, u_wind_idx]
    v_wind_arr = e_arr[:, v_wind_idx]
    n_p = len(heading_pred)
    tas_horiz = tas_pred * np.cos(gamma_pred)
    v_e = tas_horiz * np.sin(heading_pred) + u_wind_arr[:n_p]
    v_n = tas_horiz * np.cos(heading_pred) + v_wind_arr[:n_p]
    lat_pred = np.empty(n_p, dtype=np.float64)
    lon_pred = np.empty(n_p, dtype=np.float64)
    lat_pred[0] = lat_arr[0]
    lon_pred[0] = lon_arr[0]
    for k in range(1, n_p):
        lat_rad_k = np.radians(lat_pred[k - 1])
        d_lat_deg = np.degrees(v_n[k - 1] * STEP_S / R_EARTH_M)
        d_lon_deg = np.degrees(
            v_e[k - 1] * STEP_S / (R_EARTH_M * max(np.cos(lat_rad_k), 1e-6))
        )
        lat_pred[k] = lat_pred[k - 1] + d_lat_deg
        lon_pred[k] = lon_pred[k - 1] + d_lon_deg


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


# --- Figure: 4 rows × 2 cols, priority order ---
# Row 0: Heading | Alt
# Row 1: TAS     | FPA (γ)
# Row 2: GroundT | VZ
# Row 3: CAS     | Mach
# Time-axis panels share x within each column. The ground-track panel (2,0)
# uses lat/lon coords, so its row-2 cell is rebuilt with independent axes.
fig, axes = plt.subplots(4, 2, figsize=(14, 16))

# Build manual sharex pairs for the time-axis panels per column.
# Col 0 time panels: rows 0 (heading), 1 (TAS), 3 (CAS). Row 2 (ground track) excluded.
# Col 1 time panels: rows 0 (alt), 1 (FPA), 2 (VZ), 3 (mach).
for r_src in [1, 3]:
    axes[r_src, 0].sharex(axes[0, 0])
for r_src in [1, 2, 3]:
    axes[r_src, 1].sharex(axes[0, 1])

# ── (0, 0) Heading + target ──
ax = axes[0, 0]
if HAS_LATERAL:
    assert heading_true is not None
    assert heading_pred is not None
    assert heading_target is not None
    assert heading_target_known is not None

    h_known_mask = heading_target_known == 1.0
    heading_true_deg = np.degrees(heading_true) % 360.0
    heading_pred_deg = np.degrees(heading_pred) % 360.0
    heading_target_deg_plot = np.where(
        h_known_mask, np.degrees(heading_target) % 360.0, np.nan
    )
    ax.plot(time_true, heading_true_deg, "k.", ms=1.5, label="True", alpha=0.6)
    ax.plot(time_pred, heading_pred_deg, "r--", lw=1.2, label="Predicted", alpha=0.8)
    ax.plot(time_true, heading_target_deg_plot, "b-", lw=2.0, label="Target", alpha=0.5)
    ax.set_ylim(-10, 370)
    if (~h_known_mask).any():
        ax.fill_between(
            time_true,
            -10,
            370,
            where=~h_known_mask,
            alpha=0.08,
            color="gray",
            label="Heading target unknown",
        )
    ax.set_ylabel("Heading [°]")
    ax.legend(loc="best", fontsize=8)
else:
    ax.text(0.5, 0.5, "no lateral channel", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Heading [°]")
ax.grid(True, alpha=0.3)

# ── (0, 1) Altitude + target ──
ax = axes[0, 1]
ax.plot(time_true, alt_true, "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, alt_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, alt_target, "b-", lw=2.0, label="Target", alpha=0.4)
_set_ylim(ax, alt_true)
ax.set_ylabel("Altitude [m]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── (1, 0) TAS + target ──
ax = axes[1, 0]
ax.plot(time_true, tas_true, "k-", lw=1.5, label="True", alpha=0.8)
ax.plot(time_pred, tas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, tas_target, "b-", lw=2.0, label="Target", alpha=0.4)
_set_ylim(ax, tas_true)
_shade_unknown(ax, time_true, tas_unknown, "TAS unknown")
ax.set_ylabel("TAS [m/s]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── (1, 1) FPA + γ_target ──
ax = axes[1, 1]
ax.plot(time_true, np.degrees(gamma_true), "k-", lw=0.8, label="True", alpha=0.5)
ax.plot(time_pred, np.degrees(gamma_pred), "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, np.degrees(gamma_target), "b-", lw=3.0, label="γ target (known)", alpha=0.9)
unknown_mask = gamma_known_full == 0.0
if unknown_mask.any():
    ax.fill_between(
        time_true,
        ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -10,
        10,
        where=unknown_mask,
        alpha=0.08,
        color="gray",
        label="γ unknown",
    )
_set_ylim(ax, np.degrees(gamma_true))
ax.set_ylabel("FPA [°]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── (2, 0) Ground track on a PlateCarree background ──
# Replace the default subplot with a cartopy GeoAxes (independent of sharex).
axes[2, 0].remove()
proj = ccrs.PlateCarree()
ax = fig.add_subplot(4, 2, 5, projection=proj)  # row 2, col 0 → linear index 5
if HAS_LATERAL:
    lon_min = min(lon_arr.min(), lon_pred.min() if lon_pred is not None else lon_arr.min())
    lon_max = max(lon_arr.max(), lon_pred.max() if lon_pred is not None else lon_arr.max())
    lat_min = min(lat_arr.min(), lat_pred.min() if lat_pred is not None else lat_arr.min())
    lat_max = max(lat_arr.max(), lat_pred.max() if lat_pred is not None else lat_arr.max())
    lon_margin = max(0.1, 0.10 * (lon_max - lon_min))
    lat_margin = max(0.1, 0.10 * (lat_max - lat_min))
    ax.set_extent(
        [lon_min - lon_margin, lon_max + lon_margin, lat_min - lat_margin, lat_max + lat_margin],
        crs=proj,
    )
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f0fa", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f5f0e6", zorder=0)
    ax.add_feature(cfeature.COASTLINE, lw=0.6, edgecolor="0.4", zorder=1)
    ax.add_feature(cfeature.BORDERS, lw=0.4, edgecolor="0.6", linestyle=":", zorder=1)
    gl = ax.gridlines(draw_labels=True, lw=0.4, color="0.7", alpha=0.5, zorder=2)
    gl.top_labels = False
    gl.right_labels = False

    straight = ~in_turn_arr
    if straight.any():
        ax.plot(
            lon_arr[straight],
            lat_arr[straight],
            ".",
            color="tab:blue",
            ms=1.5,
            label="True (straight)",
            transform=proj,
            zorder=3,
        )
    if in_turn_arr.any():
        ax.plot(
            lon_arr[in_turn_arr],
            lat_arr[in_turn_arr],
            ".",
            color="tab:orange",
            ms=1.5,
            label="True (in_turn)",
            transform=proj,
            zorder=3,
        )
    if lat_pred is not None and lon_pred is not None:
        ax.plot(lon_pred, lat_pred, "r--", lw=1.2, alpha=0.8, label="Predicted",
                transform=proj, zorder=4)
        ax.plot(lon_pred[-1], lat_pred[-1], "rv", ms=10, mfc="none",
                label="end (pred)", transform=proj, zorder=5)
    ax.plot(lon_arr[0], lat_arr[0], "g^", ms=10, label="start", transform=proj, zorder=5)
    ax.plot(lon_arr[-1], lat_arr[-1], "kv", ms=10, label="end (true)", transform=proj, zorder=5)
    ax.legend(loc="best", fontsize=7)
else:
    ax.text(0.5, 0.5, "no lateral channel", ha="center", va="center", transform=ax.transAxes)

# ── (2, 1) VZ ──
ax = axes[2, 1]
ax.plot(time_true, vz_true, "k-", lw=1.5, label="True (raw_vz_ms)", alpha=0.8)
ax.plot(time_pred, vz_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, vz_sel, "b-", lw=2.0, label="VZ target (sel)", alpha=0.4)
ax.axhline(0, color="gray", ls=":", lw=0.8)
_set_ylim(ax, vz_true)
_shade_unknown(ax, time_true, vz_unknown, "VZ unknown")
ax.set_ylabel("VZ [m/s]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── (3, 0) CAS ──
ax = axes[3, 0]
ax.plot(time_true, cas_true, "k-", lw=1.5, label="True (bds_ias_ms)", alpha=0.8)
ax.plot(time_pred, cas_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, cas_sel, "b-", lw=2.0, label="CAS target (sel)", alpha=0.4)
_set_ylim(ax, cas_true)
_shade_unknown(ax, time_true, cas_unknown, "CAS unknown")
ax.set_ylabel("CAS [m/s]")
ax.set_xlabel("Time [min]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

# ── (3, 1) Mach ──
ax = axes[3, 1]
ax.plot(time_true, mach_true, "k-", lw=1.5, label="True (era_mach)", alpha=0.8)
ax.plot(time_pred, mach_pred, "r--", lw=1.2, label="Predicted", alpha=0.8)
ax.plot(time_true, mach_sel, "b-", lw=2.0, label="Mach target (sel)", alpha=0.4)
_set_ylim(ax, mach_true)
_shade_unknown(ax, time_true, mach_unknown, "Mach unknown")
ax.set_ylabel("Mach [-]")
ax.set_xlabel("Time [min]")
ax.legend(loc="best", fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(f"Neural ODE Inference — {best_fid}", fontsize=14)
fig.tight_layout()

model_suffix = "" if cli_model_name == f"{info.name}_A320" else f"_{cli_model_name}"
out_path = Path(f"data/figures/inference_check_{best_fid}{model_suffix}.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nSaved to {out_path}")
plt.close()
