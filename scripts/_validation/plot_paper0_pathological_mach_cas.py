"""Plot Mach(t), CAS(t), TAS(t) for the pathological flight #3 AAF224.

Adapted from data/figures/new_idea_segment/plot_mach_v6.py — same panels and
bilateral plateau detection — but reads from a QAR parquet (running the prod
preprocessing pipeline first to populate bds_mach_clean / bds_ias_kt_clean /
fdm_tas_from_cas_kt / fdm_mach_sel / fdm_cas_sel_kt) and overlays the FCU
truth (SPD__MACH_SEL filtered by SPD__SPD_MACH_SEL=='MACH', SPD__SPD_SEL
filtered by =='SPEED') as red dashed lines on the Mach + CAS panels.

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    unset VIRTUAL_ENV
    uv run python scripts/_validation/plot_paper0_pathological_mach_cas.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import butter, filtfilt

# Make the validation script importable for the adapter helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper0_qar_target_validation as p0  # noqa: E402

from node_fdm_data.physics.speed import cas_to_tas_real, mach_to_tas_real  # noqa: E402

KT_TO_MS = 0.514444
M_FT = 0.3048
DT = 4.0  # s (post-downsample)

QAR_PATH = Path(
    "/Users/gabriel/Downloads/QAR3/"
    "20180201_AAF224_170829_DAAG_LFPO_F-HAAF_A320-200_16335.parquet"
)
OUT_PNG = Path(
    "data/figures/paper0_qar_target_validation/"
    "pathological_flight_AAF224_mach_cas_v6.png"
)

# Detection parameters — identical to plot_mach_v6.py.
MACH_BILAT_SIGMA_S = 8.0
MACH_BILAT_SIGMA_R = 0.08
MACH_SLOPE_TOL = 6.5e-4
MACH_FLAT_TOL = 5e-2
MIN_MACH_PLATEAU_PTS = 15
MIN_MACH_PLATEAU_VALUE = 0.5

CAS_BILAT_SIGMA_S = 8.0
CAS_BILAT_SIGMA_R = 15.0
CAS_SLOPE_TOL = 0.25
CAS_FLAT_TOL = 20.0
MIN_CAS_PLATEAU_PTS = 5

BUTTER_ORDER = 4
CUTOFF_CAS_S = 180.0


def _interp_nans(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    mask = ~np.isfinite(y)
    if mask.all():
        return np.zeros_like(y)
    if mask.any():
        idx = np.arange(len(y))
        y[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
    return y


def _bilateral_1d(y: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    y = _interp_nans(y)
    n = len(y)
    half = int(np.ceil(3 * sigma_s))
    out = np.empty_like(y)
    spatial = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_s) ** 2)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        ys = y[a:b]
        sp_w = spatial[a - (i - half) : b - (i - half)]
        rng_w = np.exp(-0.5 * ((ys - y[i]) / sigma_r) ** 2)
        w = sp_w * rng_w
        out[i] = float(np.sum(w * ys) / np.sum(w))
    return out


def _butter_lowpass(y: np.ndarray, cutoff_s: float, order: int = BUTTER_ORDER) -> np.ndarray:
    y_arr = _interp_nans(y)
    nyq = 0.5 / DT
    wn = min(0.99, (1.0 / cutoff_s) / nyq)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, y_arr)


def _ms_to_kt(ms: np.ndarray) -> np.ndarray:
    return ms / KT_TO_MS


# ----- Load QAR + run prod pipeline to populate bds_*_clean / fdm_*_sel -----
df_qar = pl.read_parquet(QAR_PATH)
df_raw = p0.qar_to_raw_schema(df_qar, QAR_PATH.stem)
df_raw = p0.downsample(df_raw, p0.QAR_DOWNSAMPLE)
df = p0.run_preprocessing(df_raw)

# Aligned QAR truth (downsampled to match pipeline grid).
truth_idx = np.arange(0, df_qar.height, p0.QAR_DOWNSAMPLE)
mode = df_qar["SPD__SPD_MACH_SEL"].to_numpy()[truth_idx]
mach_sel_truth = np.where(mode == "MACH", df_qar["SPD__MACH_SEL"].to_numpy()[truth_idx], np.nan)
cas_sel_truth = np.where(mode == "SPEED", df_qar["SPD__SPD_SEL"].to_numpy()[truth_idx], np.nan)

t_min = np.arange(df.height) * DT / 60.0
era_mach = df["bds_mach_clean"].cast(pl.Float64).to_numpy()
era_cas = df["bds_ias_kt_clean"].cast(pl.Float64).to_numpy()
era_tas_kt = df["fdm_tas_from_cas_kt"].cast(pl.Float64).to_numpy()
era_temp_k = df["era_temp_K"].cast(pl.Float64).to_numpy()
alt_ft = df["raw_alt_ft"].cast(pl.Float64).to_numpy()
alt_m_arr = alt_ft * M_FT

# --- Altitude plateaus from the prod fdm_alt_sel_ft column (bilateral_vz mode).
alt_sel = df["fdm_alt_sel_ft"].cast(pl.Float64).to_numpy()
alt_plateau_mask = np.isfinite(alt_sel)
# Convert contiguous-True runs to (start, end) intervals (for the "Mach must
# overlap an alt-hold" sanity check, as in plot_mach_v6.py).
alt_plateaus_alt: list[tuple[int, int]] = []
in_run = False
start = 0
for i, m in enumerate(alt_plateau_mask):
    if m and not in_run:
        start = i
        in_run = True
    elif not m and in_run:
        alt_plateaus_alt.append((start, i - 1))
        in_run = False
if in_run:
    alt_plateaus_alt.append((start, len(alt_plateau_mask) - 1))
print(f"alt: {len(alt_plateaus_alt)} plateaus, coverage={int(alt_plateau_mask.sum())}/{len(alt_ft)}")

# --- Mach plateaus (bilateral 2-pass + slope/flat tol + alt-plateau gate).
mach_bilat = _bilateral_1d(era_mach, MACH_BILAT_SIGMA_S, MACH_BILAT_SIGMA_R)
mach_bilat = _bilateral_1d(mach_bilat, MACH_BILAT_SIGMA_S, MACH_BILAT_SIGMA_R)
dmach = np.abs(np.diff(mach_bilat, prepend=mach_bilat[0]))
flat_mach = dmach < MACH_SLOPE_TOL

mach_plateaus: list[tuple[int, int, float]] = []
i = 0
n_m = len(mach_bilat)
while i < n_m:
    if not flat_mach[i]:
        i += 1
        continue
    j = i
    while j + 1 < n_m and flat_mach[j + 1]:
        j += 1
    if j - i + 1 >= MIN_MACH_PLATEAU_PTS:
        seg = mach_bilat[i : j + 1]
        if (seg.max() - seg.min()) <= MACH_FLAT_TOL:
            endpoint_in_alt = bool(
                alt_plateau_mask[i] or alt_plateau_mask[min(j, len(alt_plateau_mask) - 1)]
            )
            contains_alt_plateau = any(a >= i and b <= j for a, b in alt_plateaus_alt)
            if endpoint_in_alt or contains_alt_plateau:
                mu = float(np.nanmean(era_mach[i : j + 1]))
                if mu >= MIN_MACH_PLATEAU_VALUE:
                    mach_plateaus.append((i, j, mu))
    i = j + 1

# --- CAS plateaus (bilateral 2-pass + slope/flat tol, excluding Mach zones).
cas_bilat = _bilateral_1d(era_cas, CAS_BILAT_SIGMA_S, CAS_BILAT_SIGMA_R)
cas_bilat = _bilateral_1d(cas_bilat, CAS_BILAT_SIGMA_S, CAS_BILAT_SIGMA_R)

mach_mask = np.zeros(len(era_cas), dtype=bool)
for a, b, _ in mach_plateaus:
    mach_mask[a : b + 1] = True

dcas = np.abs(np.diff(cas_bilat, prepend=cas_bilat[0]))
flat_cas = (dcas < CAS_SLOPE_TOL) & (~mach_mask)

cas_plateau_list: list[tuple[int, int, float]] = []
i = 0
n_c = len(cas_bilat)
while i < n_c:
    if not flat_cas[i]:
        i += 1
        continue
    j = i
    while j + 1 < n_c and flat_cas[j + 1]:
        j += 1
    if j - i + 1 >= MIN_CAS_PLATEAU_PTS:
        seg = cas_bilat[i : j + 1]
        if (seg.max() - seg.min()) <= CAS_FLAT_TOL:
            mu_c = float(np.nanmean(era_cas[i : j + 1]))
            cas_plateau_list.append((i, j, mu_c))
    i = j + 1

print(
    f"{df.height} samples | mach plateaus={len(mach_plateaus)} | "
    f"cas plateaus={len(cas_plateau_list)} | alt plateaus={len(alt_plateaus_alt)}"
)

# --- Plot ---
fig, (ax_m, ax_c, ax_t) = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

# Mach panel.
ax_m.plot(t_min, era_mach, color="0.4", lw=0.8, alpha=0.7, label="bds_mach_clean (raw)")
for a, b, mu in mach_plateaus:
    ax_m.axvspan(t_min[a], t_min[b], color="#c8f0c8", alpha=0.6, zorder=0)
    ax_m.hlines(mu, t_min[a], t_min[b], colors="green", lw=2.5, zorder=4)
# CAS plateaus shown as blue spans on Mach panel too (no propagation here).
for a, b, _ in cas_plateau_list:
    ax_m.axvspan(t_min[a], t_min[b], color="#c8d8ff", alpha=0.6, zorder=0)
# Pointwise propagation: CAS plateaus → Mach via TAS.
for a, b, cas_val_kt in cas_plateau_list:
    sl = slice(a, b + 1)
    h = alt_m_arr[sl]
    t = era_temp_k[sl].astype(np.float64)
    cas_ms = np.full_like(h, cas_val_kt * KT_TO_MS, dtype=np.float64)
    tas_ms = np.asarray(cas_to_tas_real(cas_ms, h, t), dtype=np.float64)
    a_local = np.sqrt(1.4 * 287.05287 * t)
    mach_pw = tas_ms / a_local
    ax_m.plot(t_min[sl], mach_pw, color="blue", lw=2.5, zorder=5)
# FCU truth overlay — the methodology disconnect.
ax_m.plot(
    t_min, mach_sel_truth,
    color="red", lw=1.5, ls="--", alpha=0.85, zorder=6,
    label="FCU truth (SPD__MACH_SEL where mode='MACH')",
)
ax_m.set_ylabel("Mach")
ax_m.grid(alpha=0.3)
ax_m.legend(loc="best", fontsize=8)
ax_m.set_title(
    "Pathological flight #3 (DAAG→LFPO, AAF224) — "
    "Mach / CAS / TAS plateau detection vs FCU truth"
)

# CAS panel.
ax_c.plot(t_min, era_cas, color="0.4", lw=0.8, alpha=0.7, label="bds_ias_kt_clean (raw)")
for a, b, _ in mach_plateaus:
    ax_c.axvspan(t_min[a], t_min[b], color="#c8f0c8", alpha=0.6, zorder=0)
for a, b, mu_c in cas_plateau_list:
    ax_c.axvspan(t_min[a], t_min[b], color="#c8d8ff", alpha=0.6, zorder=0)
    ax_c.hlines(mu_c, t_min[a], t_min[b], colors="blue", lw=2.5, zorder=4)
# Mach plateaus → CAS via TAS pointwise.
for a, b, mach_val in mach_plateaus:
    sl = slice(a, b + 1)
    h = alt_m_arr[sl]
    t = era_temp_k[sl].astype(np.float64)
    tas_ms = np.asarray(mach_to_tas_real(np.full_like(t, mach_val), t), dtype=np.float64)
    cas_ms_arr = np.empty_like(tas_ms)
    for k, (tas_target, hi, ti) in enumerate(zip(tas_ms, h, t, strict=False)):
        lo, hi_v = 0.0, 400.0 * KT_TO_MS
        for _ in range(40):
            mid = 0.5 * (lo + hi_v)
            tas_try = float(
                np.asarray(cas_to_tas_real(np.array([mid]), np.array([hi]), np.array([ti])))[0]
            )
            if tas_try < tas_target:
                lo = mid
            else:
                hi_v = mid
        cas_ms_arr[k] = 0.5 * (lo + hi_v)
    ax_c.plot(t_min[sl], _ms_to_kt(cas_ms_arr), color="green", lw=2.5, zorder=5)
# FCU truth overlay.
ax_c.plot(
    t_min, cas_sel_truth,
    color="red", lw=1.5, ls="--", alpha=0.85, zorder=6,
    label="FCU truth (SPD__SPD_SEL where mode='SPEED')",
)
ax_c.set_ylabel("CAS [kt]")
ax_c.grid(alpha=0.3)
ax_c.legend(loc="best", fontsize=8)

# TAS panel — informational, no truth (QAR has actual TAS = SPD__TAS, not target).
ax_t.plot(t_min, era_tas_kt, color="0.4", lw=0.8, alpha=0.7, label="fdm_tas_from_cas_kt (raw)")
for a, b, mach_val in mach_plateaus:
    sl = slice(a, b + 1)
    t_loc = era_temp_k[sl].astype(np.float64)
    tas_ms = np.asarray(mach_to_tas_real(np.full_like(t_loc, mach_val), t_loc), dtype=np.float64)
    ax_t.axvspan(t_min[a], t_min[b], color="#c8f0c8", alpha=0.6, zorder=0)
    ax_t.plot(t_min[sl], _ms_to_kt(tas_ms), color="green", lw=2.5, zorder=5)
for a, b, cas_val_kt in cas_plateau_list:
    sl = slice(a, b + 1)
    h = alt_m_arr[sl]
    t_loc = era_temp_k[sl].astype(np.float64)
    cas_ms = np.full_like(h, cas_val_kt * KT_TO_MS, dtype=np.float64)
    tas_ms = np.asarray(cas_to_tas_real(cas_ms, h, t_loc), dtype=np.float64)
    ax_t.axvspan(t_min[a], t_min[b], color="#c8d8ff", alpha=0.6, zorder=0)
    ax_t.plot(t_min[sl], _ms_to_kt(tas_ms), color="blue", lw=2.5, zorder=5)
# Actual QAR TAS for reference.
spd_tas = df_qar["SPD__TAS"].cast(pl.Float64).to_numpy()[truth_idx]
ax_t.plot(t_min, spd_tas, color="black", lw=0.6, alpha=0.5, label="SPD__TAS (QAR actual)")
ax_t.set_ylabel("TAS [kt]")
ax_t.set_xlabel("Time [min]")
ax_t.grid(alpha=0.3)
ax_t.legend(loc="best", fontsize=8)

fig.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140)
print(f"[saved] {OUT_PNG}")
