"""Plot the pathological flight (#3) from Paper 0 QAR validation.

Same 3-panel display as `data/figures/new_idea_segment/plot_alt_hyperlisse.py`
(altitude / vz / gamma + alt-hold > gamma > vz plateau spans), but reading from
a QAR parquet instead of the delta lake, and overlaying the FCU truth
(NAV__ALT_SEL_F) on the altitude panel so the methodology disconnect is
visible: the FCU dial is 34000 ft (round number), the plateau detector
finds a sub-plateau at 33416 ft, and that drives the wrong-direction climb.

Usage::

    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    unset VIRTUAL_ENV
    uv run python scripts/_validation/plot_paper0_pathological_flight.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

QAR_PATH = Path(
    "/Users/gabriel/Downloads/QAR3/"
    "20180201_AAF224_170829_DAAG_LFPO_F-HAAF_A320-200_16335.parquet"
)
DT = 4.0  # s — pipeline operates at 0.25 Hz, QAR is 1 Hz → step=4
QAR_STEP = 4

OUT_PNG = Path(
    "data/figures/paper0_qar_target_validation/"
    "pathological_flight_AAF224_alt_hyperlisse.png"
)

KT_TO_MS = 0.514444
FT_MIN_TO_MS = 0.3048 / 60

L1_LAMBDA = 5e3
L1_LAMBDA_TAS = 50.0
L1_ITERS = 400
L1_RHO = 1.0


def _interp_nans(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    mask = ~np.isfinite(y)
    if mask.all():
        return np.zeros_like(y)
    if mask.any():
        idx = np.arange(len(y))
        y[mask] = np.interp(idx[mask], idx[~mask], y[~mask])
    return y


def _l1_trend_filter(
    y: np.ndarray, lam: float, rho: float = L1_RHO, n_iter: int = L1_ITERS
) -> np.ndarray:
    """L1-trend filter via ADMM (piecewise-linear output, piecewise-constant slope)."""
    y = _interp_nans(y)
    n = len(y)
    if n < 4:
        return y.copy()
    e = np.ones(n)
    D2 = sp.diags([e, -2 * e, e], [0, 1, 2], shape=(n - 2, n)).tocsc()
    A = sp.eye(n, format="csc") + rho * (D2.T @ D2)
    solve = spla.factorized(A)
    z = np.zeros(n - 2)
    u = np.zeros(n - 2)
    thresh = lam / rho
    x = y.copy()
    for _ in range(n_iter):
        rhs = y + rho * (D2.T @ (z - u))
        x = solve(rhs)
        Dx = D2 @ x
        v = Dx + u
        z = np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)
        u = u + Dx - z
    return x


def _bilateral_1d(y: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """1D bilateral filter — flattens homogeneous zones, preserves jumps."""
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


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            runs.append((i, j))
        i = j + 1
    return runs


# --- Load + downsample to 0.25 Hz to match pipeline DT ---
df_full = pl.read_parquet(QAR_PATH)
df = df_full.with_row_index("__r").filter(pl.col("__r") % QAR_STEP == 0).drop("__r")

t_min = np.arange(df.height) * DT / 60.0
alt_ft = df["ALT__STD"].cast(pl.Float64).to_numpy()
vz_ftmin = df["ATT__VV"].cast(pl.Float64).to_numpy()
tas_kt = df["SPD__TAS"].cast(pl.Float64).to_numpy()
# FCU truth — use the filtered column (validated user choice).
fcu_alt_ft = df["NAV__ALT_SEL_F"].cast(pl.Float64).to_numpy()

alt_l1 = _l1_trend_filter(alt_ft, L1_LAMBDA)
vz_l1 = np.gradient(alt_l1, DT) * 60.0
tas_l1_kt = _l1_trend_filter(tas_kt, L1_LAMBDA_TAS)

ratio_raw = np.clip(
    (vz_ftmin * FT_MIN_TO_MS) / np.clip(tas_kt * KT_TO_MS, 1e-6, None), -1.0, 1.0
)
gamma_raw = np.arcsin(ratio_raw)

VZ_BILAT_SIGMA_S = 6.0
VZ_BILAT_SIGMA_R = 350.0
GAMMA_BILAT_SIGMA_S = 6.0
GAMMA_BILAT_SIGMA_R = 1.2e-2

vz_bilat = _bilateral_1d(vz_ftmin, VZ_BILAT_SIGMA_S, VZ_BILAT_SIGMA_R)
vz_bilat = _bilateral_1d(vz_bilat, VZ_BILAT_SIGMA_S, VZ_BILAT_SIGMA_R)
gamma_bilat = _bilateral_1d(gamma_raw, GAMMA_BILAT_SIGMA_S, GAMMA_BILAT_SIGMA_R)
gamma_bilat = _bilateral_1d(gamma_bilat, GAMMA_BILAT_SIGMA_S, GAMMA_BILAT_SIGMA_R)

MIN_PLATEAU_PTS = 10
MIN_ALT_PLATEAU_PTS = 6
n = len(vz_bilat)
t_s = np.arange(df.height) * DT

# --- Priority 1: ALTITUDE plateaus (alt-hold). vz_bilat ≈ 0.
ALT_HOLD_TOL_FTMIN = 150.0
alt_plateaus = _runs(np.abs(vz_bilat) < ALT_HOLD_TOL_FTMIN, MIN_ALT_PLATEAU_PTS)
alt_hold_mask = np.zeros(n, dtype=bool)
for a, b in alt_plateaus:
    alt_hold_mask[a : b + 1] = True

# --- Priority 2: GAMMA plateaus, excluding alt-hold zones.
GAMMA_SLOPE_TOL_RAD = 3e-4
GAMMA_FLAT_TOL_RAD = 2e-3
dgamma = np.abs(np.diff(gamma_bilat, prepend=gamma_bilat[0]))
flat_g = (dgamma < GAMMA_SLOPE_TOL_RAD) & (~alt_hold_mask)
gamma_candidates = _runs(flat_g, MIN_PLATEAU_PTS)
gamma_plateaus: list[tuple[int, int]] = []
for a, b in gamma_candidates:
    seg = gamma_bilat[a : b + 1]
    if (seg.max() - seg.min()) <= 2 * GAMMA_FLAT_TOL_RAD and abs(np.mean(seg)) >= 5e-3:
        gamma_plateaus.append((a, b))
gamma_mask = np.zeros(n, dtype=bool)
for a, b in gamma_plateaus:
    gamma_mask[a : b + 1] = True

# --- Priority 3: VZ plateaus, excluding alt-hold and γ zones.
VZ_FLAT_TOL_FTMIN = 100.0
SLOPE_TOL_FTMIN = 15.0
dvz = np.abs(np.diff(vz_bilat, prepend=vz_bilat[0]))
flat_v = (dvz < SLOPE_TOL_FTMIN) & (~alt_hold_mask) & (~gamma_mask)
vz_candidates = _runs(flat_v, MIN_PLATEAU_PTS)
plateaus: list[tuple[int, int]] = []
for a, b in vz_candidates:
    seg = vz_bilat[a : b + 1]
    if (seg.max() - seg.min()) <= 2 * VZ_FLAT_TOL_FTMIN:
        plateaus.append((a, b))

# --- Refit per-plateau values ---
alt_plateau = np.full_like(alt_ft, np.nan, dtype=np.float64)
vz_plateau = np.full_like(alt_ft, np.nan, dtype=np.float64)
for a, b in plateaus:
    sl = slice(a, b + 1)
    vz_med = float(np.mean(vz_bilat[sl]))
    vz_plateau[sl] = vz_med
    mid = (a + b) // 2
    alt_plateau[sl] = alt_ft[mid] + (vz_med / 60.0) * (t_s[sl] - t_s[mid])

alt_hold_alt = np.full_like(alt_ft, np.nan, dtype=np.float64)
alt_hold_vz = np.full_like(alt_ft, np.nan, dtype=np.float64)
for a, b in alt_plateaus:
    sl = slice(a, b + 1)
    alt_hold_vz[sl] = 0.0
    alt_hold_alt[sl] = float(np.mean(alt_ft[sl]))

gamma_plateau_arr = np.full_like(gamma_bilat, np.nan, dtype=np.float64)
alt_gamma = np.full_like(alt_ft, np.nan, dtype=np.float64)
for a, b in gamma_plateaus:
    sl = slice(a, b + 1)
    gamma_plateau_arr[sl] = float(np.mean(gamma_bilat[sl]))
    vz_med = float(np.mean(vz_bilat[sl]))
    mid = (a + b) // 2
    alt_gamma[sl] = alt_ft[mid] + (vz_med / 60.0) * (t_s[sl] - t_s[mid])

gamma_from_vz = np.full_like(gamma_bilat, np.nan, dtype=np.float64)
for a, b in plateaus:
    sl = slice(a, b + 1)
    vz_med_ms = float(np.mean(vz_bilat[sl])) * FT_MIN_TO_MS
    tas_ms = tas_kt[sl] * KT_TO_MS
    gamma_from_vz[sl] = np.arcsin(
        np.clip(vz_med_ms / np.clip(tas_ms, 1e-6, None), -1.0, 1.0)
    )

gamma_from_alt_hold = np.full_like(gamma_bilat, np.nan, dtype=np.float64)
for a, b in alt_plateaus:
    gamma_from_alt_hold[a : b + 1] = 0.0

vz_from_gamma = np.full_like(vz_bilat, np.nan, dtype=np.float64)
for a, b in gamma_plateaus:
    sl = slice(a, b + 1)
    g_med = float(np.mean(gamma_bilat[sl]))
    tas_ms = tas_kt[sl] * KT_TO_MS
    vz_from_gamma[sl] = tas_ms * np.sin(g_med) / FT_MIN_TO_MS

print(
    f"{df.height} samples @ DT={DT}s | "
    f"alt-hold={len(alt_plateaus)} ({int(alt_hold_mask.sum())}) | "
    f"γ={len(gamma_plateaus)} ({int(gamma_mask.sum())}) | "
    f"vz={len(plateaus)} ({int(np.isfinite(vz_plateau).sum())})"
)
print(f"Distinct alt-hold values (ft): "
      f"{sorted({int(round(float(np.mean(alt_ft[a:b+1])))) for a, b in alt_plateaus})}")
print(f"Distinct FCU truth values (ft): {sorted(np.unique(fcu_alt_ft[np.isfinite(fcu_alt_ft)]).astype(int).tolist())}")

# --- Plot ---
fig, (ax_a, ax_v, ax_g) = plt.subplots(3, 1, figsize=(13, 11), sharex=True)


def _spans(ax: plt.Axes) -> None:
    for a, b in alt_plateaus:
        ax.axvspan(t_min[a], t_min[b], color="#c8f0c8", alpha=0.6, zorder=0)
    for a, b in gamma_plateaus:
        ax.axvspan(t_min[a], t_min[b], color="#ffd8a8", alpha=0.6, zorder=0)
    for a, b in plateaus:
        ax.axvspan(t_min[a], t_min[b], color="#c8d8ff", alpha=0.6, zorder=0)


ax_a.plot(t_min, alt_ft, color="0.4", lw=0.8, alpha=0.7, label="raw_alt_ft")
ax_a.plot(t_min, alt_plateau, "b-", lw=2.0, label=f"vz-plateau ({len(plateaus)})")
ax_a.plot(
    t_min, alt_gamma, color="darkorange", lw=2.0, label=f"γ-plateau ({len(gamma_plateaus)})"
)
ax_a.plot(t_min, alt_hold_alt, "g-", lw=2.0, label=f"alt-hold ({len(alt_plateaus)})")
# FCU truth overlay — the methodology disconnect.
ax_a.plot(
    t_min, fcu_alt_ft, color="red", lw=1.3, ls="--", alpha=0.85,
    label="FCU truth (NAV__ALT_SEL_F)",
)
_spans(ax_a)
ax_a.set_ylabel("Altitude [ft]")
ax_a.grid(alpha=0.3)
ax_a.legend(loc="best")
ax_a.set_title(
    f"Pathological flight #3 (DAAG→LFPO, AAF224) — "
    f"FCU dial 34000 ft (red) vs plateau-detector’s 33416 ft (green) → "
    f"Δ = −178 m wrong-direction during entire climb"
)

ax_v.plot(t_min, vz_ftmin, color="0.4", lw=0.8, alpha=0.7, label="raw_vz_ftmin")
ax_v.plot(t_min, alt_hold_vz, "g-", lw=2.5, label="alt-hold vz")
ax_v.plot(
    t_min, vz_from_gamma, color="darkorange", lw=3.0, zorder=10,
    label=f"vz from γ = TAS·sin(γ) ({len(gamma_plateaus)})",
)
ax_v.plot(t_min, vz_plateau, "b-", lw=2.5, label=f"vz plateaus ({len(plateaus)})")
_spans(ax_v)
ax_v.axhline(0, color="0.5", lw=0.6, ls=":")
ax_v.set_ylabel("Vz [ft/min]")
ax_v.grid(alpha=0.3)
ax_v.legend(loc="best")

ax_g.plot(t_min, gamma_raw, color="0.4", lw=0.8, alpha=0.7, label="γ raw = asin(vz/TAS)")
ax_g.plot(
    t_min, gamma_from_alt_hold, "g-", lw=2.5,
    label=f"γ from alt-hold = 0 ({len(alt_plateaus)})",
)
ax_g.plot(
    t_min, gamma_plateau_arr, color="darkorange", lw=2.5,
    label=f"γ plateaus ({len(gamma_plateaus)})",
)
ax_g.plot(
    t_min, gamma_from_vz, "b-", lw=2.5,
    label=f"γ from vz = asin(vz/TAS) ({len(plateaus)})",
)
_spans(ax_g)
ax_g.axhline(0, color="0.5", lw=0.6, ls=":")
ax_g.set_ylabel("γ [rad]")
ax_g.set_xlabel("Time [min]")
ax_g.grid(alpha=0.3)
ax_g.legend(loc="best")

fig.tight_layout()
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=140)
print(f"[saved] {OUT_PNG}")
