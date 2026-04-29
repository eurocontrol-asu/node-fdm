"""Compare current vs crossover-aware speed segment detection.

Loads a real flight from the Delta Table and plots:
- Row 1: Altitude profile with crossover altitude line
- Row 2: Mach — ground truth + current segments + proposed (above crossover)
- Row 3: CAS  — ground truth + current segments + proposed (below crossover)
- Row 4: TAS  — ground truth + current (backfill global) + proposed (fill bounded by crossover)

Usage:
    uv run python scripts/check_crossover_segments.py [--flight FLIGHT_ID] [--out PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from deltalake import DeltaTable

from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas, tas_to_cas
from node_fdm_data.segments import build_selected_params, detect_constant_segments

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KT_TO_MS = 0.514444
_FT_TO_M = 0.3048
_GAMMA = 1.4
_R = 287.05
_P0 = 101_325.0
_A0 = (_GAMMA * _R * 288.15) ** 0.5
_GM1_2 = (_GAMMA - 1.0) / 2.0  # 0.2
_G_GM1 = _GAMMA / (_GAMMA - 1.0)  # 3.5
_INV_G_GM1 = 1.0 / _G_GM1  # 2/7


def _mach_to_tas_real(mach: np.ndarray, temp_k: np.ndarray) -> np.ndarray:
    """Mach → TAS (m/s) using real temperature (not ISA)."""
    a_local = np.sqrt(_GAMMA * _R * temp_k)
    return mach * a_local


def _cas_to_mach(cas_ms: np.ndarray, h_m: np.ndarray) -> np.ndarray:
    """CAS (m/s) → Mach via ISA pressure (this is correct by definition)."""
    from node_fdm_data.physics.isa import isa_pressure
    p = np.asarray(isa_pressure(h_m), dtype=np.float64)
    cas_ratio = cas_ms / _A0
    cas_ratio = np.where(cas_ms < 0.0, np.nan, cas_ratio)
    qc = _P0 * ((1.0 + _GM1_2 * cas_ratio**2) ** _G_GM1 - 1.0)
    return np.sqrt(2.0 / (_GAMMA - 1.0) * ((qc / p + 1.0) ** _INV_G_GM1 - 1.0))


def _cas_to_tas_real(cas_ms: np.ndarray, h_m: np.ndarray, temp_k: np.ndarray) -> np.ndarray:
    """CAS (m/s) → TAS (m/s) using real temperature for Mach→TAS step."""
    mach = _cas_to_mach(cas_ms, h_m)
    return _mach_to_tas_real(mach, temp_k)


# ---------------------------------------------------------------------------
# Crossover altitude calculation
# ---------------------------------------------------------------------------


def crossover_altitude(cas_kt: float, mach: float) -> float:
    """Find altitude (ft) where CAS and Mach give the same TAS.

    Binary search between 0 and 45000 ft.
    """
    cas_ms = cas_kt * _KT_TO_MS
    lo, hi = 0.0, 45_000.0 * _FT_TO_M  # metres
    for _ in range(60):
        mid = (lo + hi) / 2
        tas_cas = float(cas_to_tas(cas_ms, mid))
        tas_mach = float(mach_to_tas(mach, mid))
        # Below crossover: CAS→TAS < Mach→TAS
        # Above crossover: CAS→TAS > Mach→TAS
        if tas_cas < tas_mach:
            lo = mid
        else:
            hi = mid
    return mid / _FT_TO_M  # return ft


# ---------------------------------------------------------------------------
# Segment detection configs
# ---------------------------------------------------------------------------

# Current pipeline defaults (tol=0.0005 for Mach)
CURRENT_CONFIG: dict = {
    "mach": {
        "tol": 0.0005,
        "min_len": 120,
        "alt_threshold": 15000,
        "use_alt": True,
        "smooth_window": 10,
    },
    "cas": {
        "tol": 0.75,
        "min_len": 30,
        "use_alt": False,
        "smooth_window": 10,
        "smooth_method": "savgol",
    },
    "tas": {
        "tol": 1.0,
        "min_len": 30,
        "use_alt": False,
    },
    "vz": {
        "tol": 25,
        "min_len": 20,
        "use_alt": False,
        "min_abs_value": 75,
        "smooth_window": 15,
        "smooth_method": "savgol",
    },
    "gamma": {
        "tol": 0.002,
        "min_len": 15,
        "use_alt": False,
        "smooth_window": 5,
        "smooth_method": "savgol",
    },
    "alt": {
        "tol": 25,
        "min_len": 5,
        "use_alt": False,
        "min_abs_value": 25,
        "smooth_window": 5,
        "smooth_method": "savgol",
    },
}

# Proposed: same detection params for both Mach and CAS
_PROPOSED_COMMON: dict = {
    "tol": 0.001,
    "smooth_window": 10,
    "smooth_method": "savgol",
}

PROPOSED_MACH_CFG: dict = {
    **_PROPOSED_COMMON,
    "min_len": 30,  # avoid short noisy segments on plateaus
    "alt_threshold": 15000,
    "use_alt": True,
}

PROPOSED_CAS_CFG: dict = {
    **_PROPOSED_COMMON,
    "tol": 0.75,  # kt — override common tol (which is in Mach units)
    "min_len": 15,
    "use_alt": False,
}


# ---------------------------------------------------------------------------
# Crossover-bounded fill
# ---------------------------------------------------------------------------


def _zone_fill(values: np.ndarray, zone_mask: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill *values* but only within *zone_mask*.

    Gaps outside the zone stay NaN. Fill never crosses zone boundaries.
    """
    out = values.copy()
    # Identify contiguous runs of zone_mask=True
    changes = np.diff(zone_mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        chunk = out[s:e]
        # forward fill
        mask = np.isnan(chunk)
        idx = np.where(~mask, np.arange(len(chunk)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = chunk[idx]
        # backward fill remaining leading NaNs
        mask2 = np.isnan(filled)
        if not mask2.all():
            idx2 = np.where(~mask2, np.arange(len(filled)), len(filled) - 1)
            # reverse accumulate min
            idx2_rev = idx2[::-1].copy()
            np.minimum.accumulate(idx2_rev, out=idx2_rev)
            idx2 = idx2_rev[::-1]
            filled = filled[idx2]
        out[s:e] = filled
    return out


# ---------------------------------------------------------------------------
# Proposed: crossover-aware segment detection
# ---------------------------------------------------------------------------


def proposed_segments(
    df: pl.DataFrame,
    df_preprocessed: pl.DataFrame,
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float | None, float | None]]:
    """Crossover-aware speed target using preprocessed altitude plateaus.

    Strategy:
    1. Detect Mach segments ONLY on altitude plateaus (fdm_alt_sel_ft ∩ era_mach)
       → guaranteed cruise-only Mach segments.
    2. First Mach segment → CAS anchor for climb transition.
       Last Mach segment → CAS anchor for descent transition.
    3. Detect CAS segments outside Mach regions (same as before).
    4. Envelope fill: min(TAS_mach, TAS_cas) resolves crossover naturally.

    Returns:
        mach_segs, cas_segs,
        tas_target_raw (segments only, no fill),
        tas_target_filled (envelope fill),
        tas_known (1.0 where segment, 0.0 elsewhere),
        crossover_ft (estimated from first Mach segment, for display only)
    """
    alt_arr = df["raw_alt_ft"].to_numpy()
    alt_m = alt_arr * _FT_TO_M
    temp_k = df["era_temp_K"].to_numpy()
    n = len(df)

    mach_col = "era_mach"
    cas_col = "era_cas_kt" if "era_cas_kt" in df.columns else "bds_ias_kt"
    mach_arr = df[mach_col].to_numpy()
    cas_arr = df[cas_col].to_numpy()

    # --- Mach: detect ONLY on altitude plateaus ---
    # Mask Mach to NaN everywhere that is NOT an altitude plateau.
    alt_sel = df_preprocessed["fdm_alt_sel_ft"].to_numpy()
    mach_masked = mach_arr.copy()
    mach_masked[np.isnan(alt_sel)] = np.nan
    mach_segs_raw = detect_constant_segments(mach_masked, alt_values=alt_arr, **PROPOSED_MACH_CFG)
    # Filter out aberrant low-Mach segments (data artifacts)
    mach_segs = [s for s in mach_segs_raw if s["var_mean"] > 0.5]

    # --- Phase detection ---
    _margin = 30
    has_climb = True
    has_descent = True
    xover_climb_ft = None
    xover_descent_ft = None
    cas_opt_climb = None
    cas_opt_descent = None
    tas_real = df["era_tas_kt"].to_numpy()
    cas_real = cas_arr.copy()
    cas_deviation_kt = 5.0

    if mach_segs:
        cruise_start_idx = mach_segs[0]["start_idx"]
        cruise_end_idx = mach_segs[-1]["end_idx"]
        cruise_mid = (cruise_start_idx + cruise_end_idx) // 2
        cruise_alt_ft = float(np.nanmean(alt_arr[cruise_start_idx : cruise_end_idx + 1]))
        first_mach_val = mach_segs[0]["var_mean"]
        last_mach_val = mach_segs[-1]["var_mean"]
        if cruise_start_idx < _margin:
            has_climb = False
        if cruise_end_idx > n - _margin:
            has_descent = False

        # --- Step 2: ALWAYS optimize CAS for transitions ---
        if has_climb and cruise_start_idx > 0:
            cas_opt_climb = _optimize_transition_cas(
                tas_real, alt_m, temp_k, first_mach_val, cruise_start_idx, direction="climb",
            )
        if has_descent and cruise_end_idx < n - 1:
            cas_opt_descent = _optimize_transition_cas(
                tas_real, alt_m, temp_k, last_mach_val, cruise_end_idx, direction="descent",
            )

        # --- Step 3: crossover from OPTIMIZED CAS ---
        if cas_opt_climb is not None:
            xover_climb_ft = crossover_altitude(cas_opt_climb, first_mach_val)
            if xover_climb_ft > cruise_alt_ft:
                xover_climb_ft = None
        if cas_opt_descent is not None:
            xover_descent_ft = crossover_altitude(cas_opt_descent, last_mach_val)
            if xover_descent_ft > cruise_alt_ft:
                xover_descent_ft = None
    else:
        cruise_start_idx = 0
        cruise_end_idx = n - 1
        cruise_mid = n // 2

    xover_ft = (xover_climb_ft, xover_descent_ft)

    # --- Step 4: detect CAS segments, filter by crossover ---
    cas_masked = cas_arr.copy()
    for seg in mach_segs:
        cas_masked[seg["start_idx"] : seg["end_idx"] + 1] = np.nan
    cas_segs_raw = detect_constant_segments(cas_masked, **PROPOSED_CAS_CFG)

    cas_segs = []
    if mach_segs:
        xover_min = min(
            x for x in (xover_climb_ft, xover_descent_ft) if x is not None
        ) if any(x is not None for x in (xover_climb_ft, xover_descent_ft)) else None
        for seg in cas_segs_raw:
            seg_mean_alt = float(np.nanmean(alt_arr[seg["start_idx"] : seg["end_idx"] + 1]))
            if seg["end_idx"] < cruise_start_idx:
                if xover_climb_ft is not None and seg_mean_alt > xover_climb_ft:
                    continue
            elif seg["start_idx"] > cruise_end_idx:
                if xover_descent_ft is not None and seg_mean_alt > xover_descent_ft:
                    continue
            else:
                if xover_min is not None and seg_mean_alt > xover_min:
                    continue
            cas_segs.append(seg)
    else:
        cas_segs = list(cas_segs_raw)

    # --- Build TAS target from segments only (no fill) ---
    tas_raw = np.full(n, np.nan)
    tas_known = np.zeros(n)

    for seg in cas_segs:
        s, e = seg["start_idx"], seg["end_idx"] + 1
        cas_ms = seg["var_mean"] * _KT_TO_MS
        tas_raw[s:e] = np.asarray(_cas_to_tas_real(cas_ms, alt_m[s:e], temp_k[s:e])) / _KT_TO_MS
        tas_known[s:e] = 1.0

    for seg in mach_segs:
        s, e = seg["start_idx"], seg["end_idx"] + 1
        tas_raw[s:e] = np.asarray(_mach_to_tas_real(seg["var_mean"], temp_k[s:e])) / _KT_TO_MS
        tas_known[s:e] = 1.0

    # --- Step 5: Envelope fill ---
    # Mach: segments only + propagate first backward / last forward (if crossover exists)
    mach_target = np.full(n, np.nan)
    for seg in mach_segs:
        s, e = seg["start_idx"], seg["end_idx"] + 1
        mach_target[s:e] = seg["var_mean"]

    cruise_first = mach_segs[0]["start_idx"] if mach_segs else 0
    cruise_last = mach_segs[-1]["end_idx"] if mach_segs else n - 1
    mach_filled = mach_target.copy()
    if mach_segs:
        if has_climb and xover_climb_ft is not None:
            # Extend first Mach backward, stop at last CAS segment before cruise
            climb_stop = 0
            cas_before = [s for s in cas_segs if s["end_idx"] < cruise_first]
            if cas_before:
                climb_stop = cas_before[-1]["start_idx"]
            mach_filled[climb_stop:cruise_first] = mach_segs[0]["var_mean"]
        if has_descent and xover_descent_ft is not None:
            # Extend last Mach forward, stop at first CAS segment after cruise
            descent_stop = n
            cas_after = [s for s in cas_segs if s["start_idx"] > cruise_last]
            if cas_after:
                descent_stop = cas_after[0]["end_idx"] + 1
            mach_filled[cruise_last + 1:descent_stop] = mach_segs[-1]["var_mean"]

    # CAS: detected segments + optimized transition segments
    cas_target_arr = np.full(n, np.nan)
    for seg in cas_segs:
        s, e = seg["start_idx"], seg["end_idx"] + 1
        cas_target_arr[s:e] = seg["var_mean"]

    # Inject optimized CAS in transition zones (walk until deviation from real CAS)
    if mach_segs:
        if cas_opt_climb is not None and xover_climb_ft is not None:
            for i in range(cruise_start_idx - 1, -1, -1):
                if np.isnan(cas_real[i]):
                    continue
                if abs(cas_real[i] - cas_opt_climb) > cas_deviation_kt:
                    break
                if not np.isnan(cas_target_arr[i]):
                    continue  # don't overwrite detected segments
                cas_target_arr[i] = cas_opt_climb
        if cas_opt_descent is not None and xover_descent_ft is not None:
            for i in range(cruise_end_idx + 1, n):
                if np.isnan(cas_real[i]):
                    continue
                if abs(cas_real[i] - cas_opt_descent) > cas_deviation_kt:
                    break
                if not np.isnan(cas_target_arr[i]):
                    continue
                cas_target_arr[i] = cas_opt_descent

    # CAS fill: connect last CAS before cruise → crossover (climb),
    # first CAS after cruise → crossover (descent).
    # STOP at crossover altitude.
    cas_filled = cas_target_arr.copy()
    if mach_segs:
        if xover_climb_ft is not None:
            climb_part = cas_target_arr[:cruise_mid]
            last_cas_idx = -1
            for i in range(len(climb_part) - 1, -1, -1):
                if not np.isnan(climb_part[i]):
                    last_cas_idx = i
                    break
            if last_cas_idx >= 0:
                fill_val = climb_part[last_cas_idx]
                for i in range(last_cas_idx, cruise_mid):
                    if alt_arr[i] > xover_climb_ft:
                        break
                    cas_filled[i] = fill_val

        if xover_descent_ft is not None:
            descent_part = cas_target_arr[cruise_mid:]
            first_cas_idx = -1
            for i in range(len(descent_part)):
                if not np.isnan(descent_part[i]):
                    first_cas_idx = i
                    break
            if first_cas_idx >= 0:
                abs_idx = cruise_mid + first_cas_idx
                fill_val = descent_part[first_cas_idx]
                for i in range(abs_idx, cruise_mid - 1, -1):
                    if alt_arr[i] > xover_descent_ft:
                        break
                    cas_filled[i] = fill_val

    # Convert both to TAS
    tas_from_mach = np.where(
        np.isnan(mach_filled), np.nan,
        np.asarray(_mach_to_tas_real(mach_filled, temp_k)) / _KT_TO_MS,
    )
    tas_from_cas = np.where(
        np.isnan(cas_filled), np.nan,
        np.asarray(_cas_to_tas_real(cas_filled * _KT_TO_MS, alt_m, temp_k)) / _KT_TO_MS,
    )

    # Build TAS by zone:
    # - Both available → min(Mach, CAS) — transition zone, crossover emerges
    # - Mach only (on actual Mach segments) → Mach
    # - CAS only (outside cruise zone only) → CAS
    # - Neither / between Mach segments → NaN (unknown)
    both = ~np.isnan(tas_from_mach) & ~np.isnan(tas_from_cas)
    mach_only = ~np.isnan(tas_from_mach) & np.isnan(tas_from_cas)
    cas_only = np.isnan(tas_from_mach) & ~np.isnan(tas_from_cas)

    # Mask: inside cruise zone (between first..last Mach segment) but NOT on
    # an actual Mach segment → unknown. CAS should not fill these gaps.
    idx_arr = np.arange(n)
    in_cruise_zone = (idx_arr >= cruise_first) & (idx_arr <= cruise_last)
    on_mach_seg = ~np.isnan(mach_target)  # only actual segments, not propagated
    cruise_gap = in_cruise_zone & ~on_mach_seg  # gaps between Mach segments

    tas_filled = np.full(n, np.nan)
    tas_filled[both & ~cruise_gap] = np.minimum(tas_from_mach[both & ~cruise_gap], tas_from_cas[both & ~cruise_gap])
    tas_filled[mach_only] = tas_from_mach[mach_only]
    tas_filled[cas_only & ~cruise_gap] = tas_from_cas[cas_only & ~cruise_gap]

    return mach_segs, cas_segs, tas_raw, tas_filled, tas_known, cas_filled, xover_ft


def _optimize_transition_cas(
    tas_real: np.ndarray,
    alt_m: np.ndarray,
    temp_k: np.ndarray,
    mach_val: float,
    cruise_idx: int,
    direction: str = "climb",
    window_min: int = 60,
) -> float:
    """Find the CAS (kt) that minimizes TAS error in the transition zone.

    For climb: look at the window [cruise_idx - window, cruise_idx].
    For descent: look at the window [cruise_idx, cruise_idx + window].

    The reconstructed TAS = min(mach_to_tas(M, T), cas_to_tas(CAS, h, T)).
    Uses real temperature for conversions.
    """
    n = len(tas_real)
    if direction == "climb":
        start = max(0, cruise_idx - window_min)
        end = cruise_idx
    else:
        start = cruise_idx
        end = min(n, cruise_idx + window_min)

    if start >= end:
        h = alt_m[cruise_idx]
        return float(tas_to_cas(mach_to_tas(mach_val, h), h)) / _KT_TO_MS

    h_window = alt_m[start:end]
    t_window = temp_k[start:end]
    tas_window = tas_real[start:end]
    tas_mach_window = np.asarray(_mach_to_tas_real(mach_val, t_window)) / _KT_TO_MS

    # Valid points only (no NaN in TAS real or temp)
    valid = ~np.isnan(tas_window) & ~np.isnan(t_window)
    if valid.sum() < 5:
        h = alt_m[cruise_idx]
        return float(tas_to_cas(mach_to_tas(mach_val, h), h)) / _KT_TO_MS

    h_valid = h_window[valid]
    t_valid = t_window[valid]
    tas_valid = tas_window[valid]
    tas_mach_valid = tas_mach_window[valid]

    def cost(cas_kt: float) -> float:
        cas_ms = cas_kt * _KT_TO_MS
        tas_cas = np.asarray(_cas_to_tas_real(cas_ms, h_valid, t_valid)) / _KT_TO_MS
        tas_recon = np.minimum(tas_mach_valid, tas_cas)
        return float(np.sum((tas_recon - tas_valid) ** 2))

    # Brute search over reasonable CAS range (200-350 kt, 1 kt steps)
    best_cas = 280.0
    best_cost = np.inf
    for cas_kt in np.arange(200, 351, 1.0):
        c = cost(cas_kt)
        if c < best_cost:
            best_cost = c
            best_cas = cas_kt

    # Refine with finer search (±2 kt, 0.1 kt steps)
    for cas_kt in np.arange(best_cas - 2, best_cas + 2.1, 0.1):
        c = cost(cas_kt)
        if c < best_cost:
            best_cost = c
            best_cas = cas_kt

    return best_cas


def _ffill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values."""
    out = arr.copy()
    mask = np.isnan(out)
    if mask.all():
        return out
    idx = np.where(~mask, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    # Only fill forward (keep leading NaN)
    out = arr[idx]
    # Restore leading NaN (before first valid)
    first_valid = np.argmax(~mask)
    out[:first_valid] = np.nan
    return out


def _bfill(arr: np.ndarray) -> np.ndarray:
    """Backward-fill NaN values."""
    out = arr.copy()
    mask = np.isnan(out)
    if mask.all():
        return out
    idx = np.where(~mask, np.arange(len(out)), len(out) - 1)
    idx_rev = idx[::-1].copy()
    np.minimum.accumulate(idx_rev, out=idx_rev)
    idx = idx_rev[::-1]
    out = arr[idx]
    # Restore trailing NaN (after last valid)
    last_valid = len(out) - 1 - np.argmax(~mask[::-1])
    out[last_valid + 1 :] = np.nan
    return out


def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill."""
    return _bfill(_ffill(arr))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg_spans(segs: list[dict]) -> list[tuple[int, int, float]]:
    return [(s["start_idx"], s["end_idx"], s["var_mean"]) for s in segs]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(
    df_raw: pl.DataFrame,
    df_current: pl.DataFrame,
    mach_segs_proposed: list[dict],
    cas_segs_proposed: list[dict],
    tas_target_raw: np.ndarray,
    tas_target_filled: np.ndarray,
    tas_known: np.ndarray,
    cas_filled: np.ndarray,
    xover_ft: tuple[float | None, float | None],
    out_path: Path | None = None,
) -> None:
    xover_climb, xover_descent = xover_ft
    n = len(df_raw)
    t = np.arange(n) * 4 / 60  # minutes

    alt = df_raw["raw_alt_ft"].to_numpy()
    mach = df_raw["era_mach"].to_numpy()
    cas_col = "era_cas_kt" if "era_cas_kt" in df_raw.columns else "bds_ias_kt"
    cas = df_raw[cas_col].to_numpy()
    tas = df_raw["era_tas_kt"].to_numpy()

    mach_sel_cur = df_current["fdm_mach_sel"].to_numpy() if "fdm_mach_sel" in df_current.columns else np.full(n, np.nan)
    cas_sel_cur = df_current["fdm_cas_sel_kt"].to_numpy() if "fdm_cas_sel_kt" in df_current.columns else np.full(n, np.nan)
    tas_target_cur = df_current["fdm_tas_target_kt"].to_numpy() if "fdm_tas_target_kt" in df_current.columns else np.full(n, np.nan)

    fid = df_raw["meta_flight_id"][0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    title = f"Speed Segments — {fid}"
    if xover_climb is not None:
        title += f"  |  Xover: {xover_climb:.0f}"
    if xover_descent is not None and xover_descent != xover_climb:
        title += f" / {xover_descent:.0f} ft"
    elif xover_climb is not None:
        title += " ft"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- (0,0) Altitude + crossover lines ---
    ax = axes[0, 0]
    ax.plot(t, alt, color="k", linewidth=0.8)
    if xover_climb is not None:
        ax.axhline(xover_climb, color="#e74c3c", ls="--", lw=1, alpha=0.8,
                    label=f"Xover climb {xover_climb:.0f}")
    if xover_descent is not None and xover_descent != xover_climb:
        ax.axhline(xover_descent, color="#c0392b", ls=":", lw=1, alpha=0.8,
                    label=f"Xover desc {xover_descent:.0f}")
    ax.set_ylabel("Altitude (ft)")
    ax.set_title("Altitude profile", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.2)

    # --- (0,1) Mach ---
    ax = axes[0, 1]
    ax.plot(t, mach, color="#bdc3c7", linewidth=0.6, label="Mach truth")
    ax.plot(t, mach_sel_cur, color="#3498db", linewidth=1.8, alpha=0.5,
            label="Current (tol=0.0005)")
    mach_prop = np.full(n, np.nan)
    for s, e, v in _seg_spans(mach_segs_proposed):
        mach_prop[s : e + 1] = v
    ax.plot(t, mach_prop, color="#e74c3c", linewidth=2, label="Proposed (plateaus)")
    ax.set_ylabel("Mach")
    ax.set_title("Mach segments", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

    # --- (1,0) CAS ---
    ax = axes[1, 0]
    ax.plot(t, cas, color="#bdc3c7", linewidth=0.6, label="CAS truth")
    ax.plot(t, cas_sel_cur, color="#3498db", linewidth=1.8, alpha=0.5,
            label="Current")
    cas_prop = np.full(n, np.nan)
    for s, e, v in _seg_spans(cas_segs_proposed):
        cas_prop[s : e + 1] = v
    ax.plot(t, cas_prop, color="#e74c3c", linewidth=2, label="Detected")
    ax.plot(t, cas_filled, color="#27ae60", linewidth=1.5, linestyle="--",
            label="Filled (optim + detect)")
    ax.set_ylabel("CAS (kt)")
    ax.set_xlabel("Time (min)")
    ax.set_title("CAS segments", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

    # --- (1,1) TAS target ---
    ax = axes[1, 1]
    ax.plot(t, tas, color="#bdc3c7", linewidth=0.6, label="TAS truth")
    ax.plot(t, tas_target_cur, color="#3498db", linewidth=1.5, alpha=0.5,
            label="Current (backfill)")
    # TAS from CAS filled (shows CAS contribution separately)
    _alt_m = df_raw["raw_alt_ft"].to_numpy() * _FT_TO_M
    _temp_k = df_raw["era_temp_K"].to_numpy()
    tas_from_cas_filled = np.where(
        np.isnan(cas_filled), np.nan,
        np.asarray(_cas_to_tas_real(cas_filled * _KT_TO_MS, _alt_m, _temp_k)) / _KT_TO_MS,
    )
    ax.plot(t, tas_from_cas_filled, color="#27ae60", linewidth=1, linestyle=":",
            alpha=0.7, label="TAS from CAS")
    ax.plot(t, tas_target_filled, color="#e74c3c", linewidth=2,
            label="Proposed (envelope)")
    # Shade unknown
    unknown = np.isnan(tas_target_filled)
    ylim = (np.nanmin(tas) * 0.9, np.nanmax(tas) * 1.05)
    ax.fill_between(t, ylim[0], ylim[1], where=unknown, alpha=0.07,
                     color="#e74c3c", label="Unknown")
    ax.set_ylim(ylim)
    ax.set_ylabel("TAS (kt)")
    ax.set_xlabel("Time (min)")
    ax.set_title("TAS target", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight", type=str, default=None, help="Flight ID")
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    parser.add_argument("--n", type=int, default=3, help="Number of random flights")
    parser.add_argument("--seed", type=int, default=99, help="Random seed for flight selection")
    args = parser.parse_args()

    delta_path = Path("data/flights.delta")
    dt = DeltaTable(str(delta_path))
    df_all = pl.from_arrow(dt.to_pyarrow_table())

    if args.flight:
        flight_ids = [args.flight]
    else:
        flights = df_all.group_by("meta_flight_id").len()
        long_flights = flights.filter(pl.col("len") > 500)
        rng = np.random.default_rng(args.seed)
        sampled = rng.choice(long_flights["meta_flight_id"].to_numpy(), size=min(args.n, len(long_flights)), replace=False)
        flight_ids = list(sampled)

    for fid in flight_ids:
        print(f"\n{'='*60}")
        print(f"Flight: {fid}")
        print(f"{'='*60}")

        df_flight = df_all.filter(pl.col("meta_flight_id") == fid)
        print(f"  Rows: {len(df_flight)}")

        # Current (also serves as preprocessed input — has fdm_alt_sel_ft)
        df_current = build_selected_params(df_flight, CURRENT_CONFIG)

        # Proposed — uses preprocessed alt plateaus from df_current
        mach_p, cas_p, tas_raw, tas_filled, tas_known, cas_filled_arr, xover = proposed_segments(df_flight, df_current)

        xover_c, xover_d = xover
        xc = f"{xover_c:.0f}" if xover_c else "N/A"
        xd = f"{xover_d:.0f}" if xover_d else "N/A"
        print(f"  Crossover: climb={xc} ft  descent={xd} ft")

        n_mach_cur = (~np.isnan(df_current["fdm_mach_sel"].to_numpy())).sum() if "fdm_mach_sel" in df_current.columns else 0
        n_mach_prop = sum(s["end_idx"] - s["start_idx"] + 1 for s in mach_p)
        n_cas_cur = (~np.isnan(df_current["fdm_cas_sel_kt"].to_numpy())).sum() if "fdm_cas_sel_kt" in df_current.columns else 0
        n_cas_prop = sum(s["end_idx"] - s["start_idx"] + 1 for s in cas_p)
        n_filled = (~np.isnan(tas_filled)).sum()

        print(f"  Mach segs:  current={n_mach_cur}  proposed={n_mach_prop}")
        print(f"  CAS segs:   current={n_cas_cur}  proposed={n_cas_prop}")
        print(f"  TAS target: current=100% (backfill)  proposed segments={tas_known.sum():.0f}/{len(tas_known)}  zone-filled={n_filled}/{len(tas_filled)}")

        out = Path(args.out) if args.out else Path(f"data/figures/crossover_{fid}.png")
        out.parent.mkdir(parents=True, exist_ok=True)

        plot_comparison(
            df_flight, df_current,
            mach_p, cas_p, tas_raw, tas_filled, tas_known,
            cas_filled_arr, xover, out,
        )


if __name__ == "__main__":
    main()
