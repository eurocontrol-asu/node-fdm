"""Bench _optimize_transition_cas: brute-force vs scipy.minimize_scalar.

Loads real climb/descent windows from data/flights.delta (after segments step
has run), reconstructs the inputs of _optimize_transition_cas, and times both
the current implementation and a scipy-based replacement on each window.
Reports speedup and CAS-optimum agreement.
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

from node_fdm_data.physics.speed import cas_to_tas_real, mach_to_tas_real
from node_fdm_data.segments import _optimize_transition_cas

_KT_TO_MS = 0.5144444444444445
_FT_TO_M = 0.3048


def cost_fn(cas_kt: float, tas_real_ms, mach_const, h, t):
    tas_mach_ms = np.asarray(mach_to_tas_real(mach_const, t), dtype=np.float64)
    tas_cas_ms = np.asarray(
        cas_to_tas_real(cas_kt * _KT_TO_MS, h, t), dtype=np.float64
    )
    env = np.minimum(tas_mach_ms, tas_cas_ms)
    diff = env - tas_real_ms
    return float(np.nansum(diff * diff))


def vectorized_optimum(tas_real_kt, mach_const, alt_m, temp_k):
    """Same algorithm as _optimize_transition_cas, but cost evaluated in one
    broadcast pass per stage instead of a Python loop over the grid."""
    valid = ~np.isnan(tas_real_kt) & ~np.isnan(alt_m) & ~np.isnan(temp_k)
    if int(valid.sum()) < 2:
        return float("nan")
    tas_real_ms = tas_real_kt[valid] * _KT_TO_MS  # (M,)
    h = alt_m[valid]
    t = temp_k[valid]

    tas_mach_ms = np.asarray(mach_to_tas_real(mach_const, t), dtype=np.float64)  # (M,)

    def grid_cost(grid_kt: np.ndarray) -> np.ndarray:
        # grid_kt: (G,) -> tas_cas_ms: (G, M)
        cas_ms = grid_kt[:, None] * _KT_TO_MS  # (G, 1)
        h_b = h[None, :]
        t_b = t[None, :]
        tas_cas_ms = np.asarray(cas_to_tas_real(cas_ms, h_b, t_b), dtype=np.float64)
        env = np.minimum(tas_mach_ms[None, :], tas_cas_ms)
        diff = env - tas_real_ms[None, :]
        return np.nansum(diff * diff, axis=1)

    coarse = np.arange(200.0, 351.0, 1.0)
    coarse_costs = grid_cost(coarse)
    if not np.any(np.isfinite(coarse_costs)):
        return float("nan")
    best = float(coarse[int(np.argmin(coarse_costs))])
    fine = np.arange(best - 2.0, best + 2.0 + 0.05, 0.1)
    fine_costs = grid_cost(fine)
    return float(fine[int(np.argmin(fine_costs))])


def scipy_optimum(tas_real_kt, mach_const, alt_m, temp_k):
    valid = ~np.isnan(tas_real_kt) & ~np.isnan(alt_m) & ~np.isnan(temp_k)
    if int(valid.sum()) < 2:
        return float("nan")
    tas_real_ms = tas_real_kt[valid] * _KT_TO_MS
    h = alt_m[valid]
    t = temp_k[valid]

    res = minimize_scalar(
        cost_fn,
        args=(tas_real_ms, mach_const, h, t),
        bounds=(200.0, 350.0),
        method="bounded",
        options={"xatol": 0.1},
    )
    return float(res.x) if res.success else float("nan")


def build_windows(df: pl.DataFrame, n_max: int = 30):
    """Extract realistic climb/descent windows per flight."""
    windows = []
    flights = df.partition_by("meta_flight_id", maintain_order=True)
    for fdf in flights:
        if "fdm_tas_from_cas_kt" not in fdf.columns:
            continue
        tas = fdf["fdm_tas_from_cas_kt"].to_numpy().astype(np.float64)
        alt_ft = fdf["raw_alt_ft"].to_numpy().astype(np.float64)
        if "era_temp_K" not in fdf.columns:
            continue
        temp = fdf["era_temp_K"].to_numpy().astype(np.float64)
        mach = fdf["bds_mach"].to_numpy().astype(np.float64)
        if np.all(np.isnan(tas)) or np.all(np.isnan(mach)):
            continue
        n = len(fdf)
        # Approximate: take first 60-300 (climb) and last 60-300 (descent)
        for win_slice in (slice(0, min(300, n // 3)), slice(max(0, n - 300), n)):
            t_w = tas[win_slice]
            if np.sum(~np.isnan(t_w)) < 10:
                continue
            mach_const_arr = mach[win_slice]
            mach_valid = mach_const_arr[~np.isnan(mach_const_arr)]
            if len(mach_valid) == 0:
                continue
            mach_const = float(np.median(mach_valid))
            if not (0.5 < mach_const < 0.9):
                continue
            windows.append(
                {
                    "tas_real_kt": t_w,
                    "mach_const": mach_const,
                    "alt_m": alt_ft[win_slice] * _FT_TO_M,
                    "temp_k": temp[win_slice],
                }
            )
        if len(windows) >= n_max:
            break
    return windows[:n_max]


def main():
    df = pl.read_delta("data/flights.delta")
    print(f"Loaded {len(df)} rows, {df['meta_flight_id'].n_unique()} flights")

    windows = build_windows(df, n_max=30)
    print(f"Built {len(windows)} windows for benchmark\n")

    if not windows:
        print("No usable windows. Has segments / clean-speeds run on this delta?")
        return

    brute_times = []
    vect_times = []
    scipy_times = []
    vect_diffs = []
    scipy_diffs = []

    for i, w in enumerate(windows):
        t0 = time.perf_counter()
        cas_brute = _optimize_transition_cas(
            w["tas_real_kt"], w["mach_const"], w["alt_m"], w["temp_k"]
        )
        brute_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cas_vect = vectorized_optimum(
            w["tas_real_kt"], w["mach_const"], w["alt_m"], w["temp_k"]
        )
        vect_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cas_scipy = scipy_optimum(
            w["tas_real_kt"], w["mach_const"], w["alt_m"], w["temp_k"]
        )
        scipy_times.append(time.perf_counter() - t0)

        if np.isfinite(cas_brute) and np.isfinite(cas_vect):
            vect_diffs.append(abs(cas_brute - cas_vect))
        if np.isfinite(cas_brute) and np.isfinite(cas_scipy):
            scipy_diffs.append(abs(cas_brute - cas_scipy))
        vtag = (
            "OK"
            if np.isfinite(cas_vect)
            and np.isfinite(cas_brute)
            and abs(cas_brute - cas_vect) < 0.5
            else "DIVERGE"
        )
        print(
            f"[{i:02d}] brute={cas_brute:7.2f}  vect={cas_vect:7.2f}  scipy={cas_scipy:7.2f}"
            f"  bt={brute_times[-1]*1000:5.1f}ms"
            f"  vt={vect_times[-1]*1000:5.1f}ms"
            f"  st={scipy_times[-1]*1000:5.1f}ms  vect:{vtag}"
        )

    print("\n=== SUMMARY ===")
    print(f"windows                  : {len(windows)}")
    print(f"brute total              : {sum(brute_times)*1000:8.1f} ms")
    print(f"vectorized total         : {sum(vect_times)*1000:8.1f} ms")
    print(f"scipy total              : {sum(scipy_times)*1000:8.1f} ms")
    print(f"speedup vectorized       : {sum(brute_times)/sum(vect_times):.1f}x")
    print(f"speedup scipy            : {sum(brute_times)/sum(scipy_times):.1f}x")
    if vect_diffs:
        print(
            f"|brute-vect| optimum     : "
            f"median={np.median(vect_diffs):.4f}kt max={np.max(vect_diffs):.4f}kt"
        )
    if scipy_diffs:
        print(
            f"|brute-scipy| optimum    : "
            f"median={np.median(scipy_diffs):.3f}kt max={np.max(scipy_diffs):.3f}kt"
        )


if __name__ == "__main__":
    main()
