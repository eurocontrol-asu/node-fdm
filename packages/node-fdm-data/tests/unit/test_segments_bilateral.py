from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_data.segments import (
    build_selected_params,
    detect_gamma_plateaus_from_bilat,
    detect_vz_plateaus_from_bilat,
)


def _ramp(a: float, b: float, n: int) -> np.ndarray:
    return np.linspace(a, b, n)


def _three_zone_gamma() -> tuple[np.ndarray, np.ndarray]:
    g = np.concatenate(
        [
            np.full(60, 0.05),
            _ramp(0.05, 0.0, 10),
            np.full(60, 0.0),
            _ramp(0.0, -0.04, 10),
            np.full(60, -0.04),
        ]
    )
    mask = np.zeros_like(g, dtype=bool)
    mask[70:130] = True  # exclude alt-hold middle 60 pts
    return g, mask


def test_detect_gamma_plateaus_from_bilat_two_segments() -> None:
    g, mask = _three_zone_gamma()
    segs = detect_gamma_plateaus_from_bilat(
        g,
        mask,
        sigma_s=6.0,
        sigma_r=1.2e-2,
        slope_tol=3e-4,
        flat_tol=2e-3,
        abs_min=5e-3,
        min_len=10,
    )
    assert len(segs) == 2
    means = sorted(s["var_mean"] for s in segs)
    assert abs(means[0] - (-0.04)) < 1e-3
    assert abs(means[1] - 0.05) < 1e-3


def test_detect_gamma_plateaus_from_bilat_all_excluded() -> None:
    g, _ = _three_zone_gamma()
    mask = np.ones_like(g, dtype=bool)
    segs = detect_gamma_plateaus_from_bilat(
        g,
        mask,
        sigma_s=6.0,
        sigma_r=1.2e-2,
        slope_tol=3e-4,
        flat_tol=2e-3,
        abs_min=5e-3,
        min_len=10,
    )
    assert segs == []


def test_detect_gamma_plateaus_from_bilat_below_abs_min() -> None:
    g = np.full(200, 1e-3)
    mask = np.zeros_like(g, dtype=bool)
    segs = detect_gamma_plateaus_from_bilat(
        g,
        mask,
        sigma_s=6.0,
        sigma_r=1.2e-2,
        slope_tol=3e-4,
        flat_tol=2e-3,
        abs_min=5e-3,
        min_len=10,
    )
    assert segs == []


def _three_zone_vz() -> tuple[np.ndarray, np.ndarray]:
    vz = np.concatenate(
        [
            np.full(60, -1500.0),
            _ramp(-1500.0, 0.0, 10),
            np.full(60, 0.0),
            _ramp(0.0, 1200.0, 10),
            np.full(60, 1200.0),
        ]
    )
    mask = np.zeros_like(vz, dtype=bool)
    mask[70:130] = True
    return vz, mask


def test_detect_vz_plateaus_from_bilat_two_segments() -> None:
    vz, mask = _three_zone_vz()
    segs = detect_vz_plateaus_from_bilat(
        vz,
        mask,
        sigma_s=6.0,
        sigma_r=350.0,
        slope_tol=15.0,
        flat_tol=100.0,
        min_len=10,
    )
    assert len(segs) == 2
    means = sorted(s["var_mean"] for s in segs)
    assert abs(means[0] - (-1500.0)) < 50.0
    assert abs(means[1] - 1200.0) < 50.0


def test_detect_vz_plateaus_from_bilat_partial_nan() -> None:
    rng = np.random.default_rng(0)
    vz, mask = _three_zone_vz()
    idx = rng.choice(vz.size, size=int(0.05 * vz.size), replace=False)
    vz[idx] = np.nan
    segs = detect_vz_plateaus_from_bilat(
        vz,
        mask,
        sigma_s=6.0,
        sigma_r=350.0,
        slope_tol=15.0,
        flat_tol=100.0,
        min_len=10,
    )
    assert len(segs) >= 1


# --- dispatch tests on build_selected_params -----------------------------


def _synthetic_flight_df(n_per: int = 100) -> pl.DataFrame:
    """4-zone synthetic flight: climb / level / climb / level."""
    alt = np.concatenate(
        [
            np.linspace(0, 10000, n_per),
            np.full(n_per, 10000.0),
            np.linspace(10000, 20000, n_per),
            np.full(n_per, 20000.0),
        ]
    )
    dt = 1.0
    vz_ms = np.gradient(alt * 0.3048, dt)
    vz_ftmin = vz_ms / 0.3048 * 60.0
    n = alt.size
    tas_kt = np.full(n, 230.0 / 0.514444)
    return pl.DataFrame(
        {
            "time_s": np.arange(n, dtype=float) * dt,
            "raw_alt_ft": alt,
            "raw_vz_ftmin": vz_ftmin,
            "fdm_tas_from_cas_kt": tas_kt,
            "fdm_gamma_rad": np.arcsin(
                np.clip(vz_ms / np.clip(tas_kt * 0.514444, 1e-6, None), -1.0, 1.0)
            ),
        }
    )


_BILATERAL_GAMMA_CFG = {
    "mode": "bilateral_gamma",
    "sigma_s": 6.0,
    "sigma_r": 1.2e-2,
    "slope_tol": 3e-4,
    "flat_tol": 2e-3,
    "abs_min": 5e-3,
    "min_len": 10,
}

_BILATERAL_VZ_CFG = {
    "mode": "bilateral_vz",
    "sigma_s": 6.0,
    "sigma_r": 350.0,
    "slope_tol": 15.0,
    "flat_tol": 100.0,
    "min_len": 10,
}

_BILATERAL_ALT_CFG = {
    "mode": "bilateral_vz",
    "sigma_s": 6.0,
    "sigma_r": 350.0,
    "n_passes": 2,
    "tol_ftmin": 400.0,
    "min_len": 6,
}


def test_detect_gamma_sel_bilateral_dispatch() -> None:
    df = _synthetic_flight_df()
    cfg = {
        "alt": _BILATERAL_ALT_CFG,
        "gamma": _BILATERAL_GAMMA_CFG,
        "vz": _BILATERAL_VZ_CFG,
    }
    out = build_selected_params(df, cfg)
    assert "fdm_gamma_sel_rad" in out.columns
    g = out["fdm_gamma_sel_rad"].to_numpy()
    assert np.isfinite(g).sum() > 0


def test_vz_branch_bilateral_dispatch_with_exclusion() -> None:
    df = _synthetic_flight_df()
    cfg = {
        "alt": _BILATERAL_ALT_CFG,
        "gamma": _BILATERAL_GAMMA_CFG,
        "vz": _BILATERAL_VZ_CFG,
    }
    out = build_selected_params(df, cfg)
    assert "fdm_vz_sel_ftmin" in out.columns
    vz_sel = out["fdm_vz_sel_ftmin"].to_numpy()
    alt_sel = out["fdm_alt_sel_ft"].to_numpy()
    in_alt_hold = ~np.isnan(alt_sel)
    if in_alt_hold.any():
        assert np.all(np.isnan(vz_sel[in_alt_hold]))


def test_detect_gamma_sel_savgol_backcompat() -> None:
    df = _synthetic_flight_df()
    cfg = {
        "alt": _BILATERAL_ALT_CFG,
        "gamma": {
            "mode": "savgol_gamma",
            "tol": 0.002,
            "min_len": 15,
            "min_abs_value": 0.005,
            "smooth_window": 5,
            "smooth_method": "savgol",
        },
        "vz": _BILATERAL_VZ_CFG,
    }
    out = build_selected_params(df, cfg)
    assert "fdm_gamma_sel_rad" in out.columns
    assert out["fdm_gamma_sel_rad"].dtype == pl.Float64


def test_vz_branch_savgol_backcompat() -> None:
    df = _synthetic_flight_df()
    cfg = {
        "alt": _BILATERAL_ALT_CFG,
        "gamma": _BILATERAL_GAMMA_CFG,
        "vz": {
            "mode": "savgol_vz",
            "tol": 25,
            "min_len": 25,
            "use_alt": False,
            "min_abs_value": 75,
            "smooth_window": 15,
            "smooth_method": "savgol",
        },
    }
    out = build_selected_params(df, cfg)
    assert "fdm_vz_sel_ftmin" in out.columns
    assert out["fdm_vz_sel_ftmin"].dtype == pl.Float64
