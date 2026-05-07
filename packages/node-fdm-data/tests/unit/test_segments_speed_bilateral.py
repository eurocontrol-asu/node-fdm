"""Tests for bilateral+Butterworth Mach/CAS plateau detection and pointwise propagation.

AXM-1689: covers the new public detectors
(``detect_mach_plateaus_bilat``, ``detect_cas_plateaus_bilat``), config
defaults, and pointwise speed propagation inside Mach/CAS plateaus exercised
through ``build_selected_params``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.physics.speed import mach_to_tas_real
from node_fdm_data.segments import (
    build_selected_params,
    detect_cas_plateaus_bilat,
    detect_mach_plateaus_bilat,
)

_MS_TO_KT = 3600.0 / 1852.0
_FT_TO_M = 0.3048


def _isa_temp_k(alt_ft: np.ndarray) -> np.ndarray:
    """ISA static temperature for the troposphere — sufficient for synthetic tests."""
    h_m = alt_ft * _FT_TO_M
    return 288.15 - 0.0065 * h_m


# ---------------------------------------------------------------------------
# detect_cas_plateaus_bilat
# ---------------------------------------------------------------------------


def test_detect_cas_plateaus_bilat_constant_signal() -> None:
    """AC1, AC3 — Butterworth+bilateral pipeline detects a noisy constant CAS plateau."""
    rng = np.random.default_rng(0)
    cas = 280.0 + rng.normal(0.0, 1.0, size=200)
    mach_mask = np.zeros(200, dtype=bool)
    segs = detect_cas_plateaus_bilat(
        cas,
        mach_mask,
        cutoff_s=180.0,
        sigma_s=8.0,
        sigma_r=15.0,
        n_passes=2,
        slope_tol=0.25,
        flat_tol=20.0,
        min_len=5,
    )
    assert len(segs) >= 1
    main = max(segs, key=lambda s: s["end_idx"] - s["start_idx"])
    assert abs(main["var_mean"] - 280.0) < 1.0


def test_detect_cas_plateaus_bilat_excludes_mach() -> None:
    """AC3 — segments overlapping the mach_mask region are dropped."""
    cas = np.full(200, 280.0)
    mach_mask = np.zeros(200, dtype=bool)
    mach_mask[50:150] = True
    segs = detect_cas_plateaus_bilat(
        cas,
        mach_mask,
        cutoff_s=180.0,
        sigma_s=8.0,
        sigma_r=15.0,
        n_passes=2,
        slope_tol=0.25,
        flat_tol=20.0,
        min_len=5,
    )
    for seg in segs:
        idx = np.arange(seg["start_idx"], seg["end_idx"] + 1)
        assert not mach_mask[idx].any(), "segment leaked into mach-excluded zone"


# ---------------------------------------------------------------------------
# detect_mach_plateaus_bilat
# ---------------------------------------------------------------------------


def test_detect_mach_plateaus_bilat_with_alt_gate_pass() -> None:
    """AC2 — Mach plateau whose endpoints lie inside an alt segment is accepted."""
    n = 200
    mach = np.full(n, 0.78)
    alt_mask = np.zeros(n, dtype=bool)
    alt_mask[50:150] = True  # alt plateau [50, 149]
    segs = detect_mach_plateaus_bilat(
        mach,
        alt_mask,
        sigma_s=8.0,
        sigma_r=0.08,
        n_passes=2,
        slope_tol=6.5e-4,
        flat_tol=5e-2,
        min_len=15,
    )
    assert len(segs) >= 1


def test_detect_mach_plateaus_bilat_with_alt_gate_fail() -> None:
    """AC2 — Mach plateau disjoint from any alt plateau is rejected."""
    n = 200
    # Smooth ramp outside the central cruise zone so no plateau is detected
    # there; only the constant-Mach window [40, 159] forms a plateau.
    mach = np.concatenate(
        [np.linspace(0.30, 0.78, 40), np.full(120, 0.78), np.linspace(0.78, 0.30, 40)]
    )
    alt_mask = np.zeros(n, dtype=bool)
    alt_mask[170:195] = True  # alt plateau strictly outside the mach plateau
    segs = detect_mach_plateaus_bilat(
        mach,
        alt_mask,
        sigma_s=8.0,
        sigma_r=0.08,
        n_passes=2,
        slope_tol=6.5e-4,
        flat_tol=5e-2,
        min_len=15,
    )
    assert segs == []


# ---------------------------------------------------------------------------
# Config defaults (AC4)
# ---------------------------------------------------------------------------


def test_mach_filter_config_default_mode_bilateral() -> None:
    from node_fdm_pipeline.config import MachFilterConfig

    cfg = MachFilterConfig()
    assert cfg.mode == "bilateral_mach"
    assert cfg.sigma_s == pytest.approx(8.0)
    assert cfg.sigma_r == pytest.approx(0.08)
    assert cfg.slope_tol == pytest.approx(6.5e-4)
    assert cfg.flat_tol == pytest.approx(5e-2)
    assert cfg.min_len == 15


def test_cas_filter_config_default_mode_bilateral() -> None:
    from node_fdm_pipeline.config import CasFilterConfig

    cfg = CasFilterConfig()
    assert cfg.mode == "bilateral_cas"
    assert cfg.cutoff_s == pytest.approx(180.0)
    assert cfg.sigma_s == pytest.approx(8.0)
    assert cfg.sigma_r == pytest.approx(15.0)
    assert cfg.slope_tol == pytest.approx(0.25)
    assert cfg.flat_tol == pytest.approx(20.0)
    assert cfg.min_len == 5


# ---------------------------------------------------------------------------
# Pointwise propagation through build_selected_params (AC6, AC7, AC8)
# ---------------------------------------------------------------------------


def _synthetic_flight(
    *,
    n: int,
    mach: np.ndarray | None,
    cas_kt: np.ndarray | None,
    alt_ft: np.ndarray,
) -> pl.DataFrame:
    """Build a minimal single-flight DataFrame with all columns required by
    ``build_selected_params``.

    Provides realistic ISA temperature and a stable derived TAS column.
    """
    temp_k = _isa_temp_k(alt_ft)
    if mach is None:
        mach = np.full(n, np.nan)
    if cas_kt is None:
        cas_kt = np.full(n, np.nan)
    # fdm_tas_from_cas_kt — derived TAS used by the existing pipeline.
    # We seed it from Mach when available (best-effort) so downstream
    # consumers do not see all-NaN.
    tas_from_cas = np.where(
        np.isfinite(mach),
        mach_to_tas_real(np.where(np.isfinite(mach), mach, 0.0), temp_k) * _MS_TO_KT,
        np.nan,
    )
    return pl.DataFrame(
        {
            "raw_alt_ft": alt_ft,
            "raw_vz_ftmin": np.zeros(n),
            "bds_mach_clean": mach,
            "bds_ias_kt_clean": cas_kt,
            "fdm_tas_from_cas_kt": tas_from_cas,
            "era_temp_K": temp_k,
        }
    )


def _bilateral_default_config() -> dict[str, dict[str, object]]:
    return {
        "mach": {
            "mode": "bilateral_mach",
            "sigma_s": 8.0,
            "sigma_r": 0.08,
            "n_passes": 2,
            "slope_tol": 6.5e-4,
            "flat_tol": 5e-2,
            "min_len": 15,
        },
        "cas": {
            "mode": "bilateral_cas",
            "cutoff_s": 180.0,
            "sigma_s": 8.0,
            "sigma_r": 15.0,
            "n_passes": 2,
            "slope_tol": 0.25,
            "flat_tol": 20.0,
            "min_len": 5,
        },
        "alt": {
            "tol": 50.0,
            "min_len": 30,
            "use_alt": False,
            "smooth_window": 10,
        },
        "vz": {"tol": 50.0, "min_len": 10, "use_alt": False, "smooth_window": 10},
    }


def test_propagate_speed_plateaus_mach_to_cas_decreases_with_altitude() -> None:
    """AC6, AC7, AC8 — within a constant-Mach plateau, propagated CAS strictly
    decreases when altitude increases, and TAS matches mach_to_tas_real(M,T)."""
    n = 200
    mach = np.full(n, 0.78)
    alt_ft = np.linspace(30000.0, 38000.0, n)
    df = _synthetic_flight(n=n, mach=mach, cas_kt=None, alt_ft=alt_ft)
    out = build_selected_params(df, _bilateral_default_config())

    mach_sel = out["fdm_mach_sel"].to_numpy()
    cas_sel = out["fdm_cas_sel_kt"].to_numpy()
    tas_sel = out["fdm_tas_sel_kt"].to_numpy()

    plateau = np.isfinite(mach_sel)
    assert plateau.sum() >= 50, "Mach plateau not detected over varying altitude"

    # AC8: Mach is constant inside the plateau
    np.testing.assert_allclose(mach_sel[plateau], 0.78, atol=1e-6)
    # AC8: TAS varies with altitude (temperature changes)
    finite_tas = tas_sel[plateau]
    assert np.isfinite(finite_tas).all()
    assert finite_tas.max() - finite_tas.min() > 0.1
    # AC6: TAS matches mach_to_tas_real(M, T) * MS_TO_KT within tolerance
    expected_tas = mach_to_tas_real(0.78, _isa_temp_k(alt_ft[plateau])) * _MS_TO_KT
    np.testing.assert_allclose(finite_tas, expected_tas, atol=0.5)
    # AC8: CAS decreases monotonically with altitude inside the plateau
    finite_cas = cas_sel[plateau]
    assert np.isfinite(finite_cas).all()
    diffs = np.diff(finite_cas)
    assert (diffs <= 1e-6).all(), "CAS should be non-increasing with altitude"
    assert finite_cas[0] - finite_cas[-1] > 1.0


def test_propagate_speed_plateaus_cas_to_mach_increases_with_altitude() -> None:
    """AC6, AC7 — within a constant-CAS plateau, propagated Mach strictly
    increases when altitude increases."""
    n = 200
    cas_kt = np.full(n, 280.0)
    alt_ft = np.linspace(5000.0, 15000.0, n)
    df = _synthetic_flight(n=n, mach=None, cas_kt=cas_kt, alt_ft=alt_ft)
    out = build_selected_params(df, _bilateral_default_config())

    cas_sel = out["fdm_cas_sel_kt"].to_numpy()
    mach_sel = out["fdm_mach_sel"].to_numpy()

    plateau = np.isfinite(cas_sel)
    assert plateau.sum() >= 50, "CAS plateau not detected"
    np.testing.assert_allclose(cas_sel[plateau], 280.0, atol=1e-6)
    finite_mach = mach_sel[plateau]
    assert np.isfinite(finite_mach).all()
    diffs = np.diff(finite_mach)
    assert (diffs >= -1e-6).all(), "Mach should be non-decreasing with altitude"
    assert finite_mach[-1] - finite_mach[0] > 1e-3


# ---------------------------------------------------------------------------
# Backwards-compat: legacy savgol mode still routes through the detector (AC5)
# ---------------------------------------------------------------------------


def test_detect_mach_savgol_backcompat() -> None:
    """AC5 — selecting ``mode='savgol_mach'`` runs the legacy detector and still
    populates ``fdm_mach_sel`` for a clean constant-Mach segment."""
    n = 200
    mach = np.full(n, 0.78)
    alt_ft = np.full(n, 35000.0)
    df = _synthetic_flight(n=n, mach=mach, cas_kt=None, alt_ft=alt_ft)
    cfg = _bilateral_default_config()
    cfg["mach"] = {
        "mode": "savgol_mach",
        "tol": 0.0005,
        "min_len": 30,
        "alt_threshold": 15000,
        "smooth_window": 30,
        "use_alt": True,
    }
    out = build_selected_params(df, cfg)
    mach_sel = out["fdm_mach_sel"].to_numpy()
    assert np.isfinite(mach_sel).sum() > 0


def test_detect_cas_savgol_backcompat() -> None:
    """AC5 — selecting ``mode='savgol_cas'`` runs the legacy detector and still
    populates ``fdm_cas_sel_kt`` for a clean constant-CAS segment."""
    n = 200
    cas_kt = np.full(n, 280.0)
    alt_ft = np.full(n, 10000.0)
    df = _synthetic_flight(n=n, mach=None, cas_kt=cas_kt, alt_ft=alt_ft)
    cfg = _bilateral_default_config()
    cfg["cas"] = {
        "mode": "savgol_cas",
        "tol": 0.75,
        "min_len": 20,
        "use_alt": False,
        "smooth_window": 20,
        "smooth_method": "savgol",
    }
    out = build_selected_params(df, cfg)
    cas_sel = out["fdm_cas_sel_kt"].to_numpy()
    assert np.isfinite(cas_sel).sum() > 0
