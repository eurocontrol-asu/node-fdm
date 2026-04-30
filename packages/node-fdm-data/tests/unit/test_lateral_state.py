"""Unit tests for ``node_fdm_data.preprocessing.lateral_state``."""

from __future__ import annotations

import numpy as np

from node_fdm_data.preprocessing.lateral_state import (
    clean_track_with_medfilt,
    coalesce_heading,
    compute_drift_from_wind,
    compute_wind_std,
)

# ---------------------------------------------------------------------------
# clean_track_with_medfilt
# ---------------------------------------------------------------------------


def test_clean_track_kills_isolated_spikes() -> None:
    n = 80
    base = np.full(n, 90.0)
    base[40] = 270.0  # 1-sample spike, 180° away
    base[60] = 270.0
    base[61] = 270.0  # 2-sample spike

    cleaned = clean_track_with_medfilt(base)
    # Far from the spikes, signal must remain ~90.
    assert abs(float(cleaned[10]) - 90.0) < 1.0
    assert abs(float(cleaned[70]) - 90.0) < 1.0
    # At the spike locations, the median filter must have rejected them.
    assert abs(float(cleaned[40]) - 90.0) < 5.0
    assert abs(float(cleaned[60]) - 90.0) < 5.0


def test_clean_track_preserves_smooth_ramp() -> None:
    # A 30 s ramp at 4 s sampling = 8 samples spanning 30 deg.
    n = 80
    track = np.full(n, 90.0)
    track[30:38] = np.linspace(90.0, 120.0, 8)
    track[38:] = 120.0

    cleaned = clean_track_with_medfilt(track)
    # End of ramp should be preserved within 1.5° (Savgol smoothing edge).
    assert abs(float(cleaned[60]) - 120.0) < 1.5
    assert abs(float(cleaned[10]) - 90.0) < 1.5


def test_clean_track_handles_360_boundary() -> None:
    n = 60
    # Track at 350°, with values that cross the 0/360 boundary.
    track = np.where(np.arange(n) < 30, 350.0, 10.0)
    cleaned = clean_track_with_medfilt(track)
    # Must wrap to [0, 360).
    assert (cleaned >= 0.0).all()
    assert (cleaned < 360.0).all()


def test_clean_track_short_signal_returned_wrapped() -> None:
    track = np.array([10.0, 370.0, -5.0])
    cleaned = clean_track_with_medfilt(track)
    assert cleaned.shape == (3,)
    assert (cleaned >= 0.0).all() and (cleaned < 360.0).all()


# ---------------------------------------------------------------------------
# compute_drift_from_wind
# ---------------------------------------------------------------------------


def test_drift_zero_wind_zero_drift() -> None:
    n = 10
    heading = np.full(n, 90.0)
    tas = np.full(n, 200.0)
    u = np.zeros(n)
    v = np.zeros(n)
    drift = compute_drift_from_wind(heading, tas, u, v)
    assert np.allclose(drift, 0.0)


def test_drift_pure_crosswind_matches_atan() -> None:
    # Heading = 0° (north), TAS = 200 m/s, pure east wind = 20 m/s.
    # Wind is "from the right" (going east), pushing aircraft east → track
    # west of heading, drift should be negative under our convention.
    # Cross = u * cos(0) - v * sin(0) = u = 20.
    # Along = u * sin(0) + v * cos(0) = 0.
    # drift = atan2(20, 200) ≈ 5.71°.
    heading = np.array([0.0])
    tas = np.array([200.0])
    u = np.array([20.0])
    v = np.array([0.0])
    drift = compute_drift_from_wind(heading, tas, u, v)
    assert abs(float(drift[0]) - np.degrees(np.arctan2(20.0, 200.0))) < 1e-6


# ---------------------------------------------------------------------------
# compute_wind_std
# ---------------------------------------------------------------------------


def test_wind_std_constant_wind_zero() -> None:
    n = 50
    u = np.full(n, 10.0)
    v = np.full(n, -5.0)
    wstd = compute_wind_std(u, v)
    assert np.all(wstd < 1e-9)


def test_wind_std_step_change_detected() -> None:
    n = 60
    u = np.where(np.arange(n) < 30, 0.0, 50.0)
    v = np.zeros(n)
    wstd = compute_wind_std(u, v)
    # Around the step, std should be large.
    assert wstd[28:32].max() > 5.0
    # Far from the step, std should be small.
    assert wstd[5] < 1e-9
    assert wstd[55] < 1e-9


# ---------------------------------------------------------------------------
# coalesce_heading
# ---------------------------------------------------------------------------


def _const(x: float, n: int) -> np.ndarray:
    return np.full(n, x, dtype=np.float64)


def test_coalesce_uses_bds_when_available() -> None:
    n = 5
    bds = _const(90.0, n)
    decl = _const(2.0, n)
    track = _const(95.0, n)
    drift = _const(3.0, n)
    wstd = _const(0.5, n)
    heading, known = coalesce_heading(bds, decl, track, drift, wstd)
    assert known.all()
    # bds + decl = 92, modulo 360.
    assert np.allclose(heading, 92.0)


def test_coalesce_fallback_when_bds_missing_and_wind_ok() -> None:
    n = 5
    bds = _const(np.nan, n)
    decl = _const(2.0, n)
    track = _const(100.0, n)
    drift = _const(5.0, n)
    wstd = _const(0.5, n)
    heading, known = coalesce_heading(bds, decl, track, drift, wstd)
    assert known.all()
    assert np.allclose(heading, 95.0)  # 100 - 5


def test_coalesce_known_false_when_wind_unstable() -> None:
    n = 5
    bds = _const(np.nan, n)
    decl = _const(2.0, n)
    track = _const(100.0, n)
    drift = _const(5.0, n)
    wstd = _const(8.0, n)  # > threshold 5
    heading, known = coalesce_heading(bds, decl, track, drift, wstd)
    assert not known.any()
    assert np.isnan(heading).all()


def test_coalesce_known_false_when_fallback_inputs_nan() -> None:
    n = 5
    bds = _const(np.nan, n)
    decl = _const(2.0, n)
    track = _const(np.nan, n)
    drift = _const(5.0, n)
    wstd = _const(0.5, n)
    heading, known = coalesce_heading(bds, decl, track, drift, wstd)
    assert not known.any()
    assert np.isnan(heading).all()


def test_coalesce_mixed_sources_per_sample() -> None:
    bds = np.array([10.0, np.nan, 20.0, np.nan, np.nan])
    decl = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    track = np.array([50.0, 50.0, 50.0, 50.0, 50.0])
    drift = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    wstd = np.array([0.1, 0.1, 0.1, 9.0, 0.1])
    heading, known = coalesce_heading(bds, decl, track, drift, wstd)
    assert known.tolist() == [True, True, True, False, True]
    assert abs(heading[0] - 11.0) < 1e-9
    assert abs(heading[1] - 45.0) < 1e-9  # fallback 50-5
    assert abs(heading[2] - 21.0) < 1e-9
    assert np.isnan(heading[3])
    assert abs(heading[4] - 45.0) < 1e-9
