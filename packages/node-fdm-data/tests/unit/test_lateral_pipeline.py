"""Unit tests for the lateral channel data pipeline.

Builds a synthetic single-flight DataFrame in memory (no Delta Table, no real
I/O) and exercises ``derive_columns`` + ``convert_si`` end-to-end on the
lateral columns.  The fixture is a 60-sample straight-cruise flight with zero
wind, picked so that ``fdm_heading_known`` coverage is well above the 30 %
floor that the integration suite used to demand.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from node_fdm_data.preprocessing.convert import convert_si
from node_fdm_data.preprocessing.derive import derive_columns

_N = 120
_DT_S = 4.0
# Two turns flank a long straight segment — _build_in_turn_mask treats samples
# before the first turn and after the last turn as "no enclosing segment", so
# at least two turns are required to expose a finite-ortho region in the middle.
_TURN_1_CENTER = 30
_TURN_2_CENTER = 90
_TURN_HALFWIDTH = 3


def _build_track() -> np.ndarray:
    """Track signal with two ~30° turns flanking a long straight cruise."""
    track = np.full(_N, 90.0)
    for center, hi_track in ((_TURN_1_CENTER, 120.0), (_TURN_2_CENTER, 90.0)):
        ramp_lo = center - _TURN_HALFWIDTH
        ramp_hi = center + _TURN_HALFWIDTH
        track[ramp_lo:ramp_hi] = np.linspace(track[ramp_lo - 1], hi_track, ramp_hi - ramp_lo)
        track[ramp_hi:] = hi_track
    return track


@pytest.fixture
def straight_cruise_flight() -> pl.DataFrame:
    """Synthetic single-flight cruise with a 30° turn near the middle.

    80 samples @ 4 s, FL350, 450 kt TAS, zero wind, BDS heading aligned with
    track.  The turn is wide enough for ``detect_turning_starts`` to register
    a peak so that the enclosing-segment logic produces finite
    ``fdm_track_ortho_deg`` and ``fdm_heading_target_deg`` values.
    """
    t0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    timestamps = [t0 + timedelta(seconds=_DT_S * i) for i in range(_N)]

    track = _build_track()

    # Position: integrate ground speed along the (varying) track direction.
    lat0 = 45.0
    lon0 = 5.0
    speed_ms = 450.0 * 0.514444  # kt → m/s
    cos_lat = np.cos(np.radians(lat0))
    lats = np.empty(_N)
    lons = np.empty(_N)
    lats[0] = lat0
    lons[0] = lon0
    for i in range(1, _N):
        bearing_rad = np.radians(track[i - 1])
        d_north_m = speed_ms * _DT_S * np.cos(bearing_rad)
        d_east_m = speed_ms * _DT_S * np.sin(bearing_rad)
        lats[i] = lats[i - 1] + d_north_m / 111_320.0
        lons[i] = lons[i - 1] + d_east_m / (111_320.0 * cos_lat)

    return pl.DataFrame(
        {
            "raw_timestamp": timestamps,
            "raw_lat_deg": lats.tolist(),
            "raw_lon_deg": lons.tolist(),
            "raw_alt_ft": [35_000.0] * _N,
            "raw_track_deg": track.tolist(),
            "raw_vz_ftmin": [0.0] * _N,
            "raw_gs_kt": [450.0] * _N,
            "fdm_tas_from_cas_kt": [450.0] * _N,
            "bds_mcp_alt_sel_ft": [35_000.0] * _N,
            "bds_hdg_deg": track.tolist(),  # zero wind → heading == track
            "era_u_wind_ms": [0.0] * _N,
            "era_v_wind_ms": [0.0] * _N,
            "era_tas_kt": [450.0] * _N,
            "meta_flight_id": ["TEST_FLIGHT_s0"] * _N,
            "meta_departure": ["LFPG"] * _N,
            "meta_arrival": ["EGLL"] * _N,
        }
    )


def test_lateral_columns_present_after_derive(straight_cruise_flight: pl.DataFrame) -> None:
    out = derive_columns(straight_cruise_flight)

    expected = {
        "fdm_track_clean_deg",
        "fdm_declination_deg",
        "fdm_drift_deg",
        "fdm_wind_std_ms",
        "fdm_heading_deg",
        "fdm_heading_target_deg",
        "fdm_heading_known",
        "fdm_heading_target_known",
        "fdm_in_turn",
        "fdm_track_ortho_deg",
        "fdm_track_sel_known",
    }
    assert expected <= set(out.columns), f"missing: {expected - set(out.columns)}"


def test_lateral_si_conversions(straight_cruise_flight: pl.DataFrame) -> None:
    out = convert_si(derive_columns(straight_cruise_flight))

    assert "fdm_heading_rad" in out.columns
    assert "fdm_heading_target_rad" in out.columns

    heading_rad = out["fdm_heading_rad"].to_numpy()
    finite = heading_rad[np.isfinite(heading_rad)]
    assert finite.size > 0
    assert (finite >= 0.0).all() and (finite < 2.0 * np.pi + 1e-9).all()

    target_rad = out["fdm_heading_target_rad"].to_numpy()
    finite_t = target_rad[np.isfinite(target_rad)]
    assert finite_t.size > 0
    assert (finite_t >= -np.pi - 1e-9).all() and (finite_t <= np.pi + 1e-9).all()


def test_lateral_heading_coverage(straight_cruise_flight: pl.DataFrame) -> None:
    out = derive_columns(straight_cruise_flight)
    mean_val = out["fdm_heading_known"].mean()
    coverage = float(mean_val) if isinstance(mean_val, (int, float)) else 0.0
    # Straight cruise with BDS heading + zero wind should resolve heading on
    # nearly every sample (allow a small margin for Savgol edges).
    assert coverage > 0.3, f"heading_known coverage too low: {coverage:.2%}"


def test_lateral_target_in_principal_branch(straight_cruise_flight: pl.DataFrame) -> None:
    out = convert_si(derive_columns(straight_cruise_flight))
    target_rad = out["fdm_heading_target_rad"].to_numpy()
    finite = target_rad[np.isfinite(target_rad)]
    assert finite.size > 0
    assert np.abs(finite).max() <= np.pi + 1e-9
