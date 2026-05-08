"""Tests for node_fdm_data.preprocessing.derive — pipeline v3 étape 4."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from node_fdm_data.physics.constants import FTMIN, KT
from node_fdm_data.preprocessing.convert import convert_si
from node_fdm_data.preprocessing.derive import derive_columns


class TestDeriveGamma:
    """fdm_gamma_rad = arcsin(raw_vz_ftmin * FTMIN / (fdm_tas_from_cas_kt * KT))."""

    def test_derive_gamma(self) -> None:
        """TAS=250kt, vz=1000ft/min → correct fdm_gamma_rad."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [1000.0],
                "fdm_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_alt_sel_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
            }
        )
        result = derive_columns(df)
        assert "fdm_gamma_rad" in result.columns
        expected = np.arcsin((1000 * FTMIN) / (250 * KT))
        assert result["fdm_gamma_rad"][0] == pytest.approx(expected)

    def test_gamma_tas_near_zero(self) -> None:
        """TAS ≈ 0 → fdm_gamma_rad is clipped, no inf."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [1000.0],
                "fdm_tas_from_cas_kt": [0.001],
                "raw_gs_kt": [0.0],
                "raw_alt_ft": [1000.0],
                "bds_mcp_alt_sel_ft": [2000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
            }
        )
        result = derive_columns(df)
        gamma = result["fdm_gamma_rad"][0]
        assert np.isfinite(gamma)
        # Clipped ratio → arcsin(1.0) = π/2
        assert abs(gamma) <= np.pi / 2 + 1e-6


class TestDeriveScalarSignals:
    """Scalar derived signals from raw inputs (TAS=250, GS=240, alt=35000, sel=36000)."""

    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            pytest.param("fdm_long_wind_kt", 10.0, id="long_wind=tas-gs"),
            pytest.param("fdm_alt_diff_ft", 1000.0, id="alt_diff=sel-alt"),
        ],
    )
    def test_derive_scalar_signal(self, column: str, expected: float) -> None:
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "fdm_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_alt_sel_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
            }
        )
        result = derive_columns(df)
        assert result[column][0] == pytest.approx(expected)


class TestDeriveDistanceCum:
    """fdm_distance_cum_m = haversine cumulative per flight."""

    def test_derive_distance_cum(self) -> None:
        """3 GPS points → correct cumulative haversine distance."""
        from node_fdm_data.meteo import haversine

        lats = [48.0, 48.01, 48.02]
        lons = [2.0, 2.0, 2.0]
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0] * 3,
                "fdm_tas_from_cas_kt": [250.0] * 3,
                "raw_gs_kt": [240.0] * 3,
                "raw_alt_ft": [35000.0] * 3,
                "bds_mcp_alt_sel_ft": [36000.0] * 3,
                "raw_lat_deg": lats,
                "raw_lon_deg": lons,
                "meta_flight_id": ["F001"] * 3,
            }
        )
        result = derive_columns(df)
        assert "fdm_distance_cum_m" in result.columns
        d = result["fdm_distance_cum_m"].to_list()
        assert d[0] == 0.0

        # Verify against manual haversine
        d01 = haversine(
            np.array([lats[0]]),
            np.array([lons[0]]),
            np.array([lats[1]]),
            np.array([lons[1]]),
        )[0]
        d12 = haversine(
            np.array([lats[1]]),
            np.array([lons[1]]),
            np.array([lats[2]]),
            np.array([lons[2]]),
        )[0]
        assert d[1] == pytest.approx(d01, rel=1e-6)
        assert d[2] == pytest.approx(d01 + d12, rel=1e-6)

    def test_derive_per_flight(self) -> None:
        """2 flights in DataFrame → distance resets to 0 for each flight."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0] * 4,
                "fdm_tas_from_cas_kt": [250.0] * 4,
                "raw_gs_kt": [240.0] * 4,
                "raw_alt_ft": [35000.0] * 4,
                "bds_mcp_alt_sel_ft": [36000.0] * 4,
                "raw_lat_deg": [48.0, 48.01, 49.0, 49.01],
                "raw_lon_deg": [2.0, 2.0, 3.0, 3.0],
                "meta_flight_id": ["F001", "F001", "F002", "F002"],
            }
        )
        result = derive_columns(df)
        d = result["fdm_distance_cum_m"].to_list()
        # Flight F001 starts at 0
        assert d[0] == 0.0
        assert d[1] > 0.0
        # Flight F002 restarts at 0
        assert d[2] == 0.0
        assert d[3] > 0.0


class TestDeriveAirportDistances:
    """fdm_adep_dist_nm / fdm_ades_dist_nm via airport_coords."""

    def test_airport_distances_with_coords(self) -> None:
        """Airport distances computed when airport_coords provided."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "fdm_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_alt_sel_ft": [36000.0],
                "raw_lat_deg": [48.8566],
                "raw_lon_deg": [2.3522],
                "meta_flight_id": ["F001"],
                "meta_departure": ["LFPG"],
                "meta_arrival": ["KJFK"],
            }
        )
        # CDG ≈ (49.0097, 2.5479), JFK ≈ (40.6413, -73.7781)
        airport_coords = {
            "LFPG": (49.0097, 2.5479),
            "KJFK": (40.6413, -73.7781),
        }
        result = derive_columns(df, airport_coords=airport_coords)
        assert "fdm_adep_dist_nm" in result.columns
        assert "fdm_ades_dist_nm" in result.columns
        # ADEP (CDG) is close → small distance
        assert result["fdm_adep_dist_nm"][0] < 20.0
        # ADES (JFK) is far → large distance
        assert result["fdm_ades_dist_nm"][0] > 3000.0

    def test_airport_distances_null_departure(self) -> None:
        """meta_departure = null → fdm_adep_dist_nm = NaN."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "fdm_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_alt_sel_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
                "meta_departure": [None],
                "meta_arrival": ["KJFK"],
            }
        )
        airport_coords = {"KJFK": (40.6413, -73.7781)}
        result = derive_columns(df, airport_coords=airport_coords)
        assert result["fdm_adep_dist_nm"][0] is None or np.isnan(result["fdm_adep_dist_nm"][0])

    def test_airport_distances_no_coords(self) -> None:
        """No airport_coords → columns filled with null."""
        df = pl.DataFrame(
            {
                "raw_vz_ftmin": [0.0],
                "fdm_tas_from_cas_kt": [250.0],
                "raw_gs_kt": [240.0],
                "raw_alt_ft": [35000.0],
                "bds_mcp_alt_sel_ft": [36000.0],
                "raw_lat_deg": [48.0],
                "raw_lon_deg": [2.0],
                "meta_flight_id": ["F001"],
                "meta_departure": ["LFPG"],
                "meta_arrival": ["KJFK"],
            }
        )
        result = derive_columns(df)
        assert "fdm_adep_dist_nm" in result.columns
        assert "fdm_ades_dist_nm" in result.columns
        assert result["fdm_adep_dist_nm"].null_count() == 1
        assert result["fdm_ades_dist_nm"].null_count() == 1


# ---------------------------------------------------------------------------
# Lateral channel — synthetic single-flight scenarios
# ---------------------------------------------------------------------------

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
    """Synthetic single-flight cruise with two ~30° turns.

    120 samples @ 4 s, FL350, 450 kt TAS, zero wind, BDS heading aligned with
    track. The turns are wide enough for ``detect_turning_starts`` to register
    peaks so the enclosing-segment logic produces finite
    ``fdm_track_ortho_deg`` and ``fdm_heading_target_deg`` values.
    """
    t0 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    timestamps = [t0 + timedelta(seconds=_DT_S * i) for i in range(_N)]

    track = _build_track()

    lat0 = 45.0
    lon0 = 5.0
    speed_ms = 450.0 * 0.514444
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
            "bds_hdg_deg": track.tolist(),
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
    assert coverage > 0.3, f"heading_known coverage too low: {coverage:.2%}"


def test_lateral_target_in_principal_branch(straight_cruise_flight: pl.DataFrame) -> None:
    out = convert_si(derive_columns(straight_cruise_flight))
    target_rad = out["fdm_heading_target_rad"].to_numpy()
    finite = target_rad[np.isfinite(target_rad)]
    assert finite.size > 0
    assert np.abs(finite).max() <= np.pi + 1e-9


def test_lateral_target_defined_inside_turn(straight_cruise_flight: pl.DataFrame) -> None:
    """AC9: V3 bfill makes fdm_heading_target_known True inside detected turns.

    Pre-V3 the flag was False on every in-turn sample (track_ortho was NaN). With
    the new bfill, samples inside a turn inherit the next straight segment's
    bearing, so target_known is True wherever heading_known holds.
    """
    out = derive_columns(straight_cruise_flight)
    in_turn = out["fdm_in_turn"].to_numpy()
    target_known = out["fdm_heading_target_known"].to_numpy()
    heading_known = out["fdm_heading_known"].to_numpy()
    # Fixture must actually exercise a detected turn for AC9 to be meaningful.
    assert in_turn.any(), "fixture should produce at least one in-turn sample"
    # AC9: at least one sample is both in-turn and has a known heading target.
    assert (in_turn & heading_known & target_known).any()
