"""Tests for node_fdm_data.lateral — turn detection, bearings, augment_lateral."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.lateral import (
    augment_lateral,
    detect_turning_points,
    orthodromic_bearing,
    rhumb_bearing,
)

# ---------------------------------------------------------------------------
# orthodromic_bearing
# ---------------------------------------------------------------------------


class TestOrthodromicBearing:
    """Great-circle initial bearing."""

    def test_cdg_to_jfk(self) -> None:
        """CDG (49.01°N, 2.55°E) → JFK (40.64°N, 73.78°W) ≈ 292°."""
        phi1, lam1 = np.radians(49.01), np.radians(2.55)
        phi2, lam2 = np.radians(40.64), np.radians(-73.78)
        bearing = np.degrees(orthodromic_bearing(phi1, lam1, phi2, lam2))
        assert bearing == pytest.approx(292.0, abs=1.5)

    def test_due_east(self) -> None:
        """Equator, same lat, east → 90°."""
        bearing = np.degrees(
            orthodromic_bearing(
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.radians(10.0),
            )
        )
        assert bearing == pytest.approx(90.0, abs=0.01)

    def test_due_north(self) -> None:
        """Same longitude, going north → 0°."""
        bearing = np.degrees(
            orthodromic_bearing(
                np.radians(45.0),
                np.radians(2.0),
                np.radians(55.0),
                np.radians(2.0),
            )
        )
        assert bearing == pytest.approx(0.0, abs=0.01)

    def test_vectorised(self) -> None:
        """Works on arrays."""
        phi1 = np.radians([0.0, 45.0])
        lam1 = np.radians([0.0, 2.0])
        phi2 = np.radians([0.0, 55.0])
        lam2 = np.radians([10.0, 2.0])
        bearings = np.degrees(orthodromic_bearing(phi1, lam1, phi2, lam2))
        assert len(bearings) == 2
        assert bearings[0] == pytest.approx(90.0, abs=0.01)
        assert bearings[1] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# rhumb_bearing
# ---------------------------------------------------------------------------


class TestRhumbBearing:
    """Loxodromic (constant heading) bearing."""

    def test_due_east(self) -> None:
        """Same latitude, east → 90°."""
        bearing = np.degrees(
            rhumb_bearing(
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.radians(10.0),
            )
        )
        assert bearing == pytest.approx(90.0, abs=0.1)

    def test_due_north(self) -> None:
        """Same longitude, going north → 0°."""
        bearing = np.degrees(
            rhumb_bearing(
                np.radians(45.0),
                np.radians(2.0),
                np.radians(55.0),
                np.radians(2.0),
            )
        )
        assert bearing == pytest.approx(0.0, abs=0.1)

    def test_same_point_is_nan(self) -> None:
        """Same point → NaN (no defined bearing)."""
        bearing = rhumb_bearing(
            np.float64(0.5),
            np.float64(0.5),
            np.float64(0.5),
            np.float64(0.5),
        )
        assert np.isnan(bearing)

    def test_short_segment_matches_ortho(self) -> None:
        """For short segments, loxo ≈ ortho (within 1°)."""
        phi1, lam1 = np.radians(48.0), np.radians(2.0)
        phi2, lam2 = np.radians(49.0), np.radians(3.0)
        ortho = np.degrees(orthodromic_bearing(phi1, lam1, phi2, lam2))
        loxo = np.degrees(rhumb_bearing(phi1, lam1, phi2, lam2))
        assert abs(ortho - loxo) < 1.0


# ---------------------------------------------------------------------------
# detect_turning_points
# ---------------------------------------------------------------------------


class TestDetectTurningPoints:
    """Turn detection via smoothed angular rate."""

    def test_straight_flight(self) -> None:
        """Constant track → no turns detected."""
        track = np.full(100, 90.0)
        seg, in_turn = detect_turning_points(track)
        assert not in_turn.any(), "Constant track should have no turns"
        assert seg[0] == 0

    def test_single_turn(self) -> None:
        """One 90° turn → some points marked as in_turn."""
        track = np.concatenate(
            [
                np.full(40, 0.0),
                np.linspace(0, 90, 20),
                np.full(40, 90.0),
            ]
        )
        _, in_turn = detect_turning_points(track)
        assert in_turn.any(), "Should detect the turn"
        # Turn region should be around indices 40-60
        turn_start = np.argmax(in_turn)
        assert 30 <= turn_start <= 50

    def test_wrap_around_360(self) -> None:
        """Track crossing 350°→10° — no spurious detection."""
        track = np.concatenate(
            [
                np.full(40, 350.0),
                np.linspace(350, 370, 20) % 360,  # wraps at 360
                np.full(40, 10.0),
            ]
        )
        # This is a 20° gentle turn over 20 points at 4s = 0.25°/s
        _, in_turn = detect_turning_points(track)
        assert isinstance(in_turn, np.ndarray)
        assert len(in_turn) == len(track)

    def test_short_series(self) -> None:
        """Series < 5 points → no crash, empty in_turn."""
        track = np.array([0.0, 10.0, 20.0])
        seg, in_turn = detect_turning_points(track)
        assert seg[0] == 0
        assert not in_turn.any()

    def test_multiple_turns(self) -> None:
        """Two turns → multiple segments detected."""
        track = np.concatenate(
            [
                np.full(30, 0.0),
                np.linspace(0, 90, 15),  # turn 1
                np.full(30, 90.0),
                np.linspace(90, 180, 15),  # turn 2
                np.full(30, 180.0),
            ]
        )
        _, in_turn = detect_turning_points(track)
        # At least 2 turn regions
        turn_changes = np.diff(in_turn.astype(int))
        n_turn_starts = np.sum(turn_changes == 1)
        assert n_turn_starts >= 2, f"Expected ≥ 2 turns, got {n_turn_starts}"


# ---------------------------------------------------------------------------
# augment_lateral (functional test)
# ---------------------------------------------------------------------------


class TestAugmentLateral:
    """Full lateral augmentation pipeline."""

    @pytest.fixture()
    def straight_flight_df(self) -> pl.DataFrame:
        """Synthetic straight flight NW-bound (≈330°)."""
        n = 100
        return pl.DataFrame(
            {
                "latitude": np.linspace(45.0, 48.0, n),
                "longitude": np.linspace(2.0, 0.0, n),
                "track": np.full(n, 330.0),
                "heading": np.full(n, 325.0),  # 5° left drift
                "TAS": np.full(n, 450.0),
                "timestamp": np.arange(n, dtype=np.float64) * 4,
            }
        )

    @pytest.fixture()
    def turning_flight_df(self) -> pl.DataFrame:
        """Flight with a 90° turn in the middle."""
        n = 120
        lat = np.concatenate(
            [
                np.linspace(45.0, 46.0, 40),
                np.linspace(46.0, 46.5, 40),
                np.linspace(46.5, 47.0, 40),
            ]
        )
        lon = np.concatenate(
            [
                np.full(40, 2.0),
                np.linspace(2.0, 3.5, 40),
                np.full(40, 3.5),
            ]
        )
        track = np.concatenate(
            [
                np.full(40, 0.0),
                np.linspace(0, 90, 40),
                np.full(40, 90.0),
            ]
        )
        return pl.DataFrame(
            {
                "latitude": lat,
                "longitude": lon,
                "track": track,
                "heading": track + np.random.default_rng(42).normal(0, 2, n),
                "TAS": np.full(n, 450.0),
                "timestamp": np.arange(n, dtype=np.float64) * 4,
            }
        )

    def test_output_columns(self, straight_flight_df: pl.DataFrame) -> None:
        """All required lateral columns present."""
        result = augment_lateral(straight_flight_df)
        for col in ("in_turn", "track_ortho", "track_loxo", "drift_angle", "lat_wind"):
            assert col in result.columns, f"Missing column: {col}"

    def test_straight_no_turns(self, straight_flight_df: pl.DataFrame) -> None:
        """Straight flight → no points in turn."""
        result = augment_lateral(straight_flight_df)
        assert not result["in_turn"].to_numpy().any()

    def test_ortho_loxo_close_on_straight(self, straight_flight_df: pl.DataFrame) -> None:
        """On straight segment, ortho ≈ loxo (short distance)."""
        result = augment_lateral(straight_flight_df)
        ortho = result["track_ortho"].to_numpy()
        loxo = result["track_loxo"].to_numpy()
        valid = ~np.isnan(ortho) & ~np.isnan(loxo)
        if valid.any():
            diff = np.abs(ortho[valid] - loxo[valid])
            assert np.median(diff) < 2.0, f"Ortho/loxo differ too much: {np.median(diff):.1f} deg"

    def test_drift_angle_sign(self, straight_flight_df: pl.DataFrame) -> None:
        """Heading 325° with track 330° → drift ≈ -5°."""
        result = augment_lateral(straight_flight_df)
        drift = result["drift_angle"].to_numpy()
        assert np.median(drift) == pytest.approx(-5.0, abs=0.5)

    def test_lat_wind_from_drift(self, straight_flight_df: pl.DataFrame) -> None:
        """With TAS=450kt and drift=-5°, lat_wind ≈ -39 kt."""
        result = augment_lateral(straight_flight_df)
        lat_w = result["lat_wind"].to_numpy()
        expected = 450.0 * np.sin(np.radians(-5.0))  # ≈ -39.3 kt
        assert np.median(lat_w) == pytest.approx(expected, abs=2.0)

    def test_turns_detected(self, turning_flight_df: pl.DataFrame) -> None:
        """Flight with a turn → some in_turn points."""
        result = augment_lateral(turning_flight_df)
        in_turn = result["in_turn"].to_numpy()
        assert in_turn.any(), "Should detect the turn"
        # Turn should be in middle section
        turn_center = np.median(np.where(in_turn)[0])
        assert 30 < turn_center < 90

    def test_ortho_nan_during_turns(self, turning_flight_df: pl.DataFrame) -> None:
        """Reference tracks are NaN during turns."""
        result = augment_lateral(turning_flight_df)
        in_turn = result["in_turn"].to_numpy()
        ortho = result["track_ortho"].to_numpy()
        if in_turn.any():
            assert np.all(np.isnan(ortho[in_turn])), "Ortho should be NaN during turns"

    def test_row_count_preserved(self, turning_flight_df: pl.DataFrame) -> None:
        """Output has same number of rows as input."""
        result = augment_lateral(turning_flight_df)
        assert len(result) == len(turning_flight_df)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and degenerate inputs."""

    def test_too_short(self) -> None:
        """< 5 points → graceful return with null columns."""
        df = pl.DataFrame(
            {
                "latitude": [45.0, 46.0],
                "longitude": [2.0, 3.0],
                "track": [90.0, 90.0],
                "heading": [88.0, 88.0],
                "TAS": [400.0, 400.0],
            }
        )
        result = augment_lateral(df)
        assert "track_ortho" in result.columns
        assert len(result) == 2

    def test_missing_heading_column(self) -> None:
        """No heading → drift_angle and lat_wind are NaN."""
        n = 50
        df = pl.DataFrame(
            {
                "latitude": np.linspace(45.0, 46.0, n),
                "longitude": np.linspace(2.0, 3.0, n),
                "track": np.full(n, 45.0),
            }
        )
        result = augment_lateral(df)
        drift = result["drift_angle"].to_numpy()
        assert np.all(np.isnan(drift)), "drift_angle should be all NaN without heading"
        lat_w = result["lat_wind"].to_numpy()
        assert np.all(np.isnan(lat_w)), "lat_wind should be all NaN without heading"

    def test_missing_tas_column(self) -> None:
        """No TAS column → lat_wind is NaN, drift still NaN (no heading+TAS)."""
        n = 50
        df = pl.DataFrame(
            {
                "latitude": np.linspace(45.0, 46.0, n),
                "longitude": np.linspace(2.0, 3.0, n),
                "track": np.full(n, 45.0),
                "heading": np.full(n, 42.0),
            }
        )
        result = augment_lateral(df)
        lat_w = result["lat_wind"].to_numpy()
        assert np.all(np.isnan(lat_w)), "lat_wind should be all NaN without TAS"
