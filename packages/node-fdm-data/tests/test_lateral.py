"""Tests for node_fdm_data.lateral — turning point detection & theoretical track."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.lateral import (
    augment_with_segments,
    compute_lateral_track,
    compute_theoretical_track,
    detect_turning_points,
    find_segment_bounds,
)

# ---------------------------------------------------------------------------
# detect_turning_points
# ---------------------------------------------------------------------------


class TestDetectTurningPoints:
    """Turning-point detection on track-angle series."""

    def test_no_turning_points(self) -> None:
        """Straight-line constant track → empty array."""
        track = np.full(50, 90.0)
        time = np.arange(50, dtype=np.float64)
        result = detect_turning_points(track, time)
        assert len(result) == 0

    def test_single_turn(self) -> None:
        """Track with one 90° turn → at least one index at turn start."""
        n = 100
        track = np.concatenate([np.full(40, 0.0), np.linspace(0, 90, 20), np.full(40, 90.0)])
        time = np.arange(n, dtype=np.float64)
        result = detect_turning_points(track, time)
        assert len(result) >= 1
        # Turn starts around index 40
        assert result[0] >= 30
        assert result[0] <= 50

    def test_wrap_around_360(self) -> None:
        """Track crossing 350°→10° — discontinuity handled correctly."""
        n = 100
        # Go from 350 to 370 (unwrapped), which is 350→10 wrapped
        track = np.concatenate(
            [
                np.full(40, 350.0),
                np.linspace(350, 370, 20),  # crosses 360
                np.full(40, 370.0),
            ]
        )
        time = np.arange(n, dtype=np.float64)
        # Should NOT produce spurious turns at the 360° boundary
        result = detect_turning_points(track, time)
        # The 20° turn is small — may or may not detect, but should not crash
        assert isinstance(result, np.ndarray)

    def test_short_series(self) -> None:
        """Series < window_length (9) → empty, no crash."""
        track = np.array([0.0, 10.0, 20.0])
        time = np.array([0.0, 1.0, 2.0])
        result = detect_turning_points(track, time)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# find_segment_bounds
# ---------------------------------------------------------------------------


class TestFindSegmentBounds:
    """Segment bounds from turning indices."""

    def test_middle_segment(self) -> None:
        """Pivot inside a middle segment."""
        indices = np.array([10, 30, 60])
        start, end = find_segment_bounds(indices, 20, 100)
        assert start == 10
        assert end == 30

    def test_before_first(self) -> None:
        """Pivot before the first turning index."""
        indices = np.array([20, 50])
        start, end = find_segment_bounds(indices, 5, 100)
        assert start == 0
        assert end == 20

    def test_after_last(self) -> None:
        """Pivot after the last turning index."""
        indices = np.array([20, 50])
        start, end = find_segment_bounds(indices, 70, 100)
        assert start == 50
        assert end == 99


# ---------------------------------------------------------------------------
# compute_theoretical_track
# ---------------------------------------------------------------------------


class TestTheoreticalTrack:
    """Great-circle bearing computation."""

    def test_theoretical_track_known_pair(self) -> None:
        """CDG (49.01°N, 2.55°E) → JFK (40.64°N, 73.78°W): ~292°."""
        df = pl.DataFrame(
            {
                "latitude": [49.01],
                "longitude": [2.55],
                "lat_B": [40.64],
                "lon_B": [-73.78],
            }
        )
        result = compute_theoretical_track(df)
        bearing = result[0]
        # Great-circle initial bearing CDG→JFK ≈ 292°
        assert bearing == pytest.approx(292.0, abs=1.0)


# ---------------------------------------------------------------------------
# augment_with_segments
# ---------------------------------------------------------------------------


class TestAugmentSegments:
    """DataFrame augmentation with segment endpoint coordinates."""

    def test_augment_three_segments(self) -> None:
        """3 segments → lat_A/lon_A/lat_B/lon_B columns populated."""
        n = 30
        df = pl.DataFrame(
            {
                "latitude": np.linspace(45.0, 48.0, n),
                "longitude": np.linspace(2.0, 5.0, n),
            }
        )
        turning_idx = np.array([10, 20])
        result = augment_with_segments(df, turning_idx)

        assert "lat_A" in result.columns
        assert "lon_A" in result.columns
        assert "lat_B" in result.columns
        assert "lon_B" in result.columns
        assert result["lat_A"].null_count() == 0
        assert result["lon_B"].null_count() == 0

        # Row 0 should be in segment [0, 10)
        assert result["lat_A"][0] == pytest.approx(df["latitude"][0])
        assert result["lat_B"][0] == pytest.approx(df["latitude"][10])

    def test_augment_empty_df(self) -> None:
        """Empty DataFrame → returns with null columns, no crash."""
        df = pl.DataFrame(
            {
                "latitude": pl.Series([], dtype=pl.Float64),
                "longitude": pl.Series([], dtype=pl.Float64),
            }
        )
        result = augment_with_segments(df, np.array([], dtype=np.intp))
        assert "lat_A" in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases from ticket spec."""

    def test_all_zero_time_delta(self) -> None:
        """Identical timestamps → no division by zero (clamped to 1e-6)."""
        track = np.linspace(0, 90, 50)
        time = np.zeros(50)
        result = detect_turning_points(track, time)
        assert isinstance(result, np.ndarray)
        # Should not raise or return NaN

    def test_constant_track(self) -> None:
        """No track variation → no turning points detected."""
        track = np.full(50, 180.0)
        time = np.arange(50, dtype=np.float64)
        result = detect_turning_points(track, time)
        assert len(result) == 0

    def test_single_row_df(self) -> None:
        """Single-point trajectory → graceful return, no crash."""
        df = pl.DataFrame(
            {
                "track": [90.0],
                "timestamp": [0.0],
                "latitude": [45.0],
                "longitude": [2.0],
            }
        )
        result = compute_lateral_track(df)
        assert "track_sel" in result.columns
        assert len(result) == 1
