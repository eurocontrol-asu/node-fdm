"""Tests for node_fdm_data.lateral.

Covers the public API (``orthodromic_bearing``, ``detect_turning_starts``,
``augment_lateral``) plus the segment-assignment primitives
``segment_bounds`` and ``build_in_turn_mask`` from
``node_fdm_data.lateral_segments``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.lateral import (
    augment_lateral,
    detect_turning_starts,
    orthodromic_bearing,
)
from node_fdm_data.lateral_segments import build_in_turn_mask, segment_bounds

# ---------------------------------------------------------------------------
# orthodromic_bearing
# ---------------------------------------------------------------------------


class TestOrthodromicBearing:
    """Great-circle initial bearing (radians in / radians out, [0, 2π))."""

    @pytest.mark.parametrize(
        ("phi1", "lam1", "phi2", "lam2", "expected_deg", "tol"),
        [
            pytest.param(
                np.radians(49.01),
                np.radians(2.55),
                np.radians(40.64),
                np.radians(-73.78),
                292.0,
                1.5,
                id="cdg_to_jfk",
            ),
            pytest.param(
                np.float64(0.0),
                np.float64(0.0),
                np.float64(0.0),
                np.radians(10.0),
                90.0,
                0.01,
                id="due_east",
            ),
            pytest.param(
                np.float64(0.0),
                np.float64(0.0),
                np.radians(10.0),
                np.float64(0.0),
                0.0,
                0.01,
                id="due_north",
            ),
        ],
    )
    def test_known_bearings(  # noqa: PLR0913
        self,
        phi1: np.float64,
        lam1: np.float64,
        phi2: np.float64,
        lam2: np.float64,
        expected_deg: float,
        tol: float,
    ) -> None:
        """Known great-circle bearings match reference values."""
        bearing = np.degrees(orthodromic_bearing(phi1, lam1, phi2, lam2))
        assert bearing == pytest.approx(expected_deg, abs=tol)

    def test_range_unsigned(self) -> None:
        """Result always in [0, 2π)."""
        # Going west from prime meridian
        bearing = orthodromic_bearing(
            np.float64(0.0),
            np.float64(0.0),
            np.float64(0.0),
            np.radians(-10.0),
        )
        assert 0.0 <= float(bearing) < 2 * np.pi

    def test_vectorised(self) -> None:
        """Accepts arrays and returns elementwise bearings."""
        phi1 = np.radians(np.array([0.0, 0.0, 45.0]))
        lam1 = np.radians(np.array([0.0, 0.0, 0.0]))
        phi2 = np.radians(np.array([0.0, 10.0, 45.0]))
        lam2 = np.radians(np.array([10.0, 0.0, 10.0]))
        bearing = np.degrees(orthodromic_bearing(phi1, lam1, phi2, lam2))
        assert bearing.shape == (3,)
        assert bearing[0] == pytest.approx(90.0, abs=0.01)
        assert bearing[1] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# detect_turning_starts
# ---------------------------------------------------------------------------


class TestDetectTurningStarts:
    """Turn-start detection via Savgol + find_peaks + backtrack."""

    @pytest.mark.parametrize(
        "track",
        [
            pytest.param(np.full(100, 90.0), id="straight_flight"),
            pytest.param(
                np.where(np.arange(100) == 50, np.nan, 90.0),
                id="handles_nan_input",
            ),
            pytest.param(np.full(100, np.nan), id="all_nan"),
        ],
    )
    def test_no_turns_detected(self, track: np.ndarray) -> None:
        """Inputs without genuine turns yield empty starts."""
        starts = detect_turning_starts(track)
        assert starts.size == 0

    def test_too_short(self) -> None:
        """Series shorter than the Savgol window → empty result."""
        track = np.array([0.0, 10.0, 20.0])
        starts = detect_turning_starts(track)
        assert starts.size == 0

    def test_single_turn(self) -> None:
        """One 90° turn → one start, before the turn region."""
        track = np.concatenate(
            [
                np.full(40, 0.0),
                np.linspace(0, 90, 20),
                np.full(40, 90.0),
            ]
        )
        starts = detect_turning_starts(track)
        assert starts.size >= 1
        # The turn region is samples 40..60; start should land before its peak.
        assert 30 <= int(starts[0]) <= 50

    def test_multiple_turns(self) -> None:
        """Two 90° turns → at least two distinct starts, in order."""
        track = np.concatenate(
            [
                np.full(30, 0.0),
                np.linspace(0, 90, 15),
                np.full(30, 90.0),
                np.linspace(90, 180, 15),
                np.full(30, 180.0),
            ]
        )
        starts = detect_turning_starts(track)
        assert starts.size >= 2
        # Sorted unique — first turn before second.
        assert np.all(np.diff(starts) > 0)

    def test_wrap_around_360(self) -> None:
        """Track crossing 350°→10° → the 0/360 jump must not register."""
        track = np.concatenate(
            [
                np.full(40, 350.0),
                np.linspace(350, 370, 20) % 360,  # wraps at 360
                np.full(40, 10.0),
            ]
        )
        # 20° rotation over 80 s = 0.25°/s — above threshold but smooth.
        starts = detect_turning_starts(track)
        # Should detect the gentle turn, not double-trigger on the wrap.
        assert starts.size <= 2

    def test_returns_intp_array(self) -> None:
        """Return dtype is np.intp (indexable)."""
        track = np.full(100, 0.0)
        starts = detect_turning_starts(track)
        assert starts.dtype == np.intp


# ---------------------------------------------------------------------------
# segment_bounds
# ---------------------------------------------------------------------------


class TestSegmentBounds:
    """Per-sample (A, B) enclosing-segment indices."""

    def test_no_turns(self) -> None:
        """Empty turning_starts → A=0, B=n-1 for every sample."""
        a, b = segment_bounds(np.empty(0, dtype=np.intp), 10)
        assert np.all(a == 0)
        assert np.all(b == 9)

    def test_one_turn_in_middle(self) -> None:
        """One turn at i=5 in n=10 → samples 0..4 have B=5, samples 5..9
        have A=5 and B=n-1=9."""
        a, b = segment_bounds(np.array([5], dtype=np.intp), 10)
        assert a[0] == 0 and b[0] == 5
        assert a[4] == 0 and b[4] == 5
        assert a[5] == 5 and b[5] == 9
        assert a[9] == 5 and b[9] == 9

    def test_multiple_turns(self) -> None:
        """Three turns → samples between turns enclosed by adjacent starts."""
        starts = np.array([3, 6, 9], dtype=np.intp)
        a, b = segment_bounds(starts, 12)
        # Sample 4 sits in [3, 6)
        assert a[4] == 3 and b[4] == 6
        # Sample 7 sits in [6, 9)
        assert a[7] == 6 and b[7] == 9
        # Sample 10 is past the last turn → A=9, B=n-1=11
        assert a[10] == 9 and b[10] == 11
        # Sample 0 is before the first turn → A=0, B=3
        assert a[0] == 0 and b[0] == 3


# ---------------------------------------------------------------------------
# build_in_turn_mask
# ---------------------------------------------------------------------------


class TestBuildInTurnMask:
    """Mask of samples without a valid enclosing straight segment."""

    def test_no_turns_all_true(self) -> None:
        """No turns → every sample is "outside any segment"."""
        a = np.zeros(10, dtype=np.intp)
        b = np.full(10, 9, dtype=np.intp)
        mask = build_in_turn_mask(np.empty(0, dtype=np.intp), a, b, 10)
        assert mask.all()

    def test_head_and_tail_masked(self) -> None:
        """Samples before the first turn and at/after the last are masked."""
        starts = np.array([3, 7], dtype=np.intp)
        a, b = segment_bounds(starts, 10)
        mask = build_in_turn_mask(starts, a, b, 10)
        # Head: 0..2 before first turn
        assert mask[0] and mask[1] and mask[2]
        # Body: 3..6 inside [3, 7) — straight, not masked
        assert not mask[3] and not mask[6]
        # Tail: 7..9 at/after last turn
        assert mask[7] and mask[9]

    def test_degenerate_zero_length_segment(self) -> None:
        """A == B (degenerate) → masked, even mid-flight."""
        a = np.array([0, 5, 5, 5, 9], dtype=np.intp)
        b = np.array([5, 5, 9, 9, 9], dtype=np.intp)
        mask = build_in_turn_mask(np.array([5, 9], dtype=np.intp), a, b, 5)
        # Sample 1 has A == B == 5
        assert mask[1]


# ---------------------------------------------------------------------------
# augment_lateral (public API)
# ---------------------------------------------------------------------------


def _make_straight_flight(n: int = 100) -> pl.DataFrame:
    """Synthetic straight north-east flight — no turn."""
    return pl.DataFrame(
        {
            "latitude": np.linspace(45.0, 48.0, n),
            "longitude": np.linspace(2.0, 5.0, n),
            "track": np.full(n, 45.0),
        }
    )


def _make_turning_flight() -> pl.DataFrame:
    """Synthetic flight with two 90° turns — produces a straight middle leg
    enclosed by two detected turn-starts (so some samples are NOT in_turn)."""
    lat = np.concatenate(
        [
            np.linspace(45.0, 46.0, 40),  # leg 1: north
            np.linspace(46.0, 46.0, 40),  # leg 2: east (lat constant)
            np.linspace(46.0, 47.0, 40),  # leg 3: north again
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
            np.linspace(0, 90, 10),
            np.full(20, 90.0),
            np.linspace(90, 0, 10),
            np.full(40, 0.0),
        ]
    )
    return pl.DataFrame({"latitude": lat, "longitude": lon, "track": track})


class TestAugmentLateral:
    """Full lateral-augmentation pipeline emitting the 3 fdm_* columns."""

    def test_output_columns_exact(self) -> None:
        """Adds exactly fdm_in_turn, fdm_track_ortho_deg, fdm_track_sel_known."""
        df = _make_straight_flight()
        result = augment_lateral(df)
        added = set(result.columns) - set(df.columns)
        assert added == {"fdm_in_turn", "fdm_track_ortho_deg", "fdm_track_sel_known"}

    def test_output_dtypes(self) -> None:
        """fdm_in_turn and fdm_track_sel_known are bool; fdm_track_ortho_deg is float."""
        result = augment_lateral(_make_straight_flight())
        assert result["fdm_in_turn"].dtype == pl.Boolean
        assert result["fdm_track_sel_known"].dtype == pl.Boolean
        assert result["fdm_track_ortho_deg"].dtype == pl.Float64

    def test_row_count_preserved(self) -> None:
        """Output has same number of rows as input."""
        df = _make_turning_flight()
        result = augment_lateral(df)
        assert len(result) == len(df)

    def test_straight_flight_all_in_turn(self) -> None:
        """Straight flight (no detected turn) → every sample is "no enclosing segment"."""
        result = augment_lateral(_make_straight_flight())
        # No turns detected → entire flight has no enclosing [A, B) segment.
        assert result["fdm_in_turn"].to_numpy().all()
        # Therefore no valid lateral target.
        assert not result["fdm_track_sel_known"].to_numpy().any()

    def test_turn_detected_in_middle(self) -> None:
        """Flight with a turn → some samples are NOT in_turn (the straight legs)."""
        result = augment_lateral(_make_turning_flight())
        in_turn = result["fdm_in_turn"].to_numpy()
        # At least the head (before first turn-start) is masked.
        assert in_turn[0]
        # And at least one sample mid-flight should be inside a straight segment.
        assert (~in_turn).any()

    def test_ortho_nan_iff_in_turn(self) -> None:
        """fdm_track_ortho_deg is NaN exactly where fdm_in_turn is True."""
        result = augment_lateral(_make_turning_flight())
        in_turn = result["fdm_in_turn"].to_numpy()
        ortho = result["fdm_track_ortho_deg"].to_numpy()
        assert np.all(np.isnan(ortho[in_turn]))
        # Outside in_turn, ortho is finite.
        assert np.all(np.isfinite(ortho[~in_turn]))

    def test_known_flag_consistency(self) -> None:
        """fdm_track_sel_known == ~fdm_in_turn & isfinite(ortho)."""
        result = augment_lateral(_make_turning_flight())
        in_turn = result["fdm_in_turn"].to_numpy()
        ortho = result["fdm_track_ortho_deg"].to_numpy()
        known = result["fdm_track_sel_known"].to_numpy()
        expected = (~in_turn) & np.isfinite(ortho)
        assert np.array_equal(known, expected)

    def test_ortho_in_degrees_range(self) -> None:
        """Non-NaN ortho values are in [0, 360)."""
        result = augment_lateral(_make_turning_flight())
        ortho = result["fdm_track_ortho_deg"].to_numpy()
        finite = ortho[np.isfinite(ortho)]
        assert finite.size > 0
        assert np.all((finite >= 0.0) & (finite < 360.0))

    def test_too_short_returns_safe_defaults(self) -> None:
        """Series shorter than Savgol window → all in_turn, no known target, ortho null."""
        df = pl.DataFrame(
            {
                "latitude": [45.0, 46.0],
                "longitude": [2.0, 3.0],
                "track": [90.0, 90.0],
            }
        )
        result = augment_lateral(df)
        assert len(result) == 2
        assert result["fdm_in_turn"].to_numpy().all()
        assert not result["fdm_track_sel_known"].to_numpy().any()
        # Ortho column exists and is fully null.
        assert result["fdm_track_ortho_deg"].null_count() == 2
