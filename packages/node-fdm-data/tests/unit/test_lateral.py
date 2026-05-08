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
    detect_turn_intervals,
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
                # Single NaN sample: V3 forward-fills before bilateral so the
                # gap collapses to ~0 rotation rate -> still no turn detected.
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
        # V3 bilateral smoothing shifts the threshold-crossing slightly so the
        # tolerated window is widened from 30..50 to 25..55.
        assert 25 <= int(starts[0]) <= 55

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
# detect_turn_intervals (V3: bilateral + symmetric threshold)
# ---------------------------------------------------------------------------


def _make_turning_track() -> np.ndarray:
    """Synthetic single 90 deg turn: 40 flat, 20 ramp, 40 flat."""
    return np.concatenate(
        [
            np.full(40, 0.0),
            np.linspace(0.0, 90.0, 20),
            np.full(40, 90.0),
        ]
    )


class TestDetectTurnIntervals:
    """V3 bilateral + symmetric-threshold turn-interval detection."""

    def test_intervals_returns_5_arrays(self) -> None:
        """AC1: returns a 4-tuple of ndarrays."""
        track = np.full(100, 90.0)
        result = detect_turn_intervals(track)
        assert isinstance(result, tuple)
        assert len(result) == 4
        for arr in result:
            assert isinstance(arr, np.ndarray)

    def test_intervals_dtype_intp(self) -> None:
        """AC4: starts and ends arrays are np.intp."""
        track = _make_turning_track()
        starts, ends, _, _ = detect_turn_intervals(track)
        assert starts.dtype == np.intp
        assert ends.dtype == np.intp

    def test_intervals_starts_le_ends(self) -> None:
        """AC4: every start <= matching end."""
        track = _make_turning_track()
        starts, ends, _, _ = detect_turn_intervals(track)
        assert np.all(starts <= ends)

    def test_intervals_sorted_unique(self) -> None:
        """AC4: starts strictly increasing (sorted, no duplicates)."""
        # Two well-separated turns guarantee >1 interval to exercise diff.
        track = np.concatenate(
            [
                np.full(40, 0.0),
                np.linspace(0.0, 90.0, 20),
                np.full(60, 90.0),
                np.linspace(90.0, 180.0, 20),
                np.full(40, 180.0),
            ]
        )
        starts, _, _, _ = detect_turn_intervals(track)
        if starts.size > 1:
            assert np.all(np.diff(starts) > 0)

    def test_intervals_single_turn_brackets_peak(self) -> None:
        """AC2: a single 90 deg turn yields one [start, end] enclosing the ramp."""
        track = np.concatenate(
            [
                np.full(40, 0.0),
                np.linspace(0.0, 90.0, 20),
                np.full(40, 90.0),
            ]
        )
        starts, ends, _, _ = detect_turn_intervals(track)
        assert starts.size == 1
        assert ends.size == 1
        assert 30 <= int(starts[0]) < 50 <= int(ends[0]) <= 70

    def test_intervals_two_distinct_turns(self) -> None:
        """AC2/AC3: two well-separated turns are not merged."""
        track = np.concatenate(
            [
                np.full(40, 0.0),
                np.linspace(0.0, 90.0, 20),
                np.full(60, 90.0),
                np.linspace(90.0, 180.0, 20),
                np.full(40, 180.0),
            ]
        )
        starts, ends, _, _ = detect_turn_intervals(track)
        assert starts.size == 2
        assert ends[0] < starts[1]

    def test_intervals_overlapping_turns_merged(self) -> None:
        """AC3: nearby turns whose walk-forward/walk-back overlap collapse to one."""
        track = np.concatenate(
            [
                np.full(20, 0.0),
                np.linspace(0.0, 45.0, 15),
                np.full(5, 45.0),
                np.linspace(45.0, 90.0, 15),
                np.full(20, 90.0),
            ]
        )
        starts, ends, _, _ = detect_turn_intervals(track)
        assert starts.size == 1
        assert ends.size == 1

    def test_intervals_too_short(self) -> None:
        """AC5: signal shorter than Savgol window -> empty starts/ends, zero rates."""
        track = np.array([0.0, 10.0, 20.0])
        starts, ends, abs_rate, abs_rate_bilat = detect_turn_intervals(track)
        assert starts.size == 0
        assert ends.size == 0
        assert abs_rate.shape == (3,)
        assert abs_rate_bilat.shape == (3,)
        assert np.all(abs_rate == 0.0)
        assert np.all(abs_rate_bilat == 0.0)

    def test_intervals_all_nan(self) -> None:
        """AC5: fully NaN signal -> empty starts/ends, smoothed array of length n."""
        track = np.full(50, np.nan)
        starts, ends, _, abs_rate_bilat = detect_turn_intervals(track)
        assert starts.size == 0
        assert ends.size == 0
        assert abs_rate_bilat.shape == (50,)

    def test_intervals_bilateral_flattens_noise(self) -> None:
        """AC2: 2 passes of bilateral suppress sub-threshold IID noise."""
        rng = np.random.default_rng(0)
        track = 90.0 + rng.normal(0.0, 0.001, 200).cumsum()
        _, _, _, abs_rate_bilat = detect_turn_intervals(track)
        assert abs_rate_bilat.max() < 0.05

    def test_intervals_threshold_param_respected(self) -> None:
        """AC2: rate_threshold gates both find_peaks and walk-back/forward."""
        # 90 deg over 281 samples * dt=4s -> peak rate ~= 0.08 deg/s.
        track = np.concatenate(
            [
                np.full(50, 0.0),
                np.linspace(0.0, 90.0, 281),
                np.full(50, 90.0),
            ]
        )
        starts_low, _, _, _ = detect_turn_intervals(track, rate_threshold=0.05)
        starts_high, _, _, _ = detect_turn_intervals(track, rate_threshold=0.10)
        assert starts_low.size == 1
        assert starts_high.size == 0

    def test_legacy_detect_turning_starts_shim(self) -> None:
        """AC7: legacy shim returns the same starts as the new function."""
        track = _make_turning_track()
        with pytest.warns(DeprecationWarning):
            legacy = detect_turning_starts(track)
        new_starts = detect_turn_intervals(track)[0]
        assert np.array_equal(legacy, new_starts)

    def test_legacy_emits_deprecation_warning(self) -> None:
        """AC7: calling legacy entry-point raises DeprecationWarning."""
        track = _make_turning_track()
        with pytest.warns(DeprecationWarning):
            detect_turning_starts(track)

    def test_lateral_all_exports_new_symbol(self) -> None:
        """AC8: new symbol is part of the module's __all__."""
        import node_fdm_data.lateral as m

        assert "detect_turn_intervals" in m.__all__


# ---------------------------------------------------------------------------
# segment_bounds — V3 sémantique: takes (starts, ends, n) intervals
# ---------------------------------------------------------------------------


def test_segment_bounds_intervals_basic() -> None:
    """AC1: head/middle/tail straight-segment bounds derived from intervals."""
    starts = np.array([3, 9], dtype=np.intp)
    ends = np.array([5, 10], dtype=np.intp)
    a, b = segment_bounds(starts, ends, 15)
    # Head (before first start): A=0, B=starts[0]=3.
    assert (int(a[0]), int(b[0])) == (0, 3)
    # Straight gap between turn1[3..5] and turn2[9..10]: A=ends[0]+1=6, B=starts[1]=9.
    assert (int(a[6]), int(b[6])) == (6, 9)
    # Tail after last end: A=ends[-1]+1=11, B=n-1=14.
    assert (int(a[12]), int(b[12])) == (11, 14)


def test_segment_bounds_inside_interval() -> None:
    """AC1: a sample inside a turn interval still receives a valid B (next start)."""
    starts = np.array([3, 9], dtype=np.intp)
    ends = np.array([5, 10], dtype=np.intp)
    _, b = segment_bounds(starts, ends, 15)
    # Sample 4 lies inside [3, 5]; the next straight segment starts at idx 9.
    assert int(b[4]) == 9


def test_segment_bounds_empty_intervals() -> None:
    """AC1: no intervals → A=0, B=n-1 everywhere."""
    a, b = segment_bounds(np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), 10)
    assert np.all(a == 0)
    assert np.all(b == 9)


# ---------------------------------------------------------------------------
# build_in_turn_mask — V3 sémantique: True iff inside [s_k, e_k]
# ---------------------------------------------------------------------------


def test_build_in_turn_mask_intervals() -> None:
    """AC2: mask True exactly on [s_k, e_k]; head/tail straight = False."""
    starts = np.array([3, 9], dtype=np.intp)
    ends = np.array([5, 10], dtype=np.intp)
    mask = build_in_turn_mask(starts, ends, 15)
    assert not mask[0] and not mask[1] and not mask[2]
    assert mask[3] and mask[4] and mask[5]
    assert not mask[6] and not mask[7] and not mask[8]
    assert mask[9] and mask[10]
    assert not mask[11] and not mask[12] and not mask[13] and not mask[14]


def test_build_in_turn_mask_empty_all_false() -> None:
    """AC2: no intervals → mask is False everywhere (head+tail straight)."""
    mask = build_in_turn_mask(np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), 10)
    assert not mask.any()


# ---------------------------------------------------------------------------
# augment_lateral (public API) — V3 sémantique with bfill of track_ortho
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
    enclosed by two detected turn intervals (so some samples are NOT in_turn).
    """
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


def test_augment_lateral_straight_flight_no_in_turn() -> None:
    """AC5/AC7: straight flight → no in_turn, ortho finite via global B=n-1."""
    result = augment_lateral(_make_straight_flight(100))
    in_turn = result["fdm_in_turn"].to_numpy()
    ortho = result["fdm_track_ortho_deg"].to_numpy()
    known = result["fdm_track_sel_known"].to_numpy()
    assert not in_turn.any()
    assert known.all()
    assert np.isfinite(ortho).all()


def test_augment_lateral_turn_in_middle_in_turn_mask() -> None:
    """AC5: turning flight has both in-turn samples and straight-leg samples."""
    result = augment_lateral(_make_turning_flight())
    in_turn = result["fdm_in_turn"].to_numpy()
    assert in_turn.any()
    assert (~in_turn).any()


def test_augment_lateral_ortho_finite_after_bfill() -> None:
    """AC3/AC4: bfill+ffill make track_ortho finite on every sample."""
    result = augment_lateral(_make_turning_flight())
    ortho = result["fdm_track_ortho_deg"].to_numpy()
    assert np.isfinite(ortho).all()


def test_augment_lateral_bfill_uses_next_segment_target() -> None:
    """AC3: inside-turn samples receive the *next* straight segment's bearing.

    The fixture's leg 2 runs due east (constant latitude), so its great-circle
    initial bearing target is ≈ 90°. Samples inside the first turn must be
    back-filled with that value.
    """
    result = augment_lateral(_make_turning_flight())
    in_turn = result["fdm_in_turn"].to_numpy()
    ortho = result["fdm_track_ortho_deg"].to_numpy()
    # Restrict to first-turn samples (they sit in the leg1→leg2 transition,
    # well before the leg2→leg3 turn).
    n = ortho.size
    first_half = np.zeros(n, dtype=bool)
    first_half[: n // 2] = True
    inside_first_turn = in_turn & first_half
    assert inside_first_turn.any(), "fixture must produce a detected first turn"
    targets = ortho[inside_first_turn]
    # Tolerance is loose: bearing varies along leg 2 because lon advances.
    assert np.all(np.abs(targets - 90.0) < 5.0)


def test_augment_lateral_known_flag_is_isfinite_ortho() -> None:
    """AC6: fdm_track_sel_known = isfinite(track_ortho) (no longer ~in_turn)."""
    result = augment_lateral(_make_turning_flight())
    ortho = result["fdm_track_ortho_deg"].to_numpy()
    known = result["fdm_track_sel_known"].to_numpy()
    assert np.array_equal(known, np.isfinite(ortho))


def test_augment_lateral_output_columns_unchanged() -> None:
    """AC8: only the 3 fdm_* columns are appended; nothing else changes."""
    df = _make_turning_flight()
    result = augment_lateral(df)
    added = set(result.columns) - set(df.columns)
    assert added == {"fdm_in_turn", "fdm_track_ortho_deg", "fdm_track_sel_known"}


def test_augment_lateral_dtypes_unchanged() -> None:
    """AC8: dtypes of the three appended columns are stable."""
    result = augment_lateral(_make_straight_flight())
    assert result["fdm_in_turn"].dtype == pl.Boolean
    assert result["fdm_track_sel_known"].dtype == pl.Boolean
    assert result["fdm_track_ortho_deg"].dtype == pl.Float64


def test_augment_lateral_too_short_safe_default() -> None:
    """AC7 (defensive): below-Savgol-window inputs → no-target sentinel.

    Even though AC7 says "empty intervals → no in_turn", a too-short signal
    cannot run the V3 detector at all and falls back to the safe-default
    branch: in_turn=True everywhere, no valid target.
    """
    df = pl.DataFrame(
        {
            "latitude": [45.0, 45.1],
            "longitude": [2.0, 2.0],
            "track": [0.0, 0.0],
        }
    )
    result = augment_lateral(df)
    assert len(result) == len(df)
    assert result["fdm_in_turn"].to_numpy().all()
    assert not result["fdm_track_sel_known"].to_numpy().any()
