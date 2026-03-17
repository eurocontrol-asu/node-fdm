"""Tests for segment-based selected parameter estimation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from node_fdm_data.segments import (
    add_segment_column,
    build_selected_params,
    detect_constant_segments,
)


class TestDetectConstantSegments:
    """Tests for detect_constant_segments."""

    def test_purely_constant_signal(self) -> None:
        """A constant signal produces a single segment."""
        values = np.full(100, 0.8)
        segs = detect_constant_segments(values, tol=0.01, min_len=5, use_alt=False)
        assert len(segs) == 1
        assert segs[0]["start_idx"] == 1  # first diff is at index 1
        assert segs[0]["end_idx"] == 99
        assert pytest.approx(segs[0]["var_mean"], abs=0.01) == 0.8

    def test_two_plateaus(self) -> None:
        """Two distinct constant regions separated by a ramp."""
        values = np.concatenate(
            [
                np.full(50, 0.7),  # plateau 1
                np.linspace(0.7, 0.5, 20),  # ramp
                np.full(50, 0.5),  # plateau 2
            ]
        )
        segs = detect_constant_segments(
            values,
            tol=0.005,
            min_len=10,
            use_alt=False,
        )
        assert len(segs) == 2
        assert pytest.approx(segs[0]["var_mean"], abs=0.01) == 0.7
        assert pytest.approx(segs[1]["var_mean"], abs=0.01) == 0.5

    def test_min_len_filters_short_segments(self) -> None:
        """Segments shorter than min_len are discarded."""
        values = np.concatenate(
            [
                np.full(5, 0.8),  # too short
                np.linspace(0.8, 0.6, 10),
                np.full(50, 0.6),  # long enough
            ]
        )
        segs = detect_constant_segments(
            values,
            tol=0.005,
            min_len=10,
            use_alt=False,
        )
        assert len(segs) == 1
        assert pytest.approx(segs[0]["var_mean"], abs=0.01) == 0.6

    def test_altitude_gate(self) -> None:
        """Segments below altitude threshold are excluded."""
        values = np.full(100, 0.8)
        # Altitude ramps from 10000 to 30000
        alt = np.linspace(10_000, 30_000, 100)
        segs = detect_constant_segments(
            values,
            tol=0.01,
            min_len=5,
            use_alt=True,
            alt_values=alt,
            alt_threshold=20_000,
        )
        assert len(segs) == 1
        # Segment should start around index 50 (where alt > 20000)
        assert segs[0]["start_idx"] > 40

    def test_use_alt_requires_alt_values(self) -> None:
        """Raises ValueError when use_alt=True but no alt_values."""
        with pytest.raises(ValueError, match="alt_values required"):
            detect_constant_segments(
                np.full(10, 1.0),
                use_alt=True,
                alt_values=None,
            )

    def test_min_abs_value(self) -> None:
        """Segments with mean below min_abs_value are excluded."""
        values = np.concatenate(
            [
                np.full(50, 10.0),  # above threshold
                np.full(50, 5.0),  # below threshold
            ]
        )
        segs = detect_constant_segments(
            values,
            tol=0.5,
            min_len=10,
            use_alt=False,
            min_abs_value=8.0,
        )
        assert len(segs) == 1
        assert pytest.approx(segs[0]["var_mean"], abs=0.5) == 10.0

    def test_savgol_smoothing(self) -> None:
        """Savgol smoothing detects segments in noisy data."""
        rng = np.random.default_rng(42)
        base = np.full(100, 0.75)
        noisy = base + rng.normal(0, 0.001, 100)
        segs = detect_constant_segments(
            noisy,
            tol=0.002,
            min_len=10,
            use_alt=False,
            smooth_window=15,
            smooth_method="savgol",
        )
        assert len(segs) >= 1
        assert pytest.approx(segs[0]["var_mean"], abs=0.01) == 0.75

    def test_empty_input(self) -> None:
        """Empty input returns no segments."""
        segs = detect_constant_segments(np.array([]), tol=0.01, min_len=5, use_alt=False)
        assert segs == []


class TestAddSegmentColumn:
    """Tests for add_segment_column."""

    def test_adds_column(self) -> None:
        """Segment means are written into the new column."""
        df = pl.DataFrame({"x": list(range(20))})
        segs = [{"start_idx": 5, "end_idx": 10, "var_mean": 42.0}]
        result = add_segment_column(df, segs, "y")
        assert "y" in result.columns
        vals = result["y"].to_list()
        assert vals[5] == 42.0
        assert vals[10] == 42.0
        assert np.isnan(vals[0])  # outside segment → NaN

    def test_custom_fill_value(self) -> None:
        """Non-segment rows get the custom fill value."""
        df = pl.DataFrame({"x": list(range(10))})
        result = add_segment_column(df, [], "y", fill_value=0.0)
        assert result["y"].to_list() == [0.0] * 10

    def test_multiple_segments(self) -> None:
        """Multiple segments coexist."""
        df = pl.DataFrame({"x": list(range(30))})
        segs = [
            {"start_idx": 0, "end_idx": 9, "var_mean": 1.0},
            {"start_idx": 20, "end_idx": 29, "var_mean": 2.0},
        ]
        result = add_segment_column(df, segs, "y")
        assert result["y"][0] == 1.0
        assert result["y"][25] == 2.0
        assert np.isnan(result["y"][15])


class TestBuildSelectedParams:
    """Tests for build_selected_params."""

    def test_builds_all_columns(self) -> None:
        """All selected columns are created from synthetic flight data."""
        n = 300
        rng = np.random.default_rng(42)

        # Create a realistic flight profile
        alt = np.concatenate(
            [
                np.linspace(0, 35000, 100),  # climb
                np.full(100, 35000),  # cruise
                np.linspace(35000, 0, 100),  # descent
            ]
        )
        mach = np.concatenate(
            [
                np.linspace(0.3, 0.78, 100),
                np.full(100, 0.78),  # constant Mach in cruise
                np.linspace(0.78, 0.3, 100),
            ]
        )
        cas = np.concatenate(
            [
                np.full(100, 250.0),  # constant CAS in climb
                np.linspace(250, 280, 100),
                np.full(100, 250.0),  # constant CAS in descent
            ]
        )
        vz = np.concatenate(
            [
                np.full(100, 2000.0),  # climb rate
                np.full(100, 0.0) + rng.normal(0, 5, 100),  # cruise (near zero)
                np.full(100, -1500.0),  # descent rate
            ]
        )
        gamma = np.arcsin(np.clip(vz * (0.3048 / 60) / (450 * 0.514444 + 1e-6), -1, 1))

        df = pl.DataFrame(
            {
                "altitude_ft": alt,
                "Mach": mach + rng.normal(0, 0.0001, n),
                "CAS": cas + rng.normal(0, 0.1, n),
                "vertical_rate": vz + rng.normal(0, 1, n),
                "gamma_air": gamma,
            }
        )

        config = {
            "mach": {
                "tol": 0.0005,
                "min_len": 30,
                "alt_threshold": 15000,
                "smooth_window": 10,
                "use_alt": True,
            },
            "cas": {
                "tol": 0.75,
                "min_len": 20,
                "use_alt": False,
                "smooth_window": 10,
                "smooth_method": "savgol",
            },
            "vz": {
                "tol": 25,
                "min_len": 20,
                "use_alt": False,
                "min_abs_value": 75,
                "smooth_window": 15,
                "smooth_method": "savgol",
            },
            "gamma": {
                "tol": 0.002,
                "min_len": 15,
                "use_alt": False,
                "smooth_window": 5,
                "smooth_method": "savgol",
            },
            "alt": {
                "tol": 25,
                "min_len": 5,
                "use_alt": False,
                "min_abs_value": 25,
                "smooth_window": 5,
                "smooth_method": "savgol",
            },
        }

        result = build_selected_params(df, config)

        assert "mach_sel" in result.columns
        assert "cas_sel" in result.columns
        assert "vz_sel" in result.columns
        assert "gamma_sel" in result.columns
        assert "selected_mcp" in result.columns
        assert len(result) == n

    def test_missing_columns_handled(self) -> None:
        """Missing optional columns are gracefully skipped."""
        df = pl.DataFrame(
            {
                "altitude_ft": np.full(50, 30000.0),
                "Mach": np.full(50, 0.78),
            }
        )
        config = {"mach": {"tol": 0.001, "min_len": 10, "alt_threshold": 15000, "use_alt": True}}
        result = build_selected_params(df, config)
        assert "mach_sel" in result.columns
        assert "cas_sel" not in result.columns  # CAS column not provided

    def test_mach_and_mach_sel_coexist(self) -> None:
        """After flight_processing + build_selected_params, both mach and mach_sel exist."""
        from node_fdm_data.preprocessing.opensky import flight_processing

        n = 200
        rng = np.random.default_rng(42)

        alt = np.concatenate(
            [
                np.linspace(0, 35000, 100),
                np.full(100, 35000),
            ]
        )
        mach_vals = np.concatenate(
            [
                np.linspace(0.3, 0.78, 100),
                np.full(100, 0.78) + rng.normal(0, 0.0001, 100),
            ]
        )

        df = pl.DataFrame(
            {
                "altitude": alt,
                "Mach": mach_vals,
                "TAS": np.full(n, 450.0),
                "groundspeed": np.full(n, 430.0),
                "vertical_rate": np.full(n, 0.0),
                "IAS": np.full(n, 280.0),
                "selected_mcp": np.full(n, 36000.0),
            }
        )

        # Step 1: flight_processing renames Mach → mach
        df = flight_processing(df.lazy()).collect()
        assert "mach" in df.columns
        assert "mach_sel" not in df.columns  # not yet created

        # Step 2: build_selected_params creates mach_sel from mach
        config = {
            "mach": {
                "tol": 0.001,
                "min_len": 20,
                "alt_threshold": 15000,
                "use_alt": True,
            },
            "cas": {
                "tol": 1.0,
                "min_len": 20,
                "use_alt": False,
            },
            "vz": {
                "tol": 25,
                "min_len": 20,
                "use_alt": False,
            },
        }
        df = build_selected_params(df, config)

        # Both columns exist
        assert "mach" in df.columns, "continuous mach column missing"
        assert "mach_sel" in df.columns, "segment-detected mach_sel column missing"

        # mach is continuous (no NaN), mach_sel has NaN gaps
        assert df["mach"].null_count() == 0
        mach_sel_nulls = df["mach_sel"].is_null().sum() + df["mach_sel"].is_nan().sum()
        assert mach_sel_nulls > 0, "mach_sel should have NaN gaps outside segments"
