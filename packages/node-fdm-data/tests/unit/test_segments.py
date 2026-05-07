"""Tests for segment-based selected parameter estimation."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl
import pytest
from node_fdm_pipeline.config import AltFilterConfig

from node_fdm_data.segments import (
    add_segment_column,
    build_selected_params,
    detect_alt_hold_from_vz,
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
                "raw_alt_ft": alt,
                "bds_mach_clean": mach + rng.normal(0, 0.0001, n),
                "bds_ias_kt_clean": cas + rng.normal(0, 0.1, n),
                "raw_vz_ftmin": vz + rng.normal(0, 1, n),
                "fdm_gamma_rad": gamma,
                "bds_mcp_alt_sel_ft": np.where(alt > 15000, 35000.0, np.nan),
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

        assert "fdm_mach_sel" in result.columns
        assert "fdm_cas_sel_kt" in result.columns
        assert "fdm_vz_sel_ftmin" in result.columns
        assert "fdm_gamma_sel_rad" in result.columns
        assert "fdm_mcp_alt_sel_ft" in result.columns
        assert len(result) == n

    def test_missing_columns_handled(self) -> None:
        """Missing optional columns are gracefully skipped."""
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(50, 30000.0),
                "bds_mach_clean": np.full(50, 0.78),
            }
        )
        config = {"mach": {"tol": 0.001, "min_len": 10, "alt_threshold": 15000, "use_alt": True}}
        result = build_selected_params(df, config)
        assert "fdm_mach_sel" in result.columns
        assert "fdm_cas_sel_kt" not in result.columns  # CAS column not provided

    def test_mach_and_mach_sel_coexist(self) -> None:
        """Both bds_mach_clean and fdm_mach_sel exist after segments stage."""

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
                "raw_alt_ft": alt,
                "bds_mach_clean": mach_vals,
                "fdm_tas_from_cas_kt": np.full(n, 450.0),
                "raw_gs_kt": np.full(n, 430.0),
                "raw_vz_ftmin": np.full(n, 0.0),
                "bds_ias_kt_clean": np.full(n, 280.0),
                "bds_mcp_alt_sel_ft": np.full(n, 36000.0),
            }
        )

        # build_selected_params creates fdm_mach_sel from bds_mach_clean
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
        assert "bds_mach_clean" in df.columns, "continuous era_mach column missing"
        assert "fdm_mach_sel" in df.columns, "segment-detected fdm_mach_sel column missing"

        # era_mach is continuous (no NaN), fdm_mach_sel has NaN outside segments
        assert df["bds_mach_clean"].null_count() == 0
        mach_sel_nans = df["fdm_mach_sel"].is_nan().sum()
        assert mach_sel_nans > 0, "fdm_mach_sel should have NaN gaps outside segments"


class TestBuildSelectedParamsV3:
    """Pipeline v3 tests for build_selected_params — NaN semantics and backfill."""

    @staticmethod
    def _make_flight_df(n: int = 300) -> pl.DataFrame:
        """Build synthetic flight DataFrame with v3 column names."""
        rng = np.random.default_rng(42)
        alt = np.concatenate(
            [
                np.linspace(0, 35000, n // 3),
                np.full(n // 3, 35000),
                np.linspace(35000, 0, n - 2 * (n // 3)),
            ]
        )
        mach = np.concatenate(
            [
                np.linspace(0.3, 0.78, n // 3),
                np.full(n // 3, 0.78),
                np.linspace(0.78, 0.3, n - 2 * (n // 3)),
            ]
        )
        return pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "bds_mach_clean": mach + rng.normal(0, 0.0001, n),
                "bds_ias_kt_clean": np.full(n, 280.0) + rng.normal(0, 0.1, n),
                "raw_vz_ftmin": np.concatenate(
                    [
                        np.full(n // 3, 2000.0),
                        np.full(n // 3, 0.0) + rng.normal(0, 5, n // 3),
                        np.full(n - 2 * (n // 3), -1500.0),
                    ]
                ),
                "fdm_gamma_rad": np.full(n, 0.05) + rng.normal(0, 0.0001, n),
                "bds_mcp_alt_sel_ft": np.where(
                    (np.arange(n) > 50) & (np.arange(n) < 250),
                    36000.0,
                    np.nan,
                ),
                "bds_fms_alt_sel_ft": np.where(
                    (np.arange(n) > 30) & (np.arange(n) < 270),
                    37000.0,
                    np.nan,
                ),
            }
        )

    def test_segments_fdm_prefix(self, segment_config_full: dict[str, Any]) -> None:
        """All segment columns use the fdm_ prefix."""
        df = self._make_flight_df()
        result = build_selected_params(df, segment_config_full)
        assert "fdm_mach_sel" in result.columns
        assert "fdm_cas_sel_kt" in result.columns
        assert "fdm_vz_sel_ftmin" in result.columns
        assert "fdm_gamma_sel_rad" in result.columns
        assert "fdm_alt_sel_ft" in result.columns

    def test_segments_nan_outside(self) -> None:
        """Values outside detected segments are NaN, not 0.0."""
        n = 200
        # Constant mach plateau in the middle, ramps on edges
        mach = np.concatenate(
            [
                np.linspace(0.3, 0.78, 50),
                np.full(100, 0.78),
                np.linspace(0.78, 0.3, 50),
            ]
        )
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": mach,
            }
        )
        config = {
            "mach": {"tol": 0.001, "min_len": 10, "alt_threshold": 15000, "use_alt": True},
        }
        result = build_selected_params(df, config)
        sel = result["fdm_mach_sel"]
        # Points outside segments must be NaN (not 0.0)
        nan_count = sel.is_nan().sum()
        assert nan_count > 0, "Expected NaN outside segments"
        zero_count = (sel == 0.0).sum()
        assert zero_count == 0, "NaN outside segments must NOT be filled to 0.0"

    def test_segments_backfill_mcp(self) -> None:
        """fdm_mcp_alt_sel_ft has no NaN after forward+backward fill."""
        n = 100
        # Altitude plateau in the middle → fdm_alt_sel_ft has NaN at edges
        alt = np.full(n, 35000.0)
        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "bds_mach_clean": np.full(n, 0.78),
                "bds_mcp_alt_sel_ft": np.where(
                    (np.arange(n) > 20) & (np.arange(n) < 80), 36000.0, np.nan
                ),
            }
        )
        config = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
            "alt": {"tol": 25, "min_len": 5, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_mcp_alt_sel_ft" in result.columns
        mcp = result["fdm_mcp_alt_sel_ft"]
        assert mcp.is_nan().sum() == 0, "fdm_mcp_alt_sel_ft must never have NaN"
        assert mcp.is_null().sum() == 0, "fdm_mcp_alt_sel_ft must never have null"

    def test_segments_backfill_fms(self) -> None:
        """fdm_fms_alt_sel_ft has no NaN after forward+backward fill."""
        n = 100
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": np.full(n, 0.78),
                "bds_fms_alt_sel_ft": np.where(
                    (np.arange(n) > 10) & (np.arange(n) < 90), 37000.0, np.nan
                ),
            }
        )
        config = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
        }
        result = build_selected_params(df, config)
        assert "fdm_fms_alt_sel_ft" in result.columns
        fms = result["fdm_fms_alt_sel_ft"]
        assert fms.is_nan().sum() == 0, "fdm_fms_alt_sel_ft must never have NaN"
        assert fms.is_null().sum() == 0, "fdm_fms_alt_sel_ft must never have null"

    def test_segments_config_params(self) -> None:
        """Detection uses config parameters (custom tolerance)."""
        n = 100
        # Very tight tolerance → should detect the plateau
        mach = np.concatenate(
            [np.linspace(0.3, 0.78, 30), np.full(40, 0.78), np.linspace(0.78, 0.3, 30)]
        )
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": mach,
            }
        )
        # With very high tolerance, even the ramp should be "constant"
        config_wide = {"mach": {"tol": 0.5, "min_len": 5, "alt_threshold": 15000, "use_alt": True}}
        result_wide = build_selected_params(df, config_wide)
        nan_wide = result_wide["fdm_mach_sel"].is_nan().sum()

        # With tight tolerance, only the real plateau detected
        config_tight = {
            "mach": {"tol": 0.0001, "min_len": 5, "alt_threshold": 15000, "use_alt": True}
        }
        result_tight = build_selected_params(df, config_tight)
        nan_tight = result_tight["fdm_mach_sel"].is_nan().sum()

        assert nan_tight > nan_wide, "Tighter tolerance should produce more NaN outside segments"

    def test_all_nan_mach(self) -> None:
        """All-NaN era_mach produces all-NaN fdm_mach_sel without crash."""
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": np.full(n, np.nan),
            }
        )
        config = {"mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True}}
        result = build_selected_params(df, config)
        assert "fdm_mach_sel" in result.columns
        assert result["fdm_mach_sel"].is_nan().sum() == n

    def test_short_segment_not_detected(self) -> None:
        """Segment shorter than min_len is not detected as plateau."""
        n = 50
        mach = np.linspace(0.3, 0.78, n)
        # Insert a 2-point plateau (below min_len=10)
        mach[20:22] = 0.6
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": mach,
            }
        )
        config = {"mach": {"tol": 0.001, "min_len": 10, "alt_threshold": 15000, "use_alt": True}}
        result = build_selected_params(df, config)
        # The 2-point plateau should not be detected
        assert result["fdm_mach_sel"].is_nan().sum() == n

    def test_bds40_all_null(self) -> None:
        """When bds_mcp_alt_sel_ft is all NaN, alt segments detect on raw_alt_ft.

        raw_alt_ft constant at 35000 → level segment detected → fdm_alt_sel_ft
        is NOT all NaN.  MCP/FMS backfill columns remain all NaN.
        """
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": np.full(n, 0.78),
                "bds_mcp_alt_sel_ft": np.full(n, np.nan),
                "bds_fms_alt_sel_ft": np.full(n, np.nan),
            }
        )
        config = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
            "alt": {"tol": 25, "min_len": 5, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_alt_sel_ft" in result.columns
        # raw_alt_ft constant → level segment detected → not all NaN
        assert (~result["fdm_alt_sel_ft"].is_nan()).sum() > 0
        assert "fdm_mcp_alt_sel_ft" in result.columns
        assert result["fdm_mcp_alt_sel_ft"].is_nan().sum() == n
        assert "fdm_fms_alt_sel_ft" in result.columns
        assert result["fdm_fms_alt_sel_ft"].is_nan().sum() == n


class TestTasSelected:
    """Tests for TAS plateau detection (fdm_tas_sel_kt)."""

    def test_tas_sel_column_created(self) -> None:
        """Config with 'tas' key produces fdm_tas_sel_kt column."""
        n = 200
        rng = np.random.default_rng(42)
        alt = np.full(n, 35000.0)
        mach = np.concatenate(
            [
                np.linspace(0.3, 0.78, 50),
                np.full(100, 0.78),
                np.linspace(0.78, 0.3, 50),
            ]
        )
        # Constant TAS region in climb/descent (outside Mach plateau)
        tas = np.concatenate(
            [
                np.full(50, 450.0) + rng.normal(0, 0.1, 50),  # constant TAS
                np.linspace(450, 500, 100),  # varying
                np.full(50, 500.0) + rng.normal(0, 0.1, 50),  # constant TAS
            ]
        )
        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "bds_mach_clean": mach,
                "fdm_tas_from_cas_kt": tas,
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 10, "alt_threshold": 15000, "use_alt": True},
            "tas": {
                "tol": 1.0,
                "min_len": 10,
                "use_alt": False,
                "smooth_window": 10,
                "smooth_method": "savgol",
            },
        }
        result = build_selected_params(df, config)
        assert "fdm_tas_sel_kt" in result.columns
        # Should have some detected segments (not all NaN)
        assert (~result["fdm_tas_sel_kt"].is_nan()).sum() > 0

    def test_tas_sel_masks_mach_zones(self) -> None:
        """TAS segments are NOT detected inside Mach-constant regions."""
        n = 200
        alt = np.full(n, 35000.0)
        # Mach plateau in the middle
        mach = np.concatenate(
            [
                np.linspace(0.3, 0.78, 50),
                np.full(100, 0.78),  # Mach plateau
                np.linspace(0.78, 0.3, 50),
            ]
        )
        # TAS is also constant in the same Mach-plateau region
        tas = np.full(n, 460.0)
        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "bds_mach_clean": mach,
                "fdm_tas_from_cas_kt": tas,
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 10, "alt_threshold": 15000, "use_alt": True},
            "tas": {"tol": 1.0, "min_len": 10, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_tas_sel_kt" in result.columns
        # Inside the Mach plateau (rows 50-149), TAS should be NaN (masked)
        mach_zone = result["fdm_tas_sel_kt"][50:150]
        assert mach_zone.is_nan().sum() == len(mach_zone), (
            "TAS segments must not be detected inside Mach-constant zones"
        )

    def test_tas_sel_masks_cas_zones(self) -> None:
        """TAS segments are NOT detected inside CAS-constant regions."""
        n = 200
        alt = np.full(n, 35000.0)
        # No Mach plateau (varying Mach)
        mach = np.linspace(0.3, 0.78, n)
        # CAS plateau in the first half
        cas = np.concatenate(
            [
                np.full(100, 280.0),  # CAS plateau
                np.linspace(280, 250, 100),
            ]
        )
        # TAS constant everywhere
        tas = np.full(n, 460.0)
        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "bds_mach_clean": mach,
                "bds_ias_kt_clean": cas,
                "fdm_tas_from_cas_kt": tas,
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 10, "alt_threshold": 15000, "use_alt": True},
            "cas": {
                "tol": 0.75,
                "min_len": 10,
                "use_alt": False,
                "smooth_window": 10,
                "smooth_method": "savgol",
            },
            "tas": {"tol": 1.0, "min_len": 10, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_tas_sel_kt" in result.columns
        # Inside the CAS plateau (rows 0-99), TAS should be NaN (masked)
        cas_zone = result["fdm_tas_sel_kt"][:100]
        assert cas_zone.is_nan().sum() == len(cas_zone), (
            "TAS segments must not be detected inside CAS-constant zones"
        )

    def test_tas_sel_no_config(self) -> None:
        """Without 'tas' key in config, fdm_tas_sel_kt is not created (backward compat)."""
        n = 100
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": np.full(n, 0.78),
                "fdm_tas_from_cas_kt": np.full(n, 460.0),
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
        }
        result = build_selected_params(df, config)
        assert "fdm_tas_sel_kt" not in result.columns

    def test_tas_sel_all_nan(self) -> None:
        """All-NaN era_tas_kt produces no crash and no TAS segments."""
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": np.full(n, 0.78),
                "fdm_tas_from_cas_kt": np.full(n, np.nan),
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
            "tas": {"tol": 1.0, "min_len": 10, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_tas_sel_kt" in result.columns
        assert result["fdm_tas_sel_kt"].is_nan().sum() == n

    def test_tas_sel_short_flight(self) -> None:
        """Flight with fewer points than min_len produces no TAS segments."""
        n = 5  # fewer than min_len=10
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": np.full(n, 0.78),
                "fdm_tas_from_cas_kt": np.full(n, 460.0),
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 3, "alt_threshold": 15000, "use_alt": True},
            "tas": {"tol": 1.0, "min_len": 10, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_tas_sel_kt" in result.columns
        assert result["fdm_tas_sel_kt"].is_nan().sum() == n

    def test_tas_sel_no_column(self) -> None:
        """Missing era_tas_kt column is gracefully skipped."""
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": np.full(n, 0.78),
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
            "tas": {"tol": 1.0, "min_len": 10, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_tas_sel_kt" not in result.columns


class TestTargetColumns:
    """fdm_alt_target_ft and fdm_cas_target_kt — bfill with last-point anchor."""

    def _make_climb_df(self, n: int = 100) -> pl.DataFrame:
        """Build a synthetic climb flight: altitude increasing, CAS varying."""
        alt = np.linspace(5000, 35000, n)
        mach = np.linspace(0.4, 0.78, n)
        cas = np.linspace(250, 290, n)
        return pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "bds_mach_clean": mach,
                "bds_ias_kt_clean": cas + np.random.default_rng(42).normal(0, 0.5, n),
            }
        )

    def test_alt_target_exists_and_no_null(self) -> None:
        """fdm_alt_target_ft has no nulls (bfill fills everything)."""
        df = self._make_climb_df()
        config = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
            "cas": {"tol": 0.75, "min_len": 5, "use_alt": False},
            "alt": {"tol": 25, "min_len": 5, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_alt_target_ft" in result.columns
        # bfill + last-point anchor → no nulls, no NaN
        assert result["fdm_alt_target_ft"].is_null().sum() == 0
        assert result["fdm_alt_target_ft"].is_nan().sum() == 0

    def test_alt_target_last_row_equals_actual(self) -> None:
        """Last row of fdm_alt_target_ft equals the actual altitude."""
        n = 100
        df = pl.DataFrame({"raw_alt_ft": np.linspace(5000, 35000, n)})
        config = {
            "alt": {"tol": 25, "min_len": 5, "use_alt": False},
        }
        result = build_selected_params(df, config)
        last_target = result["fdm_alt_target_ft"][-1]
        last_actual = result["raw_alt_ft"][-1]
        assert abs(last_target - last_actual) < 1e-6

    def test_alt_target_is_bfill_of_segments(self) -> None:
        """Between segments, fdm_alt_target_ft equals the NEXT segment value."""
        alt = np.concatenate(
            [
                np.full(30, 10000.0),  # level at 10000
                np.linspace(10000, 35000, 40),  # climb
                np.full(30, 35000.0),  # level at 35000
            ]
        )
        config = {
            "alt": {"tol": 25, "min_len": 5, "use_alt": False},
        }
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, config)
        # During climb (rows ~30-69), target should be 35000 (next plateau)
        mid = result["fdm_alt_target_ft"][50]
        assert abs(mid - 35000.0) < 100  # bfilled from next segment

    def test_cas_target_exists_and_no_null(self) -> None:
        """fdm_cas_target_kt has no nulls."""
        df = self._make_climb_df()
        config = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
            "cas": {"tol": 0.75, "min_len": 5, "use_alt": False},
            "alt": {"tol": 25, "min_len": 5, "use_alt": False},
        }
        result = build_selected_params(df, config)
        assert "fdm_cas_target_kt" in result.columns
        assert result["fdm_cas_target_kt"].is_null().sum() == 0
        assert result["fdm_cas_target_kt"].is_nan().sum() == 0

    def test_cas_target_last_row_equals_actual(self) -> None:
        """Last row of fdm_cas_target_kt equals actual CAS."""
        df = self._make_climb_df()
        config = {
            "cas": {"tol": 0.75, "min_len": 5, "use_alt": False},
        }
        result = build_selected_params(df, config)
        last_target = result["fdm_cas_target_kt"][-1]
        last_actual = result["bds_ias_kt_clean"][-1]
        assert abs(last_target - last_actual) < 1e-6


# === Merged from test_build_selected_params.py: AXM-1625 AC1-AC8 (crossover-aware Mach/CAS) ===

_BSP_FT_TO_M = 0.3048
_BSP_KT_TO_MS = 0.5144444444
_BSP_MS_TO_KT = 1.0 / _BSP_KT_TO_MS


def _bsp_config(**overrides: dict[str, object]) -> dict[str, object]:
    cfg: dict[str, object] = {
        "mach": {"min_length": 30, "tolerance": 0.005},
        "cas": {"min_length": 30, "tolerance": 2.0},
        "vz": {"min_length": 10, "tolerance": 50.0},
        "alt": {"min_length": 30, "tolerance": 50.0},
    }
    cfg.update(overrides)
    return cfg


def _bsp_flight(  # noqa: PLR0913
    *,
    alt_ft: np.ndarray,
    mach: np.ndarray,
    cas: np.ndarray,
    vz: np.ndarray | None = None,
    tas_kt: np.ndarray | None = None,
    era_temp_K: np.ndarray | None = None,  # noqa: N803
) -> pl.DataFrame:
    n = alt_ft.size
    if vz is None:
        vz = np.zeros(n)
    cols: dict[str, np.ndarray] = {
        "bds_mach_clean": mach,
        "bds_ias_kt_clean": cas,
        "raw_alt_ft": alt_ft,
        "raw_vz_ftmin": vz,
    }
    if tas_kt is not None:
        cols["fdm_tas_from_cas_kt"] = tas_kt
    if era_temp_K is not None:
        cols["era_temp_K"] = era_temp_K
    return pl.DataFrame(cols)


def _bsp_three_phase_alt(
    n_climb: int, n_cruise: int, n_descent: int, cruise_ft: float = 34000.0
) -> np.ndarray:
    n = n_climb + n_cruise + n_descent
    alt = np.empty(n, dtype=np.float64)
    alt[:n_climb] = np.linspace(1000.0, cruise_ft, n_climb)
    alt[n_climb : n_climb + n_cruise] = cruise_ft
    alt[n_climb + n_cruise :] = np.linspace(cruise_ft, 1000.0, n_descent)
    return alt


class TestMachPlateauOnly:
    """AC1: Mach detected only on altitude plateaus."""

    def test_mach_detected_only_on_alt_plateau(self):
        n_climb, n_cruise, n_descent = 100, 100, 100
        alt = _bsp_three_phase_alt(n_climb, n_cruise, n_descent)
        n = alt.size
        mach = np.full(n, 0.78)
        cas = np.full(n, np.nan)

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())

        sel = out["fdm_mach_sel"].to_numpy()
        plateau_mask = alt == 34000.0
        assert np.any(~np.isnan(sel[plateau_mask]))
        assert np.all(np.isnan(sel[~plateau_mask]))

    def test_mach_outside_plateau_dropped(self):
        n = 200
        alt = np.linspace(1000.0, 34000.0, n)
        mach = np.full(n, 0.78)
        cas = np.full(n, np.nan)

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())

        assert np.all(np.isnan(out["fdm_mach_sel"].to_numpy()))


class TestAberrantMachFilter:
    """AC7: Reject Mach segments with mean < 0.5."""

    def test_aberrant_low_mach_segment_filtered(self):
        alt = _bsp_three_phase_alt(100, 100, 100)
        n = alt.size
        mach = np.full(n, 0.23)
        cas = np.full(n, np.nan)

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())

        assert np.all(np.isnan(out["fdm_mach_sel"].to_numpy()))

    def test_aberrant_high_mach_segment_kept(self):
        alt = _bsp_three_phase_alt(100, 100, 100)
        n = alt.size
        mach = np.full(n, 0.78)
        cas = np.full(n, np.nan)

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())

        sel = out["fdm_mach_sel"].to_numpy()
        plateau_mask = alt == 34000.0
        assert np.any(~np.isnan(sel[plateau_mask]))


class TestCasOptimisation:
    """AC2/AC3: Transition CAS optimisation."""

    @staticmethod
    def _build_consistent_flight(  # noqa: PLR0913
        *,
        cas_climb_kt: float,
        cas_descent_kt: float,
        mach_cruise: float,
        n_climb: int = 100,
        n_cruise: int = 100,
        n_descent: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas

        n = n_climb + n_cruise + n_descent
        alt_ft = _bsp_three_phase_alt(n_climb, n_cruise, n_descent)
        alt_m = alt_ft * _BSP_FT_TO_M

        mach = np.full(n, np.nan)
        mach[n_climb : n_climb + n_cruise] = mach_cruise

        cas_real_kt = np.full(n, np.nan)
        cas_real_kt[:n_climb] = cas_climb_kt
        cas_real_kt[n_climb + n_cruise :] = cas_descent_kt

        tas_ms = np.full(n, np.nan)
        tas_ms[:n_climb] = np.asarray(cas_to_tas(cas_climb_kt * _BSP_KT_TO_MS, alt_m[:n_climb]))
        tas_ms[n_climb : n_climb + n_cruise] = np.asarray(
            mach_to_tas(mach_cruise, alt_m[n_climb : n_climb + n_cruise])
        )
        tas_ms[n_climb + n_cruise :] = np.asarray(
            cas_to_tas(cas_descent_kt * _BSP_KT_TO_MS, alt_m[n_climb + n_cruise :])
        )
        tas_real_kt = tas_ms * _BSP_MS_TO_KT
        return alt_ft, mach, cas_real_kt, tas_real_kt

    def test_cas_optimisation_recovers_known_value(self):
        alt_ft, mach, cas_real_kt, tas_kt = self._build_consistent_flight(
            cas_climb_kt=280.0,
            cas_descent_kt=270.0,
            mach_cruise=0.78,
        )

        out = build_selected_params(
            _bsp_flight(alt_ft=alt_ft, mach=mach, cas=cas_real_kt, tas_kt=tas_kt),
            _bsp_config(),
        )
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()

        climb_vals = cas_sel[:100]
        non_nan = climb_vals[~np.isnan(climb_vals)]
        assert non_nan.size > 0
        assert np.all(np.abs(non_nan - 280.0) <= 1.0)

    def test_cas_deviation_cutoff(self):
        alt_ft, mach, cas_real_kt, tas_kt = self._build_consistent_flight(
            cas_climb_kt=280.0,
            cas_descent_kt=270.0,
            mach_cruise=0.78,
        )
        cas_real_kt = cas_real_kt.copy()
        cas_real_kt[:50] = 288.0

        out = build_selected_params(
            _bsp_flight(alt_ft=alt_ft, mach=mach, cas=cas_real_kt, tas_kt=tas_kt),
            _bsp_config(),
        )
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()
        assert np.all(np.isnan(cas_sel[:50]))


class TestEraTemperature:
    """AC4: Use era_temp_K when present, fall back to ISA otherwise."""

    def test_uses_era_temp_when_present(self):
        from node_fdm_data.physics.isa import isa_temperature

        alt = _bsp_three_phase_alt(100, 100, 100)
        alt_m = alt * _BSP_FT_TO_M
        n = alt.size
        mach = np.full(n, np.nan)
        mach[100:200] = 0.78
        cas = np.full(n, np.nan)
        temp = np.asarray(isa_temperature(alt_m), dtype=np.float64) - 10.0

        out = build_selected_params(
            _bsp_flight(alt_ft=alt, mach=mach, cas=cas, era_temp_K=temp),
            _bsp_config(),
        )

        from node_fdm_data.physics.speed import mach_to_tas_real

        expected_ms = np.asarray(mach_to_tas_real(0.78, temp[100:200]))
        expected_kt = expected_ms * _BSP_MS_TO_KT
        target = out["fdm_tas_target_kt"].to_numpy()
        idx = np.where(~np.isnan(target[100:200]))[0]
        assert idx.size > 0
        for i in idx[:5]:
            assert abs(target[100 + i] - expected_kt[i]) < 1.0

    def test_falls_back_to_isa_when_era_temp_absent(self):
        from node_fdm_data.physics.speed import mach_to_tas

        alt = _bsp_three_phase_alt(100, 100, 100)
        alt_m = alt * _BSP_FT_TO_M
        n = alt.size
        mach = np.full(n, np.nan)
        mach[100:200] = 0.78
        cas = np.full(n, np.nan)

        out = build_selected_params(
            _bsp_flight(alt_ft=alt, mach=mach, cas=cas),
            _bsp_config(),
        )

        expected_kt = np.asarray(mach_to_tas(0.78, alt_m[100:200])) * _BSP_MS_TO_KT
        target = out["fdm_tas_target_kt"].to_numpy()
        idx = np.where(~np.isnan(target[100:200]))[0]
        assert idx.size > 0
        for i in idx[:5]:
            assert abs(target[100 + i] - expected_kt[i]) < 0.5


class TestTasTargetEnvelope:
    """AC5: TAS target = min(mach_to_tas, cas_to_tas) on overlap; NaN outside."""

    def test_tas_target_uses_envelope_on_overlap(self):
        from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas

        alt_ft, mach, cas_real_kt, tas_kt = TestCasOptimisation._build_consistent_flight(
            cas_climb_kt=280.0,
            cas_descent_kt=270.0,
            mach_cruise=0.78,
        )
        alt_m = alt_ft * _BSP_FT_TO_M

        out = build_selected_params(
            _bsp_flight(alt_ft=alt_ft, mach=mach, cas=cas_real_kt, tas_kt=tas_kt),
            _bsp_config(),
        )
        mach_sel = out["fdm_mach_sel"].to_numpy()
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()
        target = out["fdm_tas_target_kt"].to_numpy()

        overlap = ~np.isnan(mach_sel) & ~np.isnan(cas_sel)
        if not overlap.any():
            pytest.skip("synthetic flight produced no overlap rows")
        tas_mach = np.asarray(mach_to_tas(mach_sel[overlap], alt_m[overlap])) * _BSP_MS_TO_KT
        tas_cas = (
            np.asarray(cas_to_tas(cas_sel[overlap] * _BSP_KT_TO_MS, alt_m[overlap]))
            * _BSP_MS_TO_KT
        )
        expected = np.minimum(tas_mach, tas_cas)
        np.testing.assert_allclose(target[overlap], expected, atol=1.0)

    def test_tas_target_nan_outside_segments(self):
        n = 200
        alt = np.linspace(1000.0, 34000.0, n)
        mach = np.full(n, np.nan)
        cas = np.full(n, np.nan)

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())

        assert np.all(np.isnan(out["fdm_tas_target_kt"].to_numpy()))

    def test_tas_target_no_global_backfill(self):
        n_climb, n_cruise, n_tail = 100, 100, 50
        n = n_climb + n_cruise + n_tail
        alt = np.empty(n)
        alt[:n_climb] = np.linspace(1000.0, 34000.0, n_climb)
        alt[n_climb : n_climb + n_cruise] = 34000.0
        alt[n_climb + n_cruise :] = np.nan
        mach = np.full(n, np.nan)
        mach[n_climb : n_climb + n_cruise] = 0.78
        cas = np.full(n, np.nan)

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())
        target = out["fdm_tas_target_kt"].to_numpy()
        assert np.all(np.isnan(target[-n_tail:]))


class TestTasTargetKnownMaskBSP:
    """AC6: fdm_tas_target_known column."""

    def test_tas_target_known_mask_emitted(self):
        alt = _bsp_three_phase_alt(100, 100, 100)
        n = alt.size
        mach = np.full(n, np.nan)
        mach[100:200] = 0.78
        cas = np.full(n, np.nan)

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())

        assert "fdm_tas_target_known" in out.columns
        known = out["fdm_tas_target_known"].to_numpy()
        target = out["fdm_tas_target_kt"].to_numpy()
        np.testing.assert_array_equal(known.astype(bool), ~np.isnan(target))


class TestNoOptimisationEdgeCases:
    """AC8: Skip optimisation when first/last plateau is within margin."""

    def test_no_climb_skips_optimisation(self):
        n = 200
        alt = np.empty(n)
        alt[:5] = np.linspace(33000.0, 34000.0, 5)
        alt[5:150] = 34000.0
        alt[150:] = np.linspace(34000.0, 1000.0, 50)
        mach = np.full(n, np.nan)
        mach[5:150] = 0.78
        cas = np.full(n, np.nan)
        cas[150:] = 270.0

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()
        assert np.all(np.isnan(cas_sel[:5]))

    def test_no_descent_skips_optimisation(self):
        n = 200
        alt = np.empty(n)
        alt[:50] = np.linspace(1000.0, 34000.0, 50)
        alt[50:195] = 34000.0
        alt[195:] = np.linspace(34000.0, 33000.0, 5)
        mach = np.full(n, np.nan)
        mach[50:195] = 0.78
        cas = np.full(n, np.nan)
        cas[:50] = 280.0

        out = build_selected_params(_bsp_flight(alt_ft=alt, mach=mach, cas=cas), _bsp_config())
        cas_sel = out["fdm_cas_sel_kt"].to_numpy()
        assert np.all(np.isnan(cas_sel[-5:]))


class TestBuildSelectedParamsEmpty:
    """AC1: Empty DataFrame — no error, expected columns present."""

    def test_empty_dataframe(self):
        df = pl.DataFrame(
            {
                "bds_mach_clean": pl.Series([], dtype=pl.Float64),
                "bds_ias_kt_clean": pl.Series([], dtype=pl.Float64),
                "raw_alt_ft": pl.Series([], dtype=pl.Float64),
                "raw_vz_ftmin": pl.Series([], dtype=pl.Float64),
            }
        )
        out = build_selected_params(df, _bsp_config())
        for col in (
            "fdm_mach_sel",
            "fdm_cas_sel_kt",
            "fdm_alt_sel_ft",
            "fdm_tas_target_kt",
            "fdm_tas_target_known",
        ):
            assert col in out.columns
        assert len(out) == 0


# === Merged from test_gamma_from_alt.py: AXM-802 (altitude-plateau gamma=0) ===


def _gfa_alt_config() -> dict[str, Any]:
    """Minimal config that enables altitude segment detection."""
    return {
        "alt": {"tol": 25, "min_len": 5, "use_alt": False},
    }


class TestGammaFromAltColumn:
    """fdm_gamma_from_alt_rad column: 0.0 in level-flight, NaN elsewhere."""

    def test_column_created_when_alt_segments_exist(self) -> None:
        alt = np.concatenate(
            [
                np.linspace(5000, 35000, 50),
                np.full(50, 35000.0),
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _gfa_alt_config())
        assert "fdm_gamma_from_alt_rad" in result.columns

    def test_zero_inside_altitude_plateau(self) -> None:
        n = 100
        alt = np.full(n, 35000.0)
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _gfa_alt_config())

        alt_sel = result["fdm_alt_sel_ft"]
        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()

        level_mask = ~alt_sel.is_nan()
        assert level_mask.sum() > 0
        stabilised = gamma[15:]
        assert np.nansum(stabilised == 0.0) > 0

    def test_nan_outside_altitude_plateau(self) -> None:
        alt = np.concatenate(
            [
                np.linspace(5000, 35000, 50),
                np.full(50, 35000.0),
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _gfa_alt_config())

        alt_sel = result["fdm_alt_sel_ft"]
        gamma = result["fdm_gamma_from_alt_rad"]

        non_level_mask = alt_sel.is_nan()
        assert non_level_mask.sum() > 0
        gamma_outside = gamma.filter(non_level_mask)
        assert gamma_outside.is_nan().all()

    def test_exact_alignment_with_alt_sel(self) -> None:
        alt = np.concatenate(
            [
                np.full(60, 10000.0),
                np.linspace(10000, 35000, 80),
                np.full(60, 35000.0),
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _gfa_alt_config())

        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()

        level_mask = ~np.isnan(alt_sel)
        assert np.all(np.isnan(gamma[~level_mask]))
        level_vals = gamma[level_mask]
        n_nan = np.isnan(level_vals).sum()
        n_zero = (level_vals == 0.0).sum()
        assert n_nan > 0
        assert n_zero > 0
        assert n_nan + n_zero == len(level_vals)


class TestGammaFromAltIntegration:
    """Integration: gamma_from_alt inside full build_selected_params."""

    def test_full_config_produces_column(self, segment_config_full: dict[str, Any]) -> None:
        n = 300
        rng = np.random.default_rng(42)
        alt = np.concatenate(
            [
                np.linspace(0, 35000, 100),
                np.full(100, 35000),
                np.linspace(35000, 0, 100),
            ]
        )
        mach = np.concatenate(
            [
                np.linspace(0.3, 0.78, 100),
                np.full(100, 0.78),
                np.linspace(0.78, 0.3, 100),
            ]
        )
        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "era_mach": mach + rng.normal(0, 0.0001, n),
            }
        )
        result = build_selected_params(df, segment_config_full)

        assert "fdm_gamma_from_alt_rad" in result.columns
        gamma = result["fdm_gamma_from_alt_rad"]
        assert (gamma == 0.0).sum() > 0

    def test_coexists_with_gamma_sel(self) -> None:
        n = 300
        rng = np.random.default_rng(42)
        alt = np.concatenate(
            [
                np.linspace(0, 35000, 100),
                np.full(100, 35000),
                np.linspace(35000, 0, 100),
            ]
        )
        vz = np.concatenate(
            [
                np.full(100, 2000.0),
                np.full(100, 0.0) + rng.normal(0, 5, 100),
                np.full(100, -1500.0),
            ]
        )
        gamma = np.arcsin(np.clip(vz * (0.3048 / 60) / (450 * 0.514444 + 1e-6), -1, 1))
        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "era_mach": np.linspace(0.3, 0.78, n),
                "raw_vz_ftmin": vz,
                "fdm_gamma_rad": gamma,
            }
        )
        config = {
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

        assert "fdm_gamma_sel_rad" in result.columns
        assert "fdm_gamma_from_alt_rad" in result.columns

    def test_length_preserved(self) -> None:
        n = 150
        alt = np.concatenate(
            [
                np.full(50, 10000.0),
                np.linspace(10000, 35000, 50),
                np.full(50, 35000.0),
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _gfa_alt_config())
        assert len(result) == n
        assert len(result["fdm_gamma_from_alt_rad"]) == n


class TestGammaFromAltEdgeCases:
    """Edge cases for altitude-plateau gamma detection."""

    def test_no_alt_config_no_column(self) -> None:
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "era_mach": np.full(n, 0.78),
            }
        )
        config: dict[str, Any] = {
            "mach": {"tol": 0.001, "min_len": 5, "alt_threshold": 15000, "use_alt": True},
        }
        result = build_selected_params(df, config)
        assert "fdm_gamma_from_alt_rad" not in result.columns

    def test_no_altitude_plateau_all_nan(self) -> None:
        n = 100
        alt = np.linspace(5000, 35000, n)
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _gfa_alt_config())

        assert "fdm_gamma_from_alt_rad" in result.columns
        assert result["fdm_gamma_from_alt_rad"].is_nan().all()

    def test_entire_flight_level(self) -> None:
        n = 100
        df = pl.DataFrame({"raw_alt_ft": np.full(n, 35000.0)})
        result = build_selected_params(df, _gfa_alt_config())

        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        level_count = (~np.isnan(alt_sel)).sum()
        zero_count = (gamma == 0.0).sum()
        nan_count = np.isnan(gamma[~np.isnan(alt_sel)]).sum()
        assert zero_count + nan_count == level_count
        assert nan_count <= 15

    def test_multiple_level_segments(self) -> None:
        alt = np.concatenate(
            [
                np.full(40, 10000.0),
                np.linspace(10000, 35000, 40),
                np.full(40, 35000.0),
                np.linspace(35000, 20000, 40),
                np.full(40, 20000.0),
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _gfa_alt_config())

        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()

        level_mask = ~np.isnan(alt_sel)
        assert level_mask.sum() > 0
        level_vals = gamma[level_mask]
        n_zero = (level_vals == 0.0).sum()
        n_nan = np.isnan(level_vals).sum()
        assert n_zero > 0
        assert n_nan > 0
        assert n_nan + n_zero == len(level_vals)

    def test_short_flight_below_min_len(self) -> None:
        n = 3
        df = pl.DataFrame({"raw_alt_ft": np.full(n, 35000.0)})
        result = build_selected_params(df, _gfa_alt_config())

        assert "fdm_gamma_from_alt_rad" in result.columns
        assert result["fdm_gamma_from_alt_rad"].is_nan().all()

    def test_dtype_is_float64(self) -> None:
        n = 50
        df = pl.DataFrame({"raw_alt_ft": np.full(n, 35000.0)})
        result = build_selected_params(df, _gfa_alt_config())

        assert result["fdm_gamma_from_alt_rad"].dtype == pl.Float64

    def test_segment_shorter_than_relax(self) -> None:
        alt = np.concatenate(
            [
                np.linspace(5000, 35000, 50),
                np.full(10, 35000.0),
                np.linspace(35000, 5000, 50),
            ]
        )
        df = pl.DataFrame({"raw_alt_ft": alt})
        result = build_selected_params(df, _gfa_alt_config())

        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        level_mask = ~np.isnan(alt_sel)
        if level_mask.sum() > 0:
            assert np.all(np.isnan(gamma[level_mask]))

    def test_relax_zero_disables_relaxation(self) -> None:
        n = 100
        alt = np.full(n, 35000.0)
        df = pl.DataFrame({"raw_alt_ft": alt})
        config = {**_gfa_alt_config(), "alt_hold_relax": 0}
        result = build_selected_params(df, config)

        gamma = result["fdm_gamma_from_alt_rad"].to_numpy()
        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        level_mask = ~np.isnan(alt_sel)
        assert (gamma[level_mask] == 0.0).all()


# === Merged from test_gamma_target.py: AXM-809 (unified gamma target v2) ===

_GT_FT_MIN_TO_MS = 0.3048 / 60
_GT_KT_TO_MS = 0.514444


def _gt_make_standard_flight(n: int = 300, *, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    third = n // 3

    alt = np.concatenate(
        [
            np.linspace(0, 35_000, third),
            np.full(n - 2 * third, 35_000),
            np.linspace(35_000, 0, third),
        ]
    )

    vz = np.concatenate(
        [
            np.full(third, 1500.0) + rng.normal(0, 5, third),
            np.full(n - 2 * third, 0.0) + rng.normal(0, 5, n - 2 * third),
            np.full(third, -1500.0) + rng.normal(0, 5, third),
        ]
    )

    tas_kt = np.concatenate(
        [
            np.linspace(200, 450, third),
            np.full(n - 2 * third, 450.0) + rng.normal(0, 0.2, n - 2 * third),
            np.linspace(450, 200, third),
        ]
    )

    vz_ms = vz * _GT_FT_MIN_TO_MS
    tas_ms = tas_kt * _GT_KT_TO_MS
    gamma = np.arcsin(np.clip(vz_ms / np.where(tas_ms == 0, np.nan, tas_ms), -1, 1))

    return pl.DataFrame(
        {
            "raw_alt_ft": alt,
            "raw_vz_ftmin": vz,
            "fdm_tas_from_cas_kt": tas_kt,
            "fdm_gamma_rad": gamma,
        }
    )


class TestGammaTargetExists:
    """fdm_gamma_target_rad column is created with full config."""

    def test_gamma_target_exists(self, segment_config_full: dict[str, Any]) -> None:
        df = _gt_make_standard_flight()
        result = build_selected_params(df, segment_config_full)
        assert "fdm_gamma_target_rad" in result.columns


class TestGammaTargetVzSource:
    """Where vz plateau is detected, gamma_target ~ arcsin(vz_ms / tas_ms)."""

    def test_gamma_target_vz_source(self, segment_config_full: dict[str, Any]) -> None:
        n = 200
        vz = np.concatenate(
            [
                np.linspace(0, 1500, 50),
                np.full(100, 1500.0),
                np.linspace(1500, 0, 50),
            ]
        )
        tas_kt = np.full(n, 300.0)
        alt = np.linspace(5_000, 35_000, n)
        vz_ms = vz * _GT_FT_MIN_TO_MS
        tas_ms = tas_kt * _GT_KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / tas_ms, -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, segment_config_full)

        expected_gamma = float(np.arcsin(1500.0 * _GT_FT_MIN_TO_MS / (300.0 * _GT_KT_TO_MS)))
        target = result["fdm_gamma_target_rad"].to_numpy()

        plateau_vals = target[70:130]
        np.testing.assert_allclose(plateau_vals, expected_gamma, atol=0.01)


class TestGammaTargetGammaSource:
    """Where gamma plateau is detected, gamma_target equals that plateau value."""

    def test_gamma_target_gamma_source(self) -> None:
        n = 200
        rng = np.random.default_rng(42)
        gamma_val = -0.05
        gamma = np.concatenate(
            [
                np.linspace(-0.1, gamma_val, 50),
                np.full(100, gamma_val) + rng.normal(0, 0.0001, 100),
                np.linspace(gamma_val, 0.0, 50),
            ]
        )
        tas_kt = np.full(n, 400.0)
        alt = np.linspace(35_000, 10_000, n)

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        config: dict[str, Any] = {
            "gamma": {
                "tol": 0.002,
                "min_len": 15,
                "use_alt": False,
                "smooth_window": 5,
                "smooth_method": "savgol",
            },
        }
        result = build_selected_params(df, config)

        target = result["fdm_gamma_target_rad"].to_numpy()
        plateau_vals = target[70:130]
        np.testing.assert_allclose(plateau_vals, gamma_val, atol=0.005)


class TestGammaTargetZeroTas:
    """vz_sel with zero TAS — no crash, clamped gamma."""

    def test_zero_tas_no_crash(self, segment_config_full: dict[str, Any]) -> None:
        n = 100
        alt = np.linspace(5_000, 15_000, n)
        vz = np.full(n, 1500.0)
        tas_kt = np.zeros(n)
        gamma = np.full(n, 0.0)

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, segment_config_full)

        if "fdm_gamma_target_rad" in result.columns:
            target = result["fdm_gamma_target_rad"].to_numpy()
            assert not np.any(np.isinf(target[~np.isnan(target)]))


class TestGammaTargetKnownMask:
    """Gaps between segments have gamma_target_known=0 (filled to 0.0, not NaN)."""

    def test_gamma_target_known_gaps(self) -> None:
        n = 300
        rng = np.random.default_rng(42)
        third = n // 3

        vz = np.concatenate(
            [
                np.full(third, 1500.0) + rng.normal(0, 5, third),
                rng.uniform(-200, 200, n - 2 * third),
                np.full(third, -1500.0) + rng.normal(0, 5, third),
            ]
        )
        alt = np.concatenate(
            [
                np.linspace(5_000, 20_000, third),
                np.linspace(20_000, 20_500, n - 2 * third),
                np.linspace(20_500, 5_000, third),
            ]
        )
        tas_kt = np.full(n, 350.0)
        vz_ms = vz * _GT_FT_MIN_TO_MS
        tas_ms = tas_kt * _GT_KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / tas_ms, -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        config: dict[str, Any] = {
            "vz": {
                "tol": 25,
                "min_len": 20,
                "use_alt": False,
                "min_abs_value": 75,
                "smooth_window": 15,
                "smooth_method": "savgol",
            },
        }
        result = build_selected_params(df, config)

        assert "fdm_gamma_target_known" in result.columns
        known = result["fdm_gamma_target_known"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()

        unknown_count = (known == 0).sum()
        assert unknown_count > 0
        assert not np.any(np.isnan(target))


class TestGammaTargetPriorityVzOverGamma:
    """vz->gamma (highest priority) overrides gamma_sel when both overlap."""

    def test_gamma_target_priority_vz_over_gamma(self) -> None:
        n = 200
        rng = np.random.default_rng(42)

        vz_val = 1000.0
        tas_val_kt = 400.0
        expected_vz_gamma = float(
            np.arcsin(vz_val * _GT_FT_MIN_TO_MS / (tas_val_kt * _GT_KT_TO_MS))
        )
        forced_gamma = 0.08

        vz = np.full(n, vz_val) + rng.normal(0, 2, n)
        tas_kt = np.full(n, tas_val_kt)
        alt = np.linspace(5_000, 25_000, n)
        gamma = np.full(n, forced_gamma) + rng.normal(0, 0.0001, n)

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        config: dict[str, Any] = {
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
        }

        result = build_selected_params(df, config)
        target = result["fdm_gamma_target_rad"].to_numpy()

        vz_sel = result["fdm_vz_sel_ftmin"].to_numpy()
        vz_mask = ~np.isnan(vz_sel)
        assert vz_mask.sum() > 20

        np.testing.assert_allclose(
            target[vz_mask],
            expected_vz_gamma,
            atol=0.005,
            err_msg="vz->gamma should override gamma_sel",
        )


class TestGammaTargetPriorityGammaOverAlt:
    """gamma_sel overrides gamma_from_alt (alt is lowest priority)."""

    def test_gamma_target_priority_gamma_over_alt(
        self, segment_config_full: dict[str, Any]
    ) -> None:
        n = 200
        rng = np.random.default_rng(42)

        alt = np.full(n, 35_000.0)
        gamma_val = -0.03
        gamma = np.full(n, gamma_val) + rng.normal(0, 0.0001, n)
        vz = rng.normal(0, 5, n)
        tas_kt = np.full(n, 450.0)

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, segment_config_full)

        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        alt_mask = ~np.isnan(alt_sel)
        assert alt_mask.sum() > 0

        gamma_sel = result["fdm_gamma_sel_rad"].to_numpy()
        gamma_mask = ~np.isnan(gamma_sel)
        assert gamma_mask.sum() > 0

        overlap = alt_mask & gamma_mask
        assert overlap.sum() > 0

        target = result["fdm_gamma_target_rad"].to_numpy()
        np.testing.assert_allclose(
            target[overlap],
            gamma_val,
            atol=0.005,
            err_msg="gamma_sel should override gamma_from_alt",
        )


class TestGammaTargetNoNearZero:
    """Gamma_sel does not detect near-zero plateaus during cruise."""

    def test_gamma_target_no_near_zero(self, segment_config_full: dict[str, Any]) -> None:
        df = _gt_make_standard_flight()
        result = build_selected_params(df, segment_config_full)

        if "fdm_gamma_sel_rad" not in result.columns:
            pytest.skip("gamma_sel not produced")

        gamma_sel = result["fdm_gamma_sel_rad"].to_numpy()
        valid = gamma_sel[~np.isnan(gamma_sel)]

        if len(valid) > 0:
            assert np.all(np.abs(valid) > 0.005), (
                f"gamma_sel contains near-zero values: {valid[np.abs(valid) <= 0.005]}"
            )


class TestGammaDiffNanFilledZero:
    """gamma_diff = 0 where gamma_target is NaN (not NaN propagation)."""

    def test_gamma_diff_nan_filled_zero(self) -> None:
        from node_fdm_data.preprocessing.convert import convert_si

        n = 50
        rng = np.random.default_rng(42)

        gamma_actual = rng.uniform(-0.05, 0.05, n)
        gamma_target = gamma_actual.copy()
        gamma_target[15:35] = np.nan

        df = pl.DataFrame(
            {
                "fdm_gamma_target_rad": gamma_target,
                "fdm_gamma_rad": gamma_actual,
                "raw_alt_ft": np.linspace(10_000, 35_000, n),
                "fdm_tas_from_cas_kt": np.full(n, 400.0),
                "raw_vz_ftmin": np.full(n, 0.0),
            }
        )

        result = convert_si(df)

        assert "fdm_gamma_diff_rad" in result.columns
        diff = result["fdm_gamma_diff_rad"].to_numpy()

        nan_mask = np.isnan(gamma_target)
        assert not np.any(np.isnan(diff[nan_mask]))
        np.testing.assert_allclose(diff[nan_mask], 0.0, atol=1e-10)


class TestGammaTargetAllUnknown:
    """Very short flight, no segments → gamma_target_known all 0, target all 0.0."""

    def test_all_unknown_target(self, segment_config_full: dict[str, Any]) -> None:
        rng = np.random.default_rng(99)
        n = 10
        vz = rng.uniform(-500, 500, n)
        tas_kt = rng.uniform(200, 400, n)
        alt = np.linspace(5_000, 10_000, n)
        vz_ms = vz * _GT_FT_MIN_TO_MS
        tas_ms = tas_kt * _GT_KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / np.where(tas_ms == 0, np.nan, tas_ms), -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, segment_config_full)

        known = result["fdm_gamma_target_known"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()
        assert np.all(known == 0)
        assert not np.any(np.isnan(target))


class TestGammaTargetOnlyAltSel:
    """Only alt_sel present → gamma_target = 0 in cruise, known=0 elsewhere."""

    def test_only_alt_sel(self) -> None:
        n = 300
        rng = np.random.default_rng(42)
        third = n // 3

        alt = np.concatenate(
            [
                np.linspace(10_000, 35_000, third),
                np.full(n - 2 * third, 35_000.0),
                np.linspace(35_000, 10_000, third),
            ]
        )
        vz = rng.uniform(-200, 200, n)
        tas_kt = np.full(n, 400.0)
        vz_ms = vz * _GT_FT_MIN_TO_MS
        tas_ms = tas_kt * _GT_KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / tas_ms, -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        config: dict[str, Any] = {
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

        target_arr = result["fdm_gamma_target_rad"].to_numpy()
        known = result["fdm_gamma_target_known"].to_numpy()

        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        alt_mask = ~np.isnan(alt_sel)
        assert alt_mask.sum() > 0

        stabilised = alt_mask.copy()
        gamma_from_alt = result["fdm_gamma_from_alt_rad"].to_numpy()
        relaxed = alt_mask & np.isnan(gamma_from_alt)
        stabilised[relaxed] = False
        assert stabilised.sum() > 0
        np.testing.assert_allclose(target_arr[stabilised], 0.0, atol=1e-10)
        assert np.all(known[stabilised] == 1.0)

        assert np.all(known[relaxed] == 0.0)

        non_alt_mask = np.isnan(alt_sel)
        assert np.all(known[non_alt_mask] == 0.0)


class TestGammaTargetAltHoldSource:
    """Where only alt plateau detected (no vz/gamma), gamma_target = 0."""

    def test_gamma_target_alt_hold_source(self, segment_config_full: dict[str, Any]) -> None:
        n = 200
        rng = np.random.default_rng(42)
        alt = np.concatenate(
            [
                np.linspace(10_000, 35_000, 50),
                np.full(100, 35_000.0),
                np.linspace(35_000, 10_000, 50),
            ]
        )
        vz = np.concatenate(
            [
                np.full(50, 1500.0),
                np.full(100, 0.0) + rng.normal(0, 5, 100),
                np.full(50, -1500.0),
            ]
        )
        tas_kt = np.full(n, 450.0)
        vz_ms = vz * _GT_FT_MIN_TO_MS
        tas_ms = tas_kt * _GT_KT_TO_MS
        gamma = np.arcsin(np.clip(vz_ms / np.where(tas_ms == 0, np.nan, tas_ms), -1, 1))

        df = pl.DataFrame(
            {
                "raw_alt_ft": alt,
                "raw_vz_ftmin": vz,
                "fdm_tas_from_cas_kt": tas_kt,
                "fdm_gamma_rad": gamma,
            }
        )

        result = build_selected_params(df, segment_config_full)

        alt_sel = result["fdm_alt_sel_ft"].to_numpy()
        target = result["fdm_gamma_target_rad"].to_numpy()

        alt_hold_mask = ~np.isnan(alt_sel)
        assert alt_hold_mask.sum() > 0

        gamma_from_alt = result["fdm_gamma_from_alt_rad"].to_numpy()
        stabilised_alt = alt_hold_mask & ~np.isnan(gamma_from_alt)
        if "fdm_vz_sel_ftmin" in result.columns:
            vz_sel = result["fdm_vz_sel_ftmin"].to_numpy()
            only_alt = stabilised_alt & np.isnan(vz_sel)
        else:
            only_alt = stabilised_alt
        if only_alt.sum() > 0:
            np.testing.assert_allclose(target[only_alt], 0.0, atol=1e-10)


# === Merged from test_tas_target.py: unified TAS target from segments ===


def _tt_make_flight(  # noqa: PLR0913
    n: int = 300,
    *,
    mach_cruise: float = 0.78,
    cas_climb: float = 250.0,
    cas_descent: float = 250.0,
    tas_cruise_kt: float = 450.0,
    include_mach: bool = True,
    include_cas: bool = True,
    include_tas: bool = True,
    alt_nan_indices: list[int] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a synthetic 3-phase flight (climb / cruise / descent)."""
    rng = np.random.default_rng(seed)
    third = n // 3

    alt = np.concatenate(
        [
            np.linspace(0, 35_000, third),
            np.full(n - 2 * third, 35_000),
            np.linspace(35_000, 0, third),
        ]
    )
    if alt_nan_indices:
        alt = alt.copy()
        for idx in alt_nan_indices:
            alt[idx] = np.nan

    cols: dict[str, np.ndarray] = {"raw_alt_ft": alt}

    if include_mach:
        mach = np.concatenate(
            [
                np.linspace(0.3, mach_cruise, third),
                np.full(n - 2 * third, mach_cruise) + rng.normal(0, 0.0001, n - 2 * third),
                np.linspace(mach_cruise, 0.3, third),
            ]
        )
        cols["bds_mach_clean"] = mach

    if include_cas:
        cas = np.concatenate(
            [
                np.full(third, cas_climb) + rng.normal(0, 0.1, third),
                np.linspace(cas_climb, 280, n - 2 * third),
                np.full(third, cas_descent) + rng.normal(0, 0.1, third),
            ]
        )
        cols["bds_ias_kt_clean"] = cas

    if include_tas:
        tas = np.concatenate(
            [
                np.linspace(200, tas_cruise_kt, third),
                np.full(n - 2 * third, tas_cruise_kt) + rng.normal(0, 0.2, n - 2 * third),
                np.linspace(tas_cruise_kt, 200, third),
            ]
        )
        cols["fdm_tas_from_cas_kt"] = tas

    return pl.DataFrame(cols)


def _tt_config(
    *,
    include_mach: bool = True,
    include_cas: bool = True,
    include_tas: bool = True,
) -> dict[str, dict[str, object]]:
    cfg: dict[str, dict[str, object]] = {}
    if include_mach:
        cfg["mach"] = {
            "tol": 0.0005,
            "min_len": 30,
            "alt_threshold": 15_000,
            "smooth_window": 10,
            "use_alt": True,
        }
    if include_cas:
        cfg["cas"] = {
            "tol": 0.75,
            "min_len": 20,
            "use_alt": False,
            "smooth_window": 10,
            "smooth_method": "savgol",
        }
    if include_tas:
        cfg["tas"] = {
            "tol": 0.5,
            "min_len": 20,
            "use_alt": False,
            "smooth_window": 10,
            "smooth_method": "savgol",
        }
    return cfg


class TestTasTargetNoNan:
    """fdm_tas_target_kt is non-NaN inside Mach/CAS coverage (AXM-1625 AC5/AC6)."""

    def test_tas_target_non_nan_inside_coverage(self) -> None:
        df = _tt_make_flight()
        result = build_selected_params(df, _tt_config())

        assert "fdm_tas_target_kt" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        mach_sel = result["fdm_mach_sel"].to_numpy()
        cas_sel = result["fdm_cas_sel_kt"].to_numpy()
        coverage = ~np.isnan(mach_sel) | ~np.isnan(cas_sel)
        assert np.all(~np.isnan(target[coverage]))

    def test_tas_target_known_mask_matches_target(self) -> None:
        df = _tt_make_flight()
        result = build_selected_params(df, _tt_config())

        assert "fdm_tas_target_known" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        known = result["fdm_tas_target_known"].to_numpy().astype(bool)
        np.testing.assert_array_equal(known, ~np.isnan(target))


class TestTasTargetMachPriority:
    """When Mach and CAS segments overlap, Mach-derived TAS wins."""

    def test_tas_target_mach_priority(self) -> None:
        df = _tt_make_flight()
        result = build_selected_params(df, _tt_config())

        n = len(result)
        third = n // 3
        cruise_start = third
        cruise_end = 2 * third

        mach_sel = result["fdm_mach_sel"]
        tas_target = result["fdm_tas_target_kt"]

        for i in range(cruise_start, cruise_end):
            mach_val = mach_sel[i]
            if mach_val is not None and not np.isnan(mach_val):
                target_val = tas_target[i]
                assert target_val is not None and not np.isnan(target_val)


class TestSiConversion:
    """fdm_tas_target_ms approx fdm_tas_target_kt x 0.514444."""

    def test_si_conversion(self) -> None:
        from node_fdm_data.conversions import kt_to_ms

        df = _tt_make_flight()
        result = build_selected_params(df, _tt_config())

        assert "fdm_tas_target_kt" in result.columns

        converted = result.select(
            kt_to_ms("fdm_tas_target_kt").alias("manual_ms"),
        )
        manual_ms = converted["manual_ms"]

        expected = result["fdm_tas_target_kt"].to_numpy() * 0.514444
        np.testing.assert_allclose(
            manual_ms.to_numpy(),
            expected,
            rtol=1e-4,
        )


class TestNoMachSegments:
    """No Mach segments — only CAS/TAS segments contribute; gaps stay NaN."""

    def test_no_mach_segments_yields_nan_gaps(self) -> None:
        df = _tt_make_flight(include_mach=False)
        cfg = _tt_config(include_mach=False)
        result = build_selected_params(df, cfg)

        assert "fdm_tas_target_kt" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        cas_sel = result["fdm_cas_sel_kt"].to_numpy()
        tas_sel_present = "fdm_tas_sel_kt" in result.columns
        tas_sel = (
            result["fdm_tas_sel_kt"].to_numpy()
            if tas_sel_present
            else np.full(len(target), np.nan)
        )
        coverage = ~np.isnan(cas_sel) | ~np.isnan(tas_sel)
        assert np.all(~np.isnan(target[coverage]))
        known = result["fdm_tas_target_known"].to_numpy().astype(bool)
        np.testing.assert_array_equal(known, ~np.isnan(target))


class TestNoSegmentsAtAll:
    """No segments detected — target stays NaN end-to-end and known is False."""

    def test_no_segments_target_all_nan(self) -> None:
        rng = np.random.default_rng(99)
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.linspace(0, 10_000, n),
                "fdm_tas_from_cas_kt": rng.uniform(200, 400, n),
            }
        )
        cfg: dict[str, dict[str, object]] = {}
        result = build_selected_params(df, cfg)

        assert "fdm_tas_target_kt" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        assert np.all(np.isnan(target))
        known = result["fdm_tas_target_known"].to_numpy().astype(bool)
        assert not known.any()


class TestNanInAltitude:
    """NaN in altitude — target stays NaN where altitude is NaN (no backfill)."""

    def test_nan_altitude_yields_nan_target(self) -> None:
        nan_indices = [50, 51, 52, 150, 151]
        df = _tt_make_flight(alt_nan_indices=nan_indices)
        result = build_selected_params(df, _tt_config())

        assert "fdm_tas_target_kt" in result.columns
        target = result["fdm_tas_target_kt"].to_numpy()
        known = result["fdm_tas_target_known"].to_numpy().astype(bool)
        np.testing.assert_array_equal(known, ~np.isnan(target))


class TestDetectAltHoldFromVz:
    """AXM-1687 — bilateral vz-based altitude plateau detection."""

    @staticmethod
    def _three_plateau_signal(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Synthetic 3-plateau alt + matching vz (ft/min, dt=4 s)."""
        rng = np.random.default_rng(seed)
        plateaus = [10000.0, 20000.0, 35000.0]
        parts: list[np.ndarray] = [np.full(60, plateaus[0])]
        for prev, nxt in pairwise(plateaus):
            parts.append(np.linspace(prev, nxt, 40))
            parts.append(np.full(60, nxt))
        alt = np.concatenate(parts)
        alt = np.round(alt / 25.0) * 25.0
        alt = alt + rng.uniform(-50.0, 50.0, alt.shape)
        dt_s = 4.0
        vz_ftmin = np.gradient(alt, dt_s) * 60.0
        return vz_ftmin, alt

    def test_detect_alt_hold_from_vz_three_plateaus(self) -> None:
        """AC1, AC2 — 3 segments, each var_mean within ±100 ft."""
        vz, alt = self._three_plateau_signal()
        segs = detect_alt_hold_from_vz(
            vz,
            alt,
            sigma_s=6.0,
            sigma_r=350.0,
            n_passes=2,
            tol_ftmin=400.0,
            min_len=6,
        )
        assert len(segs) == 3
        truths = [10000.0, 20000.0, 35000.0]
        for seg, truth in zip(segs, truths, strict=True):
            assert abs(seg["var_mean"] - truth) < 100.0
            assert "start_idx" in seg and "end_idx" in seg

    def test_detect_alt_hold_from_vz_all_nan(self) -> None:
        """AC3 — fully-NaN input returns [] without raising."""
        vz = np.full(200, np.nan)
        alt = np.full(200, np.nan)
        segs = detect_alt_hold_from_vz(
            vz,
            alt,
            sigma_s=6.0,
            sigma_r=350.0,
            n_passes=2,
            tol_ftmin=400.0,
            min_len=6,
        )
        assert segs == []

    def test_detect_alt_hold_from_vz_partial_nan(self) -> None:
        """AC3 — 5% NaN in vz is interpolated; plateaus still detected."""
        rng = np.random.default_rng(1)
        vz, alt = self._three_plateau_signal(seed=1)
        idx = rng.choice(vz.size, size=max(1, vz.size // 20), replace=False)
        vz_nan = vz.copy()
        vz_nan[idx] = np.nan
        segs = detect_alt_hold_from_vz(
            vz_nan,
            alt,
            sigma_s=6.0,
            sigma_r=350.0,
            n_passes=2,
            tol_ftmin=400.0,
            min_len=6,
        )
        assert len(segs) >= 1

    def test_detect_alt_hold_from_vz_too_short(self) -> None:
        """AC1 — runs shorter than min_len are discarded."""
        vz = np.zeros(5)
        alt = np.full(5, 30000.0)
        segs = detect_alt_hold_from_vz(
            vz,
            alt,
            sigma_s=6.0,
            sigma_r=350.0,
            n_passes=2,
            tol_ftmin=400.0,
            min_len=6,
        )
        assert segs == []


class TestAltFilterConfigBilateral:
    """AXM-1687 — AltFilterConfig mode + bilateral params."""

    def test_alt_filter_config_default_mode_bilateral(self) -> None:
        """AC4 — defaults: bilateral_vz mode + script-calibrated params."""
        cfg = AltFilterConfig()
        assert cfg.mode == "bilateral_vz"
        assert cfg.sigma_s == 6.0
        assert cfg.sigma_r == 350.0
        assert cfg.n_passes == 2
        assert cfg.tol_ftmin == 400.0
        assert cfg.min_len == 6
        assert cfg.tol == 25
        assert cfg.use_alt is False
        assert cfg.min_abs_value == 25
        assert cfg.smooth_window == 5
        assert cfg.smooth_method == "savgol"

    def test_alt_filter_config_savgol_mode_legacy_defaults(self) -> None:
        """AC4 — savgol_alt mode constructs and exposes legacy fields."""
        cfg = AltFilterConfig(mode="savgol_alt")
        assert cfg.mode == "savgol_alt"
        assert cfg.tol == 25
        assert cfg.use_alt is False
        assert cfg.min_abs_value == 25
        assert cfg.smooth_window == 5
        assert cfg.smooth_method == "savgol"


def _alt_sel_fixture_df(seed: int = 0) -> pl.DataFrame:
    """Constant-cruise polars DF (n=200) with required columns."""
    rng = np.random.default_rng(seed)
    n = 200
    alt = np.full(n, 35000.0) + rng.normal(0, 5.0, n)
    vz = rng.normal(0, 5.0, n)
    mach = np.full(n, 0.78) + rng.normal(0, 0.0001, n)
    cas = np.full(n, 280.0) + rng.normal(0, 0.1, n)
    tas = np.full(n, 460.0)
    gamma = np.zeros(n)
    return pl.DataFrame(
        {
            "raw_alt_ft": alt,
            "raw_vz_ftmin": vz,
            "bds_mach_clean": mach,
            "bds_ias_kt_clean": cas,
            "fdm_tas_from_cas_kt": tas,
            "fdm_gamma_rad": gamma,
        }
    )


class TestDetectAltSelDispatch:
    """AXM-1687 — _detect_alt_sel dispatches on cfg['mode']."""

    def test_detect_alt_sel_bilateral_mode_writes_alt_sel(
        self, segment_config_full: dict[str, Any]
    ) -> None:
        """AC5, AC6 — bilateral_vz mode populates fdm_alt_sel_ft."""
        df = _alt_sel_fixture_df()
        cfg = dict(segment_config_full)
        cfg["alt"] = {
            "mode": "bilateral_vz",
            "sigma_s": 6.0,
            "sigma_r": 350.0,
            "n_passes": 2,
            "tol_ftmin": 400.0,
            "min_len": 6,
        }
        result = build_selected_params(df, cfg)
        assert "fdm_alt_sel_ft" in result.columns
        non_nan = (~np.isnan(result["fdm_alt_sel_ft"].to_numpy())).sum()
        assert non_nan >= 100

    def test_detect_alt_sel_savgol_backcompat(self, segment_config_full: dict[str, Any]) -> None:
        """AC5 — savgol_alt mode coverage matches direct detect_constant_segments snapshot."""
        df = _alt_sel_fixture_df()
        legacy_alt_cfg = {
            "mode": "savgol_alt",
            "tol": 25,
            "min_len": 5,
            "use_alt": False,
            "min_abs_value": 25,
            "smooth_window": 5,
            "smooth_method": "savgol",
        }
        cfg = dict(segment_config_full)
        cfg["alt"] = legacy_alt_cfg
        result = build_selected_params(df, cfg)
        assert "fdm_alt_sel_ft" in result.columns

        alt_arr = df["raw_alt_ft"].to_numpy()
        expected_segs = detect_constant_segments(
            alt_arr,
            tol=25,
            min_len=5,
            use_alt=False,
            min_abs_value=25,
            smooth_window=5,
            smooth_method="savgol",
        )
        expected_df = add_segment_column(df, expected_segs, "fdm_alt_sel_ft")
        expected = expected_df["fdm_alt_sel_ft"].to_numpy()
        actual = result["fdm_alt_sel_ft"].to_numpy()
        assert (~np.isnan(actual)).sum() == (~np.isnan(expected)).sum()
