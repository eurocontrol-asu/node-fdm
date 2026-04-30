"""Tests for segment-based selected parameter estimation."""

from __future__ import annotations

from typing import Any

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
                "raw_alt_ft": alt,
                "bds_mach_clean": mach + rng.normal(0, 0.0001, n),
                "bds_ias_kt_clean": cas + rng.normal(0, 0.1, n),
                "raw_vz_ftmin": vz + rng.normal(0, 1, n),
                "fdm_gamma_rad": gamma,
                "bds_mcp_sel_alt_ft": np.where(alt > 15000, 35000.0, np.nan),
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
                "bds_tas_from_cas_kt": np.full(n, 450.0),
                "raw_gs_kt": np.full(n, 430.0),
                "raw_vz_ftmin": np.full(n, 0.0),
                "bds_ias_kt_clean": np.full(n, 280.0),
                "bds_mcp_sel_alt_ft": np.full(n, 36000.0),
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
                "bds_mcp_sel_alt_ft": np.where(
                    (np.arange(n) > 50) & (np.arange(n) < 250),
                    36000.0,
                    np.nan,
                ),
                "bds_fms_sel_alt_ft": np.where(
                    (np.arange(n) > 30) & (np.arange(n) < 270),
                    37000.0,
                    np.nan,
                ),
            }
        )

    @staticmethod
    def _full_config() -> dict[str, Any]:
        """Config with all parameter sections."""
        return {
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

    def test_segments_fdm_prefix(self) -> None:
        """All segment columns use the fdm_ prefix."""
        df = self._make_flight_df()
        result = build_selected_params(df, self._full_config())
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
                "bds_mcp_sel_alt_ft": np.where(
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
                "bds_fms_sel_alt_ft": np.where(
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
        """When bds_mcp_sel_alt_ft is all NaN, alt segments detect on raw_alt_ft.

        raw_alt_ft constant at 35000 → level segment detected → fdm_alt_sel_ft
        is NOT all NaN.  MCP/FMS backfill columns remain all NaN.
        """
        n = 50
        df = pl.DataFrame(
            {
                "raw_alt_ft": np.full(n, 35000.0),
                "bds_mach_clean": np.full(n, 0.78),
                "bds_mcp_sel_alt_ft": np.full(n, np.nan),
                "bds_fms_sel_alt_ft": np.full(n, np.nan),
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
                "bds_tas_from_cas_kt": tas,
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
                "bds_tas_from_cas_kt": tas,
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
                "bds_tas_from_cas_kt": tas,
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
                "bds_tas_from_cas_kt": np.full(n, 460.0),
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
                "bds_tas_from_cas_kt": np.full(n, np.nan),
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
                "bds_tas_from_cas_kt": np.full(n, 460.0),
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
