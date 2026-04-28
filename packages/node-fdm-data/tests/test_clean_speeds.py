"""Tests for speed cleaning (Hampel + ERA5 fill)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds, clean_speeds


class TestHampel:
    """Hampel filter behaviour."""

    def test_hampel_removes_isolated_spike(self) -> None:
        """A single 10-sigma spike is replaced by NaN; neighbours unchanged."""
        n = 50
        values = np.full(n, 100.0)
        values[25] = 1000.0
        era = np.full(n, np.nan)

        result = clean_speeds(
            values,
            era,
            window=7,
            k=3.0,
            era_dev_max=None,
            n_passes=1,
            interp_max_gap=0,
        )

        assert math.isnan(result[25])
        assert result[24] == pytest.approx(100.0)
        assert result[26] == pytest.approx(100.0)

    def test_hampel_keeps_clean_signal(self) -> None:
        """A smooth sinusoid passes through unchanged (no false positives)."""
        n = 100
        t = np.arange(n)
        values = 100.0 + 5.0 * np.sin(t * 0.1)
        era = np.full(n, np.nan)

        result = clean_speeds(
            values.copy(),
            era,
            window=7,
            k=3.0,
            era_dev_max=None,
            n_passes=3,
            interp_max_gap=0,
        )

        np.testing.assert_allclose(result, values, equal_nan=True)

    def test_hampel_multi_pass_removes_cluster(self) -> None:
        """A 3-point outlier cluster is fully removed via multi-pass Hampel."""
        n = 50
        values = np.full(n, 100.0)
        values[24] = 1000.0
        values[25] = 1000.0
        values[26] = 1000.0
        era = np.full(n, np.nan)

        result = clean_speeds(
            values,
            era,
            window=7,
            k=3.0,
            era_dev_max=None,
            n_passes=3,
            interp_max_gap=0,
        )

        assert math.isnan(result[24])
        assert math.isnan(result[25])
        assert math.isnan(result[26])


class TestEraDevCap:
    """ERA-deviation cap behaviour."""

    def test_era_dev_cap_flags_systematic_bias(self) -> None:
        """Smooth +0.10 bias vs ERA with era_dev_max=0.05 is flagged NaN."""
        n = 50
        era = np.full(n, 0.80)
        values = era + 0.10

        result = clean_speeds(
            values.copy(),
            era,
            window=7,
            k=3.0,
            era_dev_max=0.05,
            n_passes=1,
            interp_max_gap=0,
        )

        assert all(math.isnan(v) for v in result)

    def test_era_dev_cap_disabled_when_none(self) -> None:
        """era_dev_max=None disables the cap; shifted region preserved."""
        n = 50
        era = np.full(n, 0.80)
        values = era + 0.10

        result = clean_speeds(
            values.copy(),
            era,
            window=7,
            k=3.0,
            era_dev_max=None,
            n_passes=1,
            interp_max_gap=0,
        )

        np.testing.assert_allclose(result, values)


class TestInterpolation:
    """Linear interpolation of short gaps."""

    def test_interpolate_short_gap(self) -> None:
        """A NaN run of length 3 between valid anchors is filled linearly."""
        values = np.array([10.0, 11.0, np.nan, np.nan, np.nan, 15.0, 16.0])
        era = np.full_like(values, np.nan)

        result = clean_speeds(
            values,
            era,
            window=3,
            k=10.0,
            era_dev_max=None,
            n_passes=1,
            interp_max_gap=10,
        )

        assert result[2] == pytest.approx(12.0)
        assert result[3] == pytest.approx(13.0)
        assert result[4] == pytest.approx(14.0)

    def test_interpolate_leaves_long_gap(self) -> None:
        """A NaN run of length 20 with interp_max_gap=10 stays NaN."""
        values = np.full(30, np.nan)
        values[0] = 1.0
        values[29] = 100.0
        era = np.full(30, np.nan)

        result = clean_speeds(
            values,
            era,
            window=3,
            k=10.0,
            era_dev_max=None,
            n_passes=1,
            interp_max_gap=10,
        )

        for i in range(1, 29):
            assert math.isnan(result[i])

    def test_interpolate_gap_at_edges(self) -> None:
        """NaN run with no left or right anchor stays NaN (no extrapolation)."""
        n = 20
        values = np.full(n, np.nan)
        values[8:12] = 100.0
        era = np.full(n, np.nan)

        result = clean_speeds(
            values,
            era,
            window=3,
            k=10.0,
            era_dev_max=None,
            n_passes=1,
            interp_max_gap=20,
        )

        for i in range(0, 8):
            assert math.isnan(result[i])
        for i in range(12, n):
            assert math.isnan(result[i])


class TestCleanSpeedsAllNan:
    """Edge case: all-NaN input."""

    def test_clean_speeds_all_nan_input(self) -> None:
        """All-NaN values + all-NaN era returns all-NaN, no error raised."""
        n = 30
        values = np.full(n, np.nan)
        era = np.full(n, np.nan)

        result = clean_speeds(
            values,
            era,
            window=7,
            k=3.0,
            era_dev_max=0.05,
            n_passes=3,
            interp_max_gap=10,
        )

        assert all(math.isnan(v) for v in result)


class TestCleanBdsSpeeds:
    """Polars wrapper clean_bds_speeds."""

    @staticmethod
    def _make_df() -> pl.DataFrame:
        n = 50
        bds_mach = np.full(n, 0.80)
        bds_ias_kt = np.full(n, 250.0)
        bds_tas_kt = np.full(n, 230.0)
        bds_mach[25] = 1.5
        bds_ias_kt[25] = 800.0
        bds_tas_kt[25] = 800.0

        return pl.DataFrame(
            {
                "bds_mach": bds_mach,
                "bds_ias_kt": bds_ias_kt,
                "bds_tas_kt": bds_tas_kt,
                "era_mach": np.full(n, 0.80),
                "era_tas_kt": np.full(n, 460.0),
                "era_cas_kt": np.full(n, 250.0),
            }
        )

    def test_clean_bds_speeds_creates_clean_columns(self) -> None:
        df = self._make_df()
        result = clean_bds_speeds(df)
        assert "bds_mach_clean" in result.columns
        assert "bds_ias_kt_clean" in result.columns
        assert "bds_tas_kt_clean" in result.columns

    def test_clean_bds_speeds_preserves_raw(self) -> None:
        df = self._make_df()
        raw_mach = df["bds_mach"].to_list()
        raw_ias = df["bds_ias_kt"].to_list()
        raw_tas = df["bds_tas_kt"].to_list()

        result = clean_bds_speeds(df)

        assert result["bds_mach"].to_list() == raw_mach
        assert result["bds_ias_kt"].to_list() == raw_ias
        assert result["bds_tas_kt"].to_list() == raw_tas

    def test_clean_bds_speeds_idempotent(self) -> None:
        df = self._make_df()
        first = clean_bds_speeds(df)
        second = clean_bds_speeds(first)

        for col in ("bds_mach_clean", "bds_ias_kt_clean", "bds_tas_kt_clean"):
            a = first[col].to_list()
            b = second[col].to_list()
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=False):
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    assert y is None or (isinstance(y, float) and math.isnan(y))
                else:
                    assert x == pytest.approx(y)

    def test_clean_bds_speeds_missing_input_columns(self) -> None:
        """When bds_mach is absent, clean function must not raise."""
        df = pl.DataFrame(
            {
                "bds_ias_kt": [250.0, 251.0, 252.0],
                "bds_tas_kt": [230.0, 231.0, 232.0],
                "era_tas_kt": [460.0, 461.0, 462.0],
                "era_cas_kt": [250.0, 251.0, 252.0],
            }
        )
        result = clean_bds_speeds(df)
        if "bds_mach_clean" in result.columns:
            assert result["bds_mach_clean"].null_count() == result.height
