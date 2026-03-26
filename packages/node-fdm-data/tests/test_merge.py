"""Tests for BDS + ERA5 merge module."""

from __future__ import annotations

import polars as pl
import pytest

from node_fdm_data.preprocessing.merge import merge_bds_era5


class TestMergeBdsPreferred:
    """BDS values should be preferred over ERA5."""

    def test_bds_present_takes_precedence(self) -> None:
        """When BDS is present, ERA5 is ignored."""
        df = pl.DataFrame(
            {
                "bds_tas_kt": [250.0],
                "era_tas_kt": [245.0],
                "bds_ias_kt": [200.0],
                "era_cas_kt": [195.0],
                "bds_mach": [0.82],
                "era_mach": [0.80],
            }
        )
        result = merge_bds_era5(df)
        assert result["ekf_input_tas_kt"][0] == pytest.approx(250.0)
        assert result["ekf_input_cas_kt"][0] == pytest.approx(200.0)
        assert result["ekf_input_mach"][0] == pytest.approx(0.82)


class TestMergeEra5Fallback:
    """ERA5 fills gaps when BDS is null."""

    def test_bds_null_uses_era5(self) -> None:
        """When BDS is null, ERA5 is used."""
        df = pl.DataFrame(
            {
                "bds_tas_kt": [None],
                "era_tas_kt": [245.0],
                "bds_ias_kt": [None],
                "era_cas_kt": [195.0],
                "bds_mach": [None],
                "era_mach": [0.80],
            },
            schema={
                "bds_tas_kt": pl.Float64,
                "era_tas_kt": pl.Float64,
                "bds_ias_kt": pl.Float64,
                "era_cas_kt": pl.Float64,
                "bds_mach": pl.Float64,
                "era_mach": pl.Float64,
            },
        )
        result = merge_bds_era5(df)
        assert result["ekf_input_tas_kt"][0] == pytest.approx(245.0)
        assert result["ekf_input_cas_kt"][0] == pytest.approx(195.0)
        assert result["ekf_input_mach"][0] == pytest.approx(0.80)


class TestMergeBothNull:
    """Both null → output is null."""

    def test_both_null_remains_null(self) -> None:
        df = pl.DataFrame(
            {
                "bds_tas_kt": [None],
                "era_tas_kt": [None],
                "bds_ias_kt": [None],
                "era_cas_kt": [None],
                "bds_mach": [None],
                "era_mach": [None],
            },
            schema={
                "bds_tas_kt": pl.Float64,
                "era_tas_kt": pl.Float64,
                "bds_ias_kt": pl.Float64,
                "era_cas_kt": pl.Float64,
                "bds_mach": pl.Float64,
                "era_mach": pl.Float64,
            },
        )
        result = merge_bds_era5(df)
        assert result["ekf_input_tas_kt"][0] is None
        assert result["ekf_input_cas_kt"][0] is None
        assert result["ekf_input_mach"][0] is None


class TestMergeMixedAvailability:
    """Mixed: some rows BDS, some ERA5."""

    def test_mixed_rows(self) -> None:
        df = pl.DataFrame(
            {
                "bds_tas_kt": [250.0, None, 260.0],
                "era_tas_kt": [245.0, 248.0, 255.0],
            }
        )
        result = merge_bds_era5(df)
        assert result["ekf_input_tas_kt"].to_list() == pytest.approx([250.0, 248.0, 260.0])


class TestMergeMissingColumns:
    """Graceful handling of missing BDS or ERA5 columns."""

    def test_only_bds(self) -> None:
        df = pl.DataFrame({"bds_tas_kt": [250.0]})
        result = merge_bds_era5(df)
        assert result["ekf_input_tas_kt"][0] == pytest.approx(250.0)

    def test_only_era5(self) -> None:
        df = pl.DataFrame({"era_tas_kt": [245.0]})
        result = merge_bds_era5(df)
        assert result["ekf_input_tas_kt"][0] == pytest.approx(245.0)

    def test_neither_column(self) -> None:
        df = pl.DataFrame({"raw_alt_ft": [35000.0]})
        result = merge_bds_era5(df)
        assert result["ekf_input_tas_kt"][0] is None


class TestMergeIdempotent:
    """Running merge twice gives same result."""

    def test_idempotency(self) -> None:
        df = pl.DataFrame(
            {
                "bds_tas_kt": [250.0],
                "era_tas_kt": [245.0],
                "bds_ias_kt": [200.0],
                "era_cas_kt": [195.0],
                "bds_mach": [0.82],
                "era_mach": [0.80],
            }
        )
        result1 = merge_bds_era5(df)
        result2 = merge_bds_era5(result1)
        assert result2["ekf_input_tas_kt"].to_list() == result1["ekf_input_tas_kt"].to_list()
