"""Tests for node_fdm_data.preprocessing.convert — SI conversion & derivatives."""

from __future__ import annotations

import polars as pl
import pytest

from node_fdm_data.preprocessing.convert import (
    SI_CONVERSIONS,
    SI_DERIVATIVES,
    compute_derivatives,
    convert_si,
)


class TestConvertSI:
    """SI unit conversion tests (étape 6)."""

    @pytest.fixture()
    def pipeline_df(self) -> pl.DataFrame:
        """DataFrame with v3 pipeline column names from étapes 0-5."""
        return pl.DataFrame(
            {
                "raw_alt_ft": [30000.0, 35000.0],
                "bds_mcp_sel_alt_ft": [31000.0, 36000.0],
                "era_tas_kt": [250.0, 450.0],
                "bds_ias_kt": [200.0, 280.0],
                "raw_gs_kt": [240.0, 430.0],
                "fdm_long_wind_kt": [10.0, 20.0],
                "raw_vz_ftmin": [1000.0, 0.0],
                "fdm_adep_dist_nm": [0.0, 50.0],
                "fdm_ades_dist_nm": [200.0, 150.0],
            }
        )

    def test_convert_ft_to_m(self, pipeline_df: pl.DataFrame) -> None:
        """raw_alt_ft -> raw_alt_m via ft_to_m (x 0.3048)."""
        result = convert_si(pipeline_df)
        assert "raw_alt_m" in result.columns
        assert result["raw_alt_m"][0] == pytest.approx(30000.0 * 0.3048)

    def test_convert_kt_to_ms(self, pipeline_df: pl.DataFrame) -> None:
        """era_tas_kt -> era_tas_ms via kt_to_ms (x 0.514444)."""
        result = convert_si(pipeline_df)
        assert "era_tas_ms" in result.columns
        assert result["era_tas_ms"][0] == pytest.approx(250.0 * 0.514444, rel=1e-4)

    def test_convert_preserves_source(self, pipeline_df: pl.DataFrame) -> None:
        """Source columns remain after conversion (additive, no replace)."""
        original_cols = set(pipeline_df.columns)
        result = convert_si(pipeline_df)
        # All original columns still present
        for col in original_cols:
            assert col in result.columns, f"Source column {col} was removed"

    def test_convert_all_expected_targets(self, pipeline_df: pl.DataFrame) -> None:
        """All SI_CONVERSIONS targets are produced when source columns exist."""
        result = convert_si(pipeline_df)
        for src, _, tgt in SI_CONVERSIONS:
            if src in pipeline_df.columns:
                assert tgt in result.columns, f"Missing target column: {tgt}"

    def test_convert_ftmin_to_ms(self, pipeline_df: pl.DataFrame) -> None:
        """raw_vz_ftmin -> raw_vz_ms via ftmin_to_ms (x 0.00508)."""
        result = convert_si(pipeline_df)
        assert "raw_vz_ms" in result.columns
        assert result["raw_vz_ms"][0] == pytest.approx(1000.0 * 0.3048 / 60.0, rel=1e-4)

    def test_convert_nm_to_m(self, pipeline_df: pl.DataFrame) -> None:
        """fdm_adep_dist_nm -> fdm_adep_dist_m via nm_to_m (x 1852)."""
        result = convert_si(pipeline_df)
        assert "fdm_adep_dist_m" in result.columns
        assert result["fdm_adep_dist_m"][1] == pytest.approx(50.0 * 1852.0)

    def test_convert_missing_source_skipped(self) -> None:
        """Conversion is skipped when source column does not exist."""
        df = pl.DataFrame({"unrelated_col": [1.0, 2.0]})
        result = convert_si(df)
        assert result.columns == ["unrelated_col"]


class TestComputeDerivatives:
    """Temporal derivative tests (étape 7)."""

    @pytest.fixture()
    def two_flights_df(self) -> pl.DataFrame:
        """Two flights with SI columns and meta_flight_id."""
        return pl.DataFrame(
            {
                "meta_flight_id": ["A", "A", "A", "B", "B", "B"],
                "raw_alt_m": [0.0, 100.0, 300.0, 1000.0, 1200.0, 1500.0],
                "fdm_gamma_rad": [0.0, 0.01, 0.03, 0.1, 0.12, 0.15],
                "era_tas_ms": [100.0, 110.0, 130.0, 200.0, 210.0, 230.0],
            }
        )

    def test_derivatives_per_flight(self, two_flights_df: pl.DataFrame) -> None:
        """Derivatives restart for each meta_flight_id (no cross-flight bleed)."""
        result = compute_derivatives(two_flights_df)

        # Flight A: diff(raw_alt_m) = [null, 100, 200] → backward_fill → [100, 100, 200]
        a = result.filter(pl.col("meta_flight_id") == "A")
        assert a["fdm_d_vz_ms"][0] == pytest.approx(100.0)  # backward fill
        assert a["fdm_d_vz_ms"][1] == pytest.approx(100.0)
        assert a["fdm_d_vz_ms"][2] == pytest.approx(200.0)

        # Flight B: diff(raw_alt_m) = [null, 200, 300] → backward_fill → [200, 200, 300]
        b = result.filter(pl.col("meta_flight_id") == "B")
        assert b["fdm_d_vz_ms"][0] == pytest.approx(200.0)  # backward fill
        assert b["fdm_d_vz_ms"][1] == pytest.approx(200.0)
        assert b["fdm_d_vz_ms"][2] == pytest.approx(300.0)

    def test_derivatives_first_value(self, two_flights_df: pl.DataFrame) -> None:
        """First value of each derivative uses backward_fill + fill_null(0.0)."""
        result = compute_derivatives(two_flights_df)
        # First row should NOT be 0.0 — it's backward_fill of second row
        for col in ("fdm_d_vz_ms", "fdm_d_gamma_rads", "fdm_d_tas_ms"):
            assert col in result.columns
            # First value of flight A = second value (backward fill)
            a = result.filter(pl.col("meta_flight_id") == "A")
            assert a[col][0] == pytest.approx(a[col][1])

    def test_derivatives_all_targets(self, two_flights_df: pl.DataFrame) -> None:
        """All SI_DERIVATIVES targets are produced."""
        result = compute_derivatives(two_flights_df)
        for _, tgt in SI_DERIVATIVES:
            assert tgt in result.columns, f"Missing derivative: {tgt}"

    def test_derivatives_missing_source_skipped(self) -> None:
        """Derivative is skipped when source column does not exist."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["A", "A"],
                "unrelated_col": [1.0, 2.0],
            }
        )
        result = compute_derivatives(df)
        assert "fdm_d_vz_ms" not in result.columns

    def test_single_row_flight(self) -> None:
        """A single-row flight produces 0.0 derivatives (fill_null fallback)."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["A"],
                "raw_alt_m": [1000.0],
                "fdm_gamma_rad": [0.05],
                "era_tas_ms": [200.0],
            }
        )
        result = compute_derivatives(df)
        # diff of single row = null → backward_fill still null → fill_null(0.0)
        assert result["fdm_d_vz_ms"][0] == pytest.approx(0.0)
        assert result["fdm_d_gamma_rads"][0] == pytest.approx(0.0)
        assert result["fdm_d_tas_ms"][0] == pytest.approx(0.0)
