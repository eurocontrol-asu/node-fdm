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

try:
    from node_fdm_data.preprocessing.convert import DERIVATIVE_BOUNDS
except ImportError:
    DERIVATIVE_BOUNDS = {"fdm_d_alt_ms": (-75.0, 75.0)}


class TestConvertSI:
    """SI unit conversion tests (étape 6)."""

    @pytest.fixture()
    def pipeline_df(self) -> pl.DataFrame:
        """DataFrame with v3 pipeline column names from étapes 0-5."""
        return pl.DataFrame(
            {
                "raw_alt_ft": [30000.0, 35000.0],
                "bds_mcp_alt_sel_ft": [31000.0, 36000.0],
                "fdm_tas_from_cas_kt": [250.0, 450.0],
                "bds_ias_kt_clean": [200.0, 280.0],
                "raw_gs_kt": [240.0, 430.0],
                "fdm_long_wind_kt": [10.0, 20.0],
                "raw_vz_ftmin": [1000.0, 0.0],
                "fdm_adep_dist_nm": [0.0, 50.0],
                "fdm_ades_dist_nm": [200.0, 150.0],
                "era_temp_K": [228.71, 218.81],
            }
        )

    @pytest.mark.parametrize(
        ("target_col", "row_idx", "expected", "rel"),
        [
            pytest.param("raw_alt_m", 0, 30000.0 * 0.3048, None, id="ft_to_m"),
            pytest.param("era_tas_ms", 0, 250.0 * 0.514444, 1e-4, id="kt_to_ms"),
            pytest.param("raw_vz_ms", 0, 1000.0 * 0.3048 / 60.0, 1e-4, id="ftmin_to_ms"),
            pytest.param("fdm_adep_dist_m", 1, 50.0 * 1852.0, None, id="nm_to_m"),
        ],
    )
    def test_convert_si_unit_target(
        self,
        pipeline_df: pl.DataFrame,
        target_col: str,
        row_idx: int,
        expected: float,
        rel: float | None,
    ) -> None:
        """convert_si produces SI target columns from each source unit family."""
        result = convert_si(pipeline_df)
        assert target_col in result.columns
        approx = pytest.approx(expected, rel=rel) if rel is not None else pytest.approx(expected)
        assert result[target_col][row_idx] == approx

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

    @pytest.mark.parametrize(
        ("src_col", "src_value", "tgt_col", "expected", "rel"),
        [
            pytest.param(
                "fdm_alt_target_ft",
                10000.0,
                "fdm_alt_target_m",
                3048.0,
                None,
                id="target_alt_ft_to_m",
            ),
            pytest.param(
                "fdm_cas_sel_kt",
                250.0,
                "fdm_cas_sel_ms",
                250.0 * 0.514444,
                1e-4,
                id="cas_sel_kt_to_ms",
            ),
            pytest.param(
                "fdm_vz_sel_ftmin",
                1000.0,
                "fdm_vz_sel_ms",
                1000.0 * 0.3048 / 60.0,
                1e-4,
                id="vz_sel_ftmin_to_ms",
            ),
        ],
    )
    def test_convert_si_selected_target(
        self,
        src_col: str,
        src_value: float,
        tgt_col: str,
        expected: float,
        rel: float | None,
    ) -> None:
        """convert_si emits SI target column for selected setpoint inputs."""
        df = pl.DataFrame({src_col: [src_value]})
        result = convert_si(df)
        assert tgt_col in result.columns
        approx = pytest.approx(expected, rel=rel) if rel is not None else pytest.approx(expected)
        assert result[tgt_col][0] == approx

    def test_convert_si_nan_preserved(self) -> None:
        """NaN values in source column are preserved in target column."""
        df = pl.DataFrame({"fdm_cas_sel_kt": [250.0, float("nan"), 300.0]})
        result = convert_si(df)
        assert result["fdm_cas_sel_ms"][1] is None or result["fdm_cas_sel_ms"].is_nan()[1]

    def test_convert_adds_alt_diff(self) -> None:
        """fdm_alt_diff_m = fdm_alt_target_m - raw_alt_m when both exist."""
        df = pl.DataFrame(
            {
                "fdm_alt_target_m": [10000.0, 12000.0],
                "raw_alt_m": [9500.0, 11800.0],
            }
        )
        result = convert_si(df)
        assert "fdm_alt_diff_m" in result.columns
        assert result["fdm_alt_diff_m"][0] == pytest.approx(500.0)
        assert result["fdm_alt_diff_m"][1] == pytest.approx(200.0)

    def test_convert_adds_tas_diff(self) -> None:
        """fdm_tas_diff_ms = fdm_tas_target_ms - era_tas_ms when both exist."""
        df = pl.DataFrame(
            {
                "fdm_tas_target_ms": [250.0, 300.0],
                "era_tas_ms": [240.0, 290.0],
            }
        )
        result = convert_si(df)
        assert "fdm_tas_diff_ms" in result.columns
        assert result["fdm_tas_diff_ms"][0] == pytest.approx(10.0)
        assert result["fdm_tas_diff_ms"][1] == pytest.approx(10.0)

    def test_convert_adds_gamma_diff(self) -> None:
        """fdm_gamma_diff_rad = fdm_gamma_target_rad - fdm_gamma_rad."""
        df = pl.DataFrame(
            {
                "fdm_gamma_target_rad": [0.05, 0.10],
                "fdm_gamma_rad": [0.03, 0.08],
            }
        )
        result = convert_si(df)
        assert "fdm_gamma_diff_rad" in result.columns
        assert result["fdm_gamma_diff_rad"][0] == pytest.approx(0.02)
        assert result["fdm_gamma_diff_rad"][1] == pytest.approx(0.02)

    def test_convert_gamma_diff_missing_source(self) -> None:
        """No crash and no gamma_diff column when fdm_gamma_target_rad is absent."""
        df = pl.DataFrame({"fdm_gamma_rad": [0.05, 0.10]})
        result = convert_si(df)
        assert "fdm_gamma_diff_rad" not in result.columns

    def test_convert_gamma_diff_nan_propagated(self) -> None:
        """NaN in fdm_gamma_target_rad yields 0 in fdm_gamma_diff_rad (AXM-809)."""
        df = pl.DataFrame(
            {
                "fdm_gamma_target_rad": [0.05, float("nan"), 0.10],
                "fdm_gamma_rad": [0.03, 0.04, 0.08],
            }
        )
        result = convert_si(df)
        assert "fdm_gamma_diff_rad" in result.columns
        assert result["fdm_gamma_diff_rad"][0] == pytest.approx(0.02)
        assert result["fdm_gamma_diff_rad"][1] == pytest.approx(0.0)
        assert result["fdm_gamma_diff_rad"][2] == pytest.approx(0.02)

    def test_convert_gamma_diff_both_zero(self) -> None:
        """Level flight: both gamma values zero yields gamma_diff = 0."""
        df = pl.DataFrame(
            {
                "fdm_gamma_target_rad": [0.0],
                "fdm_gamma_rad": [0.0],
            }
        )
        result = convert_si(df)
        assert result["fdm_gamma_diff_rad"][0] == pytest.approx(0.0)

    def test_convert_missing_source_cols(self) -> None:
        """No crash and no diff columns when target columns are absent."""
        df = pl.DataFrame({"raw_alt_m": [9500.0], "era_tas_ms": [240.0]})
        result = convert_si(df)
        assert "fdm_alt_diff_m" not in result.columns
        assert "fdm_tas_diff_ms" not in result.columns

    def test_convert_missing_source_skipped(self) -> None:
        """Conversion is skipped when source column does not exist."""
        df = pl.DataFrame({"unrelated_col": [1.0, 2.0]})
        result = convert_si(df)
        assert result.columns == ["unrelated_col"]


class TestConvertCasRealTemp:
    """fdm_cas_ms uses real ERA5 temperature so the round-trip closes."""

    def test_convert_fdm_cas_ms_uses_real_temp(self) -> None:
        """convert_si(fdm_cas_ms) matches tas_to_cas_real(tas, alt, era_temp_K)."""
        from node_fdm_data.physics.speed import tas_to_cas_real

        df = pl.DataFrame(
            {
                "raw_alt_m": [10000.0],
                "era_tas_ms": [240.0],
                "era_temp_K": [228.71],  # warmer than ISA at FL328
            }
        )
        result = convert_si(df)
        expected = float(tas_to_cas_real(240.0, 10000.0, 228.71))
        assert result["fdm_cas_ms"][0] == pytest.approx(expected, rel=1e-9)

    def test_convert_fdm_cas_ms_round_trip_closes(self) -> None:
        """tas_to_cas_real(cas_to_tas_real(IAS, ·, T), ·, T) == IAS at machine precision."""
        import numpy as np

        from node_fdm_data.physics.speed import cas_to_tas_real

        ias_kt = 280.0
        alt_m = 10000.0
        temp_k = 218.81
        ias_ms = ias_kt * 0.514444
        tas_arr = np.atleast_1d(cas_to_tas_real(np.array([ias_ms]), alt_m, temp_k))
        tas_ms = float(tas_arr[0])
        df = pl.DataFrame(
            {
                "raw_alt_m": [alt_m],
                "era_tas_ms": [tas_ms],
                "era_temp_K": [temp_k],
            }
        )
        result = convert_si(df)
        assert result["fdm_cas_ms"][0] == pytest.approx(ias_ms, rel=1e-9)

    def test_convert_fdm_cas_ms_isa_fallback_when_temp_missing(self) -> None:
        """When era_temp_K is absent, falls back to ISA-based tas_to_cas (back-compat)."""
        from node_fdm_data.physics.speed import tas_to_cas

        df = pl.DataFrame(
            {
                "raw_alt_m": [10000.0],
                "era_tas_ms": [240.0],
            }
        )
        result = convert_si(df)
        expected = float(tas_to_cas(240.0, 10000.0))
        assert result["fdm_cas_ms"][0] == pytest.approx(expected, rel=1e-9)


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
                # Lateral channel (Phase 2B): heading values stay in [0, 2π)
                # so the wrap-aware ``compute_derivatives`` branch is exercised.
                "fdm_heading_rad": [0.0, 0.05, 0.10, 1.0, 1.05, 1.10],
            }
        )

    def test_derivatives_divided_by_dt(self, two_flights_df: pl.DataFrame) -> None:
        """Derivatives are divided by dt to produce SI units."""
        result = compute_derivatives(two_flights_df, dt=4.0)
        a = result.filter(pl.col("meta_flight_id") == "A")
        # raw_alt_m = [0, 100, 300] → diff = [null, 100, 200] → /4 = [null, 25, 50]
        # backward_fill → [25, 25, 50]
        assert a["fdm_d_alt_ms"][0] == pytest.approx(25.0)
        assert a["fdm_d_alt_ms"][1] == pytest.approx(25.0)
        assert a["fdm_d_alt_ms"][2] == pytest.approx(50.0)

    def test_derivatives_clipped(self) -> None:
        """Aberrant derivatives are clipped to physical bounds."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["A", "A", "A"],
                "raw_alt_m": [0.0, 0.0, 50000.0],  # 50km jump → aberrant
                "fdm_gamma_rad": [0.0, 0.0, 0.0],
                "era_tas_ms": [200.0, 200.0, 200.0],
            }
        )
        result = compute_derivatives(df, dt=4.0)
        _, hi = DERIVATIVE_BOUNDS["fdm_d_alt_ms"]
        assert result["fdm_d_alt_ms"][2] == pytest.approx(hi)  # 75.0

    def test_derivatives_per_flight(self, two_flights_df: pl.DataFrame) -> None:
        """Derivatives restart for each meta_flight_id (no cross-flight bleed)."""
        result = compute_derivatives(two_flights_df, dt=4.0)

        # Flight A: diff(raw_alt_m) = [null, 100, 200] → /4 → [null, 25, 50]
        # backward_fill → [25, 25, 50]
        a = result.filter(pl.col("meta_flight_id") == "A")
        assert a["fdm_d_alt_ms"][0] == pytest.approx(25.0)  # backward fill
        assert a["fdm_d_alt_ms"][1] == pytest.approx(25.0)
        assert a["fdm_d_alt_ms"][2] == pytest.approx(50.0)

        # Flight B: diff(raw_alt_m) = [null, 200, 300] → /4 → [null, 50, 75]
        # backward_fill → [50, 50, 75]
        b = result.filter(pl.col("meta_flight_id") == "B")
        assert b["fdm_d_alt_ms"][0] == pytest.approx(50.0)  # backward fill
        assert b["fdm_d_alt_ms"][1] == pytest.approx(50.0)
        assert b["fdm_d_alt_ms"][2] == pytest.approx(75.0)

    def test_derivatives_first_value(self, two_flights_df: pl.DataFrame) -> None:
        """First value of each derivative uses backward_fill + fill_null(0.0)."""
        result = compute_derivatives(two_flights_df, dt=4.0)
        # First row should NOT be 0.0 — it's backward_fill of second row
        for col in ("fdm_d_alt_ms", "fdm_d_gamma_rads", "fdm_d_tas_ms2"):
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
        assert "fdm_d_alt_ms" not in result.columns

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
        assert result["fdm_d_alt_ms"][0] == pytest.approx(0.0)
        assert result["fdm_d_gamma_rads"][0] == pytest.approx(0.0)
        assert result["fdm_d_tas_ms2"][0] == pytest.approx(0.0)
