"""Tests for node_fdm_data.preprocessing.opensky — OpenSky preprocessing."""

from __future__ import annotations

import polars as pl
import pytest

from node_fdm_data.preprocessing.opensky import flight_processing


class TestOpenSkyPreprocessing:
    """OpenSky 2025 preprocessing pipeline tests."""

    @pytest.fixture()
    def opensky_df(self) -> pl.LazyFrame:
        """Minimal OpenSky flight dataframe."""
        return pl.LazyFrame(
            {
                "raw_alt_ft": [5000.0, 10000.0, 15000.0, 20000.0, 25000.0],
                "bds_mcp_alt_sel_ft": [10000.0, 15000.0, 20000.0, 25000.0, 30000.0],
                "raw_vz_ftmin": [None, 500.0, None, 1000.0, 500.0],
                "era_mach": [0.5, None, 0.6, None, 0.7],
                "bds_ias_kt": [None, None, 200.0, 250.0, 300.0],
                "distance_m": [0.0, 500.0, 1200.0, 2000.0, 2800.0],
            }
        )

    def test_flight_processing_alt_diff(self, opensky_df: pl.LazyFrame) -> None:
        """flight_processing adds fdm_alt_diff_ft = bds_mcp_alt_sel_ft - raw_alt_ft."""
        result = flight_processing(opensky_df).collect()
        assert "fdm_alt_diff_ft" in result.columns
        expected = [5000.0, 5000.0, 5000.0, 5000.0, 5000.0]
        assert result["fdm_alt_diff_ft"].to_list() == expected

    def test_flight_processing_fills_nan(self, opensky_df: pl.LazyFrame) -> None:
        """NaN values in control columns are filled with 0."""
        result = flight_processing(opensky_df).collect()
        assert result["raw_vz_ftmin"].null_count() == 0
        assert result["era_mach"].null_count() == 0
        assert result["bds_ias_kt"].null_count() == 0

    def test_flight_processing_gamma_rad(self) -> None:
        """fdm_gamma_rad = arcsin(vz / TAS) is computed when both columns exist."""
        import numpy as np

        df = pl.LazyFrame(
            {
                "raw_alt_ft": [35000.0],
                "bds_mcp_alt_sel_ft": [36000.0],
                "era_tas_kt": [450.0],
                "raw_vz_ftmin": [1000.0],
                "fdm_mach_sel": [0.82],
                "bds_ias_kt": [280.0],
            }
        )
        result = flight_processing(df).collect()
        assert "fdm_gamma_rad" in result.columns
        gamma = result["fdm_gamma_rad"][0]
        # Manual: vz_ms = 1000 * FTMIN, tas_ms = 450 * KT
        from node_fdm_data.physics.constants import FTMIN, KT

        expected = np.arcsin((1000 * FTMIN) / (450 * KT))
        assert abs(gamma - expected) < 1e-6

    def test_flight_processing_long_wind(self) -> None:
        """fdm_long_wind_kt = TAS - GS computed when both columns exist."""
        df = pl.LazyFrame(
            {
                "raw_alt_ft": [35000.0],
                "bds_mcp_alt_sel_ft": [36000.0],
                "era_tas_kt": [450.0],
                "raw_gs_kt": [430.0],
                "raw_vz_ftmin": [0.0],
                "fdm_mach_sel": [0.82],
                "bds_ias_kt": [280.0],
            }
        )
        result = flight_processing(df).collect()
        assert "fdm_long_wind_kt" in result.columns
        assert result["fdm_long_wind_kt"][0] == pytest.approx(20.0)

    def test_flight_processing_with_traffic_names(self) -> None:
        """Columns with pre-rename names (altitude, TAS, etc) get renamed."""
        df = pl.LazyFrame(
            {
                "altitude": [35000.0],
                "selected_mcp": [36000.0],
                "vertical_rate": [1000.0],
                "Mach": [0.82],
                "IAS": [280.0],
                "TAS": [450.0],
                "groundspeed": [430.0],
            }
        )
        result = flight_processing(df).collect()
        # Check renamed columns exist
        assert "raw_alt_ft" in result.columns
        assert "bds_mcp_alt_sel_ft" in result.columns
        assert "raw_vz_ftmin" in result.columns
        assert "fdm_alt_diff_ft" in result.columns
        # Mach → era_mach (not fdm_mach_sel)
        assert "era_mach" in result.columns
        assert "fdm_mach_sel" not in result.columns

    def test_mach_column_preserved_after_rename(self) -> None:
        """Mach is renamed to 'era_mach' (continuous), not consumed into 'fdm_mach_sel'."""
        df = pl.LazyFrame(
            {
                "altitude": [35000.0, 35000.0],
                "selected_mcp": [36000.0, 36000.0],
                "vertical_rate": [0.0, 0.0],
                "Mach": [0.78, 0.80],
                "IAS": [280.0, 285.0],
                "TAS": [450.0, 455.0],
                "groundspeed": [430.0, 435.0],
            }
        )
        result = flight_processing(df).collect()
        assert "era_mach" in result.columns
        assert result["era_mach"].to_list() == [0.78, 0.80]


class TestCumulativeDistance:
    """Tests for ``cumulative_distance``."""

    def test_basic(self) -> None:
        """Cumulative distance accumulates haversine distances."""
        from node_fdm_data.preprocessing.opensky import cumulative_distance

        df = pl.DataFrame(
            {
                "latitude": [48.0, 48.01, 48.02, 48.03],
                "longitude": [2.0, 2.0, 2.0, 2.0],
            }
        )
        result = cumulative_distance(df)
        assert "distance_along_track_m" in result.columns
        d = result["distance_along_track_m"].to_list()
        assert d[0] == 0.0
        assert all(d[i] > d[i - 1] for i in range(1, len(d)))

    def test_single_point(self) -> None:
        """Single-row DataFrame gets distance = 0."""
        from node_fdm_data.preprocessing.opensky import cumulative_distance

        df = pl.DataFrame({"latitude": [48.0], "longitude": [2.0]})
        result = cumulative_distance(df)
        assert result["distance_along_track_m"][0] == 0.0

    def test_null_coords(self) -> None:
        """Null lat/lon rows are dropped — no NaN in output (AXM-511)."""
        from node_fdm_data.preprocessing.opensky import cumulative_distance

        df = pl.DataFrame(
            {
                "latitude": [48.0, None, 48.02, None, 48.04],
                "longitude": [2.0, None, 2.0, None, 2.0],
            }
        )
        result = cumulative_distance(df)
        assert len(result) == 3
        d = result["distance_along_track_m"]
        assert d.null_count() == 0
        assert d.is_nan().sum() == 0
        assert d[0] == 0.0
        assert all(d[i] > d[i - 1] for i in range(1, len(d)))
