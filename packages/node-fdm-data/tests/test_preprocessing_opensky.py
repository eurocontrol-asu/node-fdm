"""Tests for node_fdm_data.preprocessing.opensky — OpenSky preprocessing."""

from __future__ import annotations

import polars as pl
import pytest

from node_fdm_data.preprocessing.opensky import (
    LOW_THR,
    UPPER_THR,
    flight_processing,
    segment_filtering,
)


class TestOpenSkyPreprocessing:
    """OpenSky 2025 preprocessing pipeline tests."""

    @pytest.fixture()  # type: ignore[misc]
    def opensky_df(self) -> pl.LazyFrame:
        """Minimal OpenSky flight dataframe."""
        return pl.LazyFrame(
            {
                "altitude_ft": [5000.0, 10000.0, 15000.0, 20000.0, 25000.0],
                "alt_sel_ft": [10000.0, 15000.0, 20000.0, 25000.0, 30000.0],
                "vz_sel_ftmin": [None, 500.0, None, 1000.0, 500.0],
                "mach_sel": [0.5, None, 0.6, None, 0.7],
                "cas_sel_kt": [None, None, 200.0, 250.0, 300.0],
                "distance_m": [0.0, 500.0, 1200.0, 2000.0, 2800.0],
            }
        )

    def test_flight_processing_alt_diff(self, opensky_df: pl.LazyFrame) -> None:
        """flight_processing adds alt_diff_ft = alt_sel_ft - altitude_ft."""
        result = flight_processing(opensky_df).collect()
        assert "alt_diff_ft" in result.columns
        expected = [5000.0, 5000.0, 5000.0, 5000.0, 5000.0]
        assert result["alt_diff_ft"].to_list() == expected

    def test_flight_processing_fills_nan(self, opensky_df: pl.LazyFrame) -> None:
        """NaN values in control columns are filled with 0."""
        result = flight_processing(opensky_df).collect()
        assert result["vz_sel_ftmin"].null_count() == 0
        assert result["mach_sel"].null_count() == 0
        assert result["cas_sel_kt"].null_count() == 0

    def test_segment_filtering_valid(self) -> None:
        """Valid segment with diffs within thresholds returns True."""
        df = pl.DataFrame(
            {
                "distance_m": [0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0],
            }
        )
        # diffs are 500 each — within [LOW_THR, UPPER_THR]
        assert segment_filtering(df, start=0, seq_len=5) is True

    def test_segment_filtering_invalid(self) -> None:
        """Segment with diff outside thresholds returns False."""
        df = pl.DataFrame(
            {
                "distance_m": [0.0, 50.0, 100.0, 150.0, 200.0, 250.0],
            }
        )
        # diffs are 50 each — below LOW_THR
        assert segment_filtering(df, start=0, seq_len=5) is False

    def test_segment_filtering_short_flight(self) -> None:
        """Flight with < seq_len rows returns False gracefully (no error)."""
        df = pl.DataFrame(
            {
                "distance_m": [0.0, 500.0],
            }
        )
        result = segment_filtering(df, start=0, seq_len=10)
        assert result is False

    def test_thresholds_exported(self) -> None:
        """LOW_THR and UPPER_THR are accessible constants."""
        assert LOW_THR == 200
        assert UPPER_THR == 3000
