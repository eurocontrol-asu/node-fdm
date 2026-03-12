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

    @pytest.fixture()
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

    def test_flight_processing_gamma_air(self) -> None:
        """gamma_air = arcsin(vz / TAS) is computed when both columns exist."""
        import numpy as np

        df = pl.LazyFrame(
            {
                "altitude_ft": [35000.0],
                "alt_sel_ft": [36000.0],
                "tas_kt": [450.0],
                "vz_sel_ftmin": [1000.0],
                "mach_sel": [0.82],
                "cas_sel_kt": [280.0],
            }
        )
        result = flight_processing(df).collect()
        assert "gamma_air" in result.columns
        gamma = result["gamma_air"][0]
        # Manual: vz_ms = 1000 * FTMIN, tas_ms = 450 * KT
        from node_fdm_data.physics.constants import FTMIN, KT

        expected = np.arcsin((1000 * FTMIN) / (450 * KT))
        assert abs(gamma - expected) < 1e-6

    def test_flight_processing_long_wind(self) -> None:
        """long_wind = TAS - GS computed when both columns exist."""
        df = pl.LazyFrame(
            {
                "altitude_ft": [35000.0],
                "alt_sel_ft": [36000.0],
                "tas_kt": [450.0],
                "gs_kt": [430.0],
                "vz_sel_ftmin": [0.0],
                "mach_sel": [0.82],
                "cas_sel_kt": [280.0],
            }
        )
        result = flight_processing(df).collect()
        assert "long_wind" in result.columns
        assert result["long_wind"][0] == pytest.approx(20.0)

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
        assert "altitude_ft" in result.columns
        assert "alt_sel_ft" in result.columns
        assert "vz_sel_ftmin" in result.columns
        assert "alt_diff_ft" in result.columns

    def test_segment_filtering_empty_segment(self) -> None:
        """Segment of length 1 (empty diffs) returns False."""
        df = pl.DataFrame({"distance_m": [500.0]})
        assert segment_filtering(df, start=0, seq_len=1) is False


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


class TestCropOnDistanceJump:
    """Tests for ``crop_on_distance_jump``."""

    def test_crop_removes_discontinuity(self) -> None:
        """Trajectory with a jump is cropped at jump boundaries."""
        import numpy as np

        from node_fdm_data.preprocessing.opensky import crop_on_distance_jump

        n = 20
        # Normal distances with a big jump in the middle
        dists = np.concatenate(
            [
                np.linspace(0, 5000, 8),
                [5000 + 500],  # jump of 500 (> 200 threshold)
                np.linspace(6000, 10000, 11),
            ]
        )
        df = pl.DataFrame(
            {
                "distance_along_track_m": dists,
                "gs_kt": np.full(n, 200.0),
            }
        )
        result = crop_on_distance_jump(df, threshold=200.0)
        assert len(result) <= n

    def test_crop_no_jumps(self) -> None:
        """Smooth trajectory is unchanged."""
        import numpy as np

        from node_fdm_data.preprocessing.opensky import crop_on_distance_jump

        n = 10
        df = pl.DataFrame(
            {
                "distance_along_track_m": np.linspace(0, 1000, n),
                "gs_kt": np.full(n, 200.0),
            }
        )
        result = crop_on_distance_jump(df, threshold=200.0)
        assert len(result) == n

    def test_crop_filters_low_speed(self) -> None:
        """Rows with speed below min_speed are removed."""
        import numpy as np

        from node_fdm_data.preprocessing.opensky import crop_on_distance_jump

        df = pl.DataFrame(
            {
                "distance_along_track_m": [0.0, 100.0, 200.0, 300.0],
                "gs_kt": [50.0, 200.0, 200.0, 200.0],
            }
        )
        result = crop_on_distance_jump(df, threshold=200.0, min_speed=90.0)
        # First row (50 kt) should be filtered out
        assert len(result) < 4

    def test_crop_short_df(self) -> None:
        """DataFrame with < 2 rows after filter returns unchanged."""
        from node_fdm_data.preprocessing.opensky import crop_on_distance_jump

        df = pl.DataFrame(
            {
                "distance_along_track_m": [0.0],
                "gs_kt": [200.0],
            }
        )
        result = crop_on_distance_jump(df, threshold=200.0)
        assert len(result) == 1

    def test_crop_with_groundspeed_column(self) -> None:
        """Works with 'groundspeed' column name (pre-rename)."""
        import numpy as np

        from node_fdm_data.preprocessing.opensky import crop_on_distance_jump

        df = pl.DataFrame(
            {
                "distance_along_track_m": np.linspace(0, 1000, 5),
                "groundspeed": np.full(5, 200.0),
            }
        )
        result = crop_on_distance_jump(df, threshold=500.0)
        assert len(result) == 5

