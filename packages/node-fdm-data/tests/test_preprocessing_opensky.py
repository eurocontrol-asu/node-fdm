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
                "mach": [0.5, None, 0.6, None, 0.7],
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
        assert result["mach"].null_count() == 0
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
        # Mach → mach (not mach_sel)
        assert "mach" in result.columns
        assert "mach_sel" not in result.columns

    def test_mach_column_preserved_after_rename(self) -> None:
        """Mach is renamed to 'mach' (continuous), not consumed into 'mach_sel'."""
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
        assert "mach" in result.columns
        assert result["mach"].to_list() == [0.78, 0.80]

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


class TestTrainingPreprocessingSI:
    """SI unit conversions in ``training_preprocessing``."""

    @pytest.fixture()
    def flight_df(self) -> pl.DataFrame:
        """Flight DataFrame with raw units (ft, kt, ft/min, °C, NM)."""
        return pl.DataFrame(
            {
                "altitude_ft": [10000.0, 20000.0, 35000.0],
                "alt_sel_ft": [15000.0, 25000.0, 36000.0],
                "TAS": [250.0, 350.0, 450.0],
                "groundspeed": [240.0, 340.0, 430.0],
                "vertical_rate": [1000.0, 500.0, 0.0],
                "Mach": [0.5, 0.7, 0.82],
                "IAS": [200.0, 250.0, 280.0],
                "mach_sel": [0.5, 0.7, 0.82],  # from build_selected_params
                "temperature": [-10.0, -30.0, -50.0],
                "adep_dist": [0.0, 50.0, 100.0],
                "ades_dist": [200.0, 150.0, 100.0],
                "distance_along_track_m": [0.0, 5000.0, 10000.0],
            }
        )

    def test_altitude_ft_to_m(self, flight_df: pl.DataFrame) -> None:
        """altitude_ft -> altitude_m via ft_to_m (x 0.3048)."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(flight_df)
        assert "altitude_m" in result.columns
        assert result["altitude_m"][0] == pytest.approx(10000.0 * 0.3048)
        assert result["altitude_m"][2] == pytest.approx(35000.0 * 0.3048)

    def test_tas_kt_to_ms(self, flight_df: pl.DataFrame) -> None:
        """TAS (kt) -> tas_ms via kt_to_ms (x 0.514444)."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(flight_df)
        assert "tas_ms" in result.columns
        assert result["tas_ms"][2] == pytest.approx(450.0 * 0.514444, rel=1e-4)

    def test_temperature_c_to_k(self, flight_df: pl.DataFrame) -> None:
        """temperature (°C) → temperature_K via celsius_to_kelvin (+ 273.15)."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(flight_df)
        assert "temperature_K" in result.columns
        assert result["temperature_K"][0] == pytest.approx(263.15)
        assert result["temperature_K"][2] == pytest.approx(223.15)

    def test_vz_derivative_si(self, flight_df: pl.DataFrame) -> None:
        """vz_ms is diff of altitude_m (SI derivative)."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(flight_df)
        assert "vz_ms" in result.columns
        # First row: fill_null → 0.0
        assert result["vz_ms"][0] == pytest.approx(0.0)
        # Second row: (20000 - 10000) * 0.3048 = 3048.0
        assert result["vz_ms"][1] == pytest.approx(10000.0 * 0.3048)

    def test_d_gamma_rads_derivative(self, flight_df: pl.DataFrame) -> None:
        """d_gamma_rads is diff of gamma_rad."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(flight_df)
        assert "d_gamma_rads" in result.columns
        assert result["d_gamma_rads"][0] == pytest.approx(0.0)

    def test_d_tas_ms_derivative(self, flight_df: pl.DataFrame) -> None:
        """d_tas_ms is diff of tas_ms."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(flight_df)
        assert "d_tas_ms" in result.columns
        assert result["d_tas_ms"][0] == pytest.approx(0.0)
        # 100 kt difference = 100 * 0.514444 m/s
        assert result["d_tas_ms"][1] == pytest.approx(100.0 * 0.514444, rel=1e-4)

    def test_distance_nm_to_m(self, flight_df: pl.DataFrame) -> None:
        """adep_dist (NM) -> adep_dist_m via nm_to_m (x 1852)."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(flight_df)
        assert "adep_dist_m" in result.columns
        assert result["adep_dist_m"][1] == pytest.approx(50.0 * 1852.0)

    def test_gamma_rad_unchanged(self, flight_df: pl.DataFrame) -> None:
        """gamma_air (already radians) is only renamed to gamma_rad, not scaled."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(flight_df)
        assert "gamma_rad" in result.columns
        # gamma_air was computed from vz and TAS — check it's in valid range
        gamma = result["gamma_rad"][0]
        assert -1.57 < gamma < 1.57  # within ±π/2

    def test_all_schema_cols_present(self, flight_df: pl.DataFrame) -> None:
        """After preprocessing, all X_COLS and U_COLS from schema exist."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing
        from node_fdm_data.schemas.opensky import U_COLS, X_COLS

        result = training_preprocessing(flight_df)
        for col in X_COLS:
            assert col in result.columns, f"Missing X_COL: {col}"
        for col in U_COLS:
            assert col in result.columns, f"Missing U_COL: {col}"
