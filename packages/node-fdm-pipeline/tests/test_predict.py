"""Tests for prediction commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from node_fdm_pipeline.commands.predict import run_predict, run_predict_bada

if TYPE_CHECKING:
    from node_fdm_pipeline.resolver import ArchitectureInfo


class TestRunPredict:
    """Tests for ``run_predict``."""

    def _make_config_and_data(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create YAML config, split CSV, and a fake flight parquet."""
        data_dir = tmp_path / "data"
        process_dir = data_dir / "processed_flights"
        models_dir = data_dir / "models"
        acft_dir = process_dir / "A320"
        process_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        acft_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        # Create a fake flight parquet
        flight = pl.DataFrame(
            {
                "altitude_ft": [35000.0] * 10,
                "alt_sel_ft": [35000.0] * 10,
                "vz_sel_ftmin": [0.0] * 10,
                "mach_sel": [0.82] * 10,
                "cas_sel_kt": [280.0] * 10,
                "groundspeed": [450.0] * 10,
                "vertical_rate": [0.0] * 10,
                "latitude": [48.0] * 10,
                "longitude": [2.0] * 10,
                "track": [90.0] * 10,
                "flight_id": ["F001"] * 10,
                "timestamp": list(range(10)),
            }
        )
        flight_path = acft_dir / "flight001.parquet"
        flight.write_parquet(flight_path)

        # Create split CSV
        split_df = pl.DataFrame(
            {
                "filepath": [str(flight_path)],
                "icao": ["abc123"],
                "split": ["test"],
                "aircraft_type": ["A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        # Create model dir
        model_dir = models_dir / "opensky_2025_A320"
        model_dir.mkdir()

        return config, flight_path

    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_pipeline.resolver.resolve_architecture")
    def test_predict_output_format(
        self,
        mock_resolve: MagicMock,
        mock_predictor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Output parquet has pred_* columns."""
        config, _flight_path = self._make_config_and_data(tmp_path)

        # Mock preprocessing_fn as a direct callable (no FlightProcessor)
        mock_preprocessing = MagicMock()
        mock_result = MagicMock()
        mock_result.select.return_value.to_numpy.return_value = np.zeros((10, 1), dtype=np.float32)
        mock_preprocessing.return_value = mock_result

        from node_fdm_pipeline.resolver import ArchitectureInfo

        mock_resolve.return_value = ArchitectureInfo(
            name="opensky_2025",
            x_cols=["distance_m"],
            u_cols=["alt_sel_m"],
            e0_cols=["long_wind_ms"],
            dx_cols=[(1, "gs_ms")],
            preprocessing_fn=mock_preprocessing,
            segment_filter_fn=None,
            architecture_import="node_fdm.architectures.opensky",
        )

        # Mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict_flight.return_value = {
            "alt_std_m": np.zeros(10),
            "tas_ms": np.ones(10),
        }
        mock_predictor_cls.return_value = mock_predictor

        run_predict(
            arch="opensky",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        # preprocessing_fn called directly with a DataFrame (not LazyFrame)
        mock_preprocessing.assert_called_once()
        call_arg = mock_preprocessing.call_args[0][0]
        assert isinstance(call_arg, pl.DataFrame)

        # Check output dir was created
        predict_dir = tmp_path / "data" / "predicted_flights" / "A320"
        assert predict_dir.exists()

        # Check predictor was called
        mock_predictor.predict_flight.assert_called_once()

    def test_predict_missing_model(self, tmp_path: Path) -> None:
        """Warning logged when model doesn't exist, typecode skipped."""
        config, _ = self._make_config_and_data(tmp_path)

        # Remove the model dir
        model_dir = tmp_path / "data" / "models" / "opensky_2025_A320"
        if model_dir.exists():
            model_dir.rmdir()

        # Should not raise — just logs warning and skips
        run_predict(
            arch="opensky",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_data.processor.FlightProcessor")
    def test_predict_empty_test_set(
        self,
        mock_processor_cls: MagicMock,
        mock_predictor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Warning logged when no test flights for typecode."""
        config, _ = self._make_config_and_data(tmp_path)

        # Overwrite split CSV with no test flights
        process_dir = tmp_path / "data" / "processed_flights"
        split_df = pl.DataFrame(
            {
                "filepath": ["/fake.parquet"],
                "icao": ["abc123"],
                "split": ["train"],
                "aircraft_type": ["A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        # Should not raise — just logs warning and skips
        run_predict(
            arch="opensky",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        # Predictor should not have been used for prediction
        mock_predictor_cls.return_value.predict_flight.assert_not_called()


class TestPredictSIPreprocessing:
    """Regression tests for AXM-492: prediction must use SI preprocessing."""

    @pytest.fixture()
    def sample_flight(self) -> pl.DataFrame:
        """Processed parquet data with schema column names, mimicking pipeline output.

        Reflects the state after ``flight_processing`` + ``build_selected_params``
        have run (i.e. what is stored in processed_flights/ parquet files).
        ``training_preprocessing`` then converts these to SI units.
        """
        n = 20
        return pl.DataFrame(
            {
                "timestamp": list(range(n)),
                "latitude": [48.0 + i * 0.01 for i in range(n)],
                "longitude": [2.0 + i * 0.01 for i in range(n)],
                # Already renamed by flight_processing
                "raw_alt_ft": [35000.0] * n,
                "bds_mcp_sel_alt_ft": [35000.0] * n,
                "raw_vz_ftmin": [0.0] * n,
                "era_mach": [0.82] * n,
                "bds_ias_kt": [280.0] * n,
                "era_tas_kt": [450.0] * n,
                "raw_gs_kt": [440.0] * n,
                "track": [90.0] * n,
                "flight_id": ["F001"] * n,
                "temperature": [220.0] * n,
                "adep_dist": [500.0] * n,
                "ades_dist": [300.0] * n,
                "distance_along_track_m": [float(i * 1000) for i in range(n)],
                # Added by build_selected_params (segment detection)
                "fdm_mach_sel": [0.82] * n,
                "fdm_cas_sel_kt": [144.0] * n,
                "fdm_vz_sel_ftmin": [0.0] * n,
                "fdm_mcp_alt_sel_ft": [35000.0] * n,
                # Derived columns added by flight_processing
                "fdm_gamma_rad": [0.0] * n,
                "fdm_long_wind_kt": [10.0] * n,
            }
        )

    def test_predict_produces_si_columns(self, sample_flight: pl.DataFrame) -> None:
        """Prediction preprocessing outputs SI-unit columns (new schema names)."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(sample_flight)

        si_cols = {"raw_alt_m", "era_tas_ms", "fdm_gamma_rad", "era_temp_K"}
        assert si_cols.issubset(set(result.columns))

    def test_predict_preprocessing_matches_training(self, sample_flight: pl.DataFrame) -> None:
        """Columns from prediction path match training path."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing
        from node_fdm_data.schemas.opensky import E0_COLS, U_COLS, X_COLS

        result = training_preprocessing(sample_flight)

        # All schema columns must be present after preprocessing
        for col_list, label in [
            (X_COLS, "x_cols"),
            (U_COLS, "u_cols"),
            (E0_COLS, "e0_cols"),
        ]:
            missing = set(col_list) - set(result.columns)
            assert not missing, f"{label} missing columns: {missing}"

    def test_predict_no_nan_on_clean_input(self, sample_flight: pl.DataFrame) -> None:
        """No NaN in schema columns when input data is complete."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing
        from node_fdm_data.schemas.opensky import E0_COLS, U_COLS, X_COLS

        result = training_preprocessing(sample_flight)

        all_cols = X_COLS + U_COLS + E0_COLS
        for col in all_cols:
            null_count = result[col].null_count()
            nan_sum = result[col].is_nan().sum()
            assert null_count == 0, f"{col} has {null_count} nulls"
            assert nan_sum == 0, f"{col} has {nan_sum} NaNs"


class TestRunPredictBada:
    """Tests for ``run_predict_bada``."""

    def _make_config(self, tmp_path: Path) -> Path:
        """Create config with BADA settings and split CSV."""
        data_dir = tmp_path / "data"
        process_dir = data_dir / "processed_flights"
        process_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
  - UNKNOWN

bada:
  bada_4_2_dir: "{tmp_path / "bada"}"
"""
        )

        # Create split CSV
        split_df = pl.DataFrame(
            {
                "filepath": ["/fake/flight.parquet"],
                "icao": ["abc123"],
                "split": ["test"],
                "aircraft_type": ["A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        return config

    @patch("node_fdm_bada.aircraft_mapping.get_bada_identifier")
    def test_predict_bada_missing_mapping(
        self,
        mock_get_bada: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Unknown typecode → warning logged, skipped, continues."""
        config = self._make_config(tmp_path)

        mock_get_bada.side_effect = KeyError("UNKNOWN not found")

        # Should not raise — graceful skip
        run_predict_bada(config=config, typecode="UNKNOWN", jobs=1)

        mock_get_bada.assert_called_once_with("UNKNOWN")

    @patch("node_fdm_bada.aircraft_mapping.get_bada_identifier")
    def test_predict_bada_success(
        self,
        mock_get_bada: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Valid typecode → Parallel invoked, output dir created."""
        import sys

        config = self._make_config(tmp_path)

        mock_get_bada.return_value = "A320_BADA"

        # Inject fake pyBADA module into sys.modules
        mock_bada4 = MagicMock()
        mock_bada_pkg = MagicMock()
        mock_bada_pkg.bada4 = mock_bada4
        mock_parallel = MagicMock(return_value=MagicMock(return_value=[]))

        with (
            patch.dict(sys.modules, {"pyBADA": mock_bada_pkg, "pyBADA.bada4": mock_bada4}),
            patch.dict(
                sys.modules, {"joblib": MagicMock(Parallel=mock_parallel, delayed=lambda f: f)}
            ),
        ):
            run_predict_bada(config=config, typecode="A320", jobs=2)

        mock_get_bada.assert_called_once_with("A320")
        mock_parallel.assert_called_once()

        # Output dir created
        bada_dir = tmp_path / "data" / "bada_flights" / "A320"
        assert bada_dir.exists()

    def test_predict_bada_missing_split(self, tmp_path: Path) -> None:
        """SystemExit when split CSV is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )
        with pytest.raises(SystemExit, match="fdm process"):
            run_predict_bada(config=config, typecode="A320", jobs=1)

    @patch("node_fdm_bada.aircraft_mapping.get_bada_identifier")
    def test_predict_bada_empty_test_set(
        self,
        mock_get_bada: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Empty test set → warning logged, no crash."""
        import sys

        config = self._make_config(tmp_path)

        # Overwrite split CSV so A320 has train only
        process_dir = tmp_path / "data" / "processed_flights"
        split_df = pl.DataFrame(
            {
                "filepath": ["/fake/flight.parquet"],
                "icao": ["abc123"],
                "split": ["train"],
                "aircraft_type": ["A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        mock_get_bada.return_value = "A320_BADA"
        mock_bada4 = MagicMock()
        mock_bada_pkg = MagicMock()
        mock_bada_pkg.bada4 = mock_bada4

        with patch.dict(sys.modules, {"pyBADA": mock_bada_pkg, "pyBADA.bada4": mock_bada4}):
            run_predict_bada(config=config, typecode="A320", jobs=1)


class TestPredictNanFiltering:
    """Regression tests for AXM-493: filter NaN segments in prediction."""

    def _make_config_and_nan_flight(
        self,
        tmp_path: Path,
        *,
        nan_fraction: float,
    ) -> tuple[Path, Path]:
        """Create config + flight parquet with controllable NaN fraction.

        Args:
            tmp_path: Pytest temporary directory.
            nan_fraction: Fraction of rows to set mach_sel to NaN (0.0-1.0).

        Returns:
            Tuple of (config_path, flight_parquet_path).
        """
        data_dir = tmp_path / "data"
        process_dir = data_dir / "processed_flights"
        models_dir = data_dir / "models"
        acft_dir = process_dir / "A320"
        process_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        acft_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        # Build flight: n rows, first nan_count rows have NaN mach_sel
        n = 100
        nan_count = int(n * nan_fraction)
        mach_sel = [float("nan")] * nan_count + [0.82] * (n - nan_count)

        flight = pl.DataFrame(
            {
                "altitude_ft": [35000.0] * n,
                "alt_sel_ft": [35000.0] * n,
                "vz_sel_ftmin": [0.0] * n,
                "mach_sel": mach_sel,
                "cas_sel_kt": [280.0] * n,
                "groundspeed": [450.0] * n,
                "vertical_rate": [0.0] * n,
                "latitude": [48.0] * n,
                "longitude": [2.0] * n,
                "track": [90.0] * n,
                "flight_id": ["F001"] * n,
                "timestamp": list(range(n)),
            }
        )
        flight_path = acft_dir / "flight001.parquet"
        flight.write_parquet(flight_path)

        # Split CSV
        split_df = pl.DataFrame(
            {
                "filepath": [str(flight_path)],
                "icao": ["abc123"],
                "split": ["test"],
                "aircraft_type": ["A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        # Model dir
        model_dir = models_dir / "opensky_2025_A320"
        model_dir.mkdir()

        return config, flight_path

    def _mock_architecture(self) -> ArchitectureInfo:
        """Build a mock ArchitectureInfo that passes through columns.

        When nan_passthrough is True, preprocessing returns raw data as-is
        (preserving NaN for testing the filter).
        """
        from node_fdm_pipeline.resolver import ArchitectureInfo

        # Preprocessing: identity (return input as-is to preserve NaN)
        def identity_preprocess(df: pl.DataFrame) -> pl.DataFrame:
            return df

        mock_arch = ArchitectureInfo(
            name="opensky_2025",
            x_cols=["altitude_ft"],
            u_cols=["mach_sel", "cas_sel_kt"],
            e0_cols=["groundspeed"],
            dx_cols=[(1, "vertical_rate")],
            preprocessing_fn=identity_preprocess,
            segment_filter_fn=None,
            architecture_import="node_fdm.architectures.opensky",
        )
        return mock_arch

    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_pipeline.resolver.resolve_architecture")
    def test_predict_filters_nan_segments(
        self,
        mock_resolve: MagicMock,
        mock_predictor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Flight with 60% NaN mach_sel → only ~40 finite rows predicted."""
        config, _ = self._make_config_and_nan_flight(tmp_path, nan_fraction=0.6)

        mock_resolve.return_value = self._mock_architecture()

        mock_predictor = MagicMock()

        # Return arrays sized to whatever input length is passed
        def fake_predict(
            x_init: np.ndarray,
            u_seq: np.ndarray,
            e_seq: np.ndarray,
        ) -> dict[str, np.ndarray]:
            return {"altitude_ft": np.zeros(len(u_seq))}

        mock_predictor.predict_flight.side_effect = fake_predict
        mock_predictor_cls.return_value = mock_predictor

        run_predict(
            arch="opensky",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        # predict_flight must have been called with filtered (shorter) arrays
        mock_predictor.predict_flight.assert_called_once()
        call_args = mock_predictor.predict_flight.call_args
        u_seq_arg = call_args[0][1]  # second positional arg
        assert len(u_seq_arg) == 40, f"Expected 40 finite rows, got {len(u_seq_arg)}"

    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_pipeline.resolver.resolve_architecture")
    def test_predict_skips_flight_above_threshold(
        self,
        mock_resolve: MagicMock,
        mock_predictor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Flight with 90% NaN (>80% threshold) → skipped, not predicted."""
        config, _ = self._make_config_and_nan_flight(tmp_path, nan_fraction=0.9)

        mock_resolve.return_value = self._mock_architecture()

        mock_predictor = MagicMock()
        mock_predictor_cls.return_value = mock_predictor

        run_predict(
            arch="opensky",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        # predict_flight should NOT have been called — flight skipped
        mock_predictor.predict_flight.assert_not_called()

        # Output parquet should not exist
        output_file = tmp_path / "data" / "predicted_flights" / "A320" / "flight001.parquet"
        assert not output_file.exists()

    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_pipeline.resolver.resolve_architecture")
    def test_predict_clean_flight_unchanged(
        self,
        mock_resolve: MagicMock,
        mock_predictor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Flight with 0% NaN → all 100 timesteps predicted."""
        config, _ = self._make_config_and_nan_flight(tmp_path, nan_fraction=0.0)

        mock_resolve.return_value = self._mock_architecture()

        mock_predictor = MagicMock()

        def fake_predict(
            x_init: np.ndarray,
            u_seq: np.ndarray,
            e_seq: np.ndarray,
        ) -> dict[str, np.ndarray]:
            return {"altitude_ft": np.zeros(len(u_seq))}

        mock_predictor.predict_flight.side_effect = fake_predict
        mock_predictor_cls.return_value = mock_predictor

        run_predict(
            arch="opensky",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        # predict_flight called with all rows
        mock_predictor.predict_flight.assert_called_once()
        call_args = mock_predictor.predict_flight.call_args
        u_seq_arg = call_args[0][1]
        assert len(u_seq_arg) == 100, f"Expected 100 rows, got {len(u_seq_arg)}"


class TestPredictMissingSplit:
    """Edge case: missing split CSV for run_predict."""

    def test_predict_missing_split_csv(self, tmp_path: Path) -> None:
        """SystemExit when split CSV is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )
        with pytest.raises(SystemExit, match="fdm process"):
            run_predict(arch="opensky", config=config, typecode="A320")


class TestPredictXInitGuard:
    """Regression tests for AXM-494: x_init finite guard in predict_flight."""

    @patch("node_fdm.predictor.NodeFDMPredictor")
    @patch("node_fdm_pipeline.resolver.resolve_architecture")
    def test_predict_skips_flight_on_bad_x_init(
        self,
        mock_resolve: MagicMock,
        mock_predictor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """predict_flight raising ValueError → flight skipped, no output."""
        from node_fdm_pipeline.resolver import ArchitectureInfo

        # Setup config + data
        data_dir = tmp_path / "data"
        process_dir = data_dir / "processed_flights"
        models_dir = data_dir / "models"
        acft_dir = process_dir / "A320"
        process_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        acft_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        flight = pl.DataFrame(
            {
                "altitude_ft": [35000.0] * 10,
                "alt_sel_ft": [35000.0] * 10,
                "vz_sel_ftmin": [0.0] * 10,
                "mach_sel": [0.82] * 10,
                "cas_sel_kt": [280.0] * 10,
                "groundspeed": [450.0] * 10,
                "vertical_rate": [0.0] * 10,
                "latitude": [48.0] * 10,
                "longitude": [2.0] * 10,
                "track": [90.0] * 10,
                "flight_id": ["F001"] * 10,
                "timestamp": list(range(10)),
            }
        )
        flight_path = acft_dir / "flight001.parquet"
        flight.write_parquet(flight_path)

        split_df = pl.DataFrame(
            {
                "filepath": [str(flight_path)],
                "icao": ["abc123"],
                "split": ["test"],
                "aircraft_type": ["A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        model_dir = models_dir / "opensky_2025_A320"
        model_dir.mkdir()

        # Mock architecture
        mock_preprocessing = MagicMock()
        mock_result = MagicMock()
        mock_result.select.return_value.to_numpy.return_value = np.zeros((10, 1), dtype=np.float32)
        mock_preprocessing.return_value = mock_result

        mock_resolve.return_value = ArchitectureInfo(
            name="opensky_2025",
            x_cols=["distance_m"],
            u_cols=["alt_sel_m"],
            e0_cols=["long_wind_ms"],
            dx_cols=[(1, "gs_ms")],
            preprocessing_fn=mock_preprocessing,
            segment_filter_fn=None,
            architecture_import="node_fdm.architectures.opensky",
        )

        # Mock predictor to raise ValueError (simulating bad x_init)
        mock_predictor = MagicMock()
        mock_predictor.predict_flight.side_effect = ValueError(
            "x_init contains non-finite values: altitude_m=nan"
        )
        mock_predictor_cls.return_value = mock_predictor

        run_predict(
            arch="opensky",
            config=config,
            typecode="A320",
            device="cpu",
            local_model=True,
        )

        # predict_flight was called but ValueError was caught
        mock_predictor.predict_flight.assert_called_once()

        # No output parquet written for the skipped flight
        output_file = tmp_path / "data" / "predicted_flights" / "A320" / "flight001.parquet"
        assert not output_file.exists()
