"""Tests for prediction commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from node_fdm_pipeline.commands.predict import run_predict, run_predict_bada


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
        """Raw flight data with non-SI columns, mimicking processed parquet."""
        n = 20
        return pl.DataFrame(
            {
                "timestamp": list(range(n)),
                "latitude": [48.0 + i * 0.01 for i in range(n)],
                "longitude": [2.0 + i * 0.01 for i in range(n)],
                "altitude": [35000.0] * n,
                "selected_mcp": [35000.0] * n,
                "vertical_rate": [0.0] * n,
                "Mach": [0.82] * n,
                "IAS": [280.0] * n,
                "TAS": [450.0] * n,
                "groundspeed": [440.0] * n,
                "track": [90.0] * n,
                "flight_id": ["F001"] * n,
                "mach_sel": [0.82] * n,
                "temperature": [220.0] * n,
                "adep_dist": [500.0] * n,
                "ades_dist": [300.0] * n,
                "distance_along_track_m": [float(i * 1000) for i in range(n)],
            }
        )

    def test_predict_produces_si_columns(self, sample_flight: pl.DataFrame) -> None:
        """Prediction preprocessing outputs SI-unit columns."""
        from node_fdm_data.preprocessing.opensky import training_preprocessing

        result = training_preprocessing(sample_flight)

        si_cols = {"altitude_m", "tas_ms", "gamma_rad", "temperature_K"}
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
