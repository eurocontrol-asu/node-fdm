"""Tests for prediction commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl

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
    @patch("node_fdm_data.processor.FlightProcessor")
    def test_predict_output_format(
        self,
        mock_processor_cls: MagicMock,
        mock_predictor_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Output parquet has pred_* columns."""
        config, _flight_path = self._make_config_and_data(tmp_path)

        # Mock processor
        mock_processor = MagicMock()
        mock_collected = MagicMock()
        mock_collected.select.return_value.to_numpy.return_value = np.zeros(
            (10, 1), dtype=np.float32
        )
        mock_processor.process.return_value.collect.return_value = mock_collected
        mock_processor_cls.return_value = mock_processor

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
