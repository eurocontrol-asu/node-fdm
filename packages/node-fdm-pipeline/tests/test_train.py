"""Tests for the training command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from node_fdm_pipeline.commands.train import run_training


class TestRunTraining:
    """Tests for ``run_training``."""

    def _make_config(self, tmp_path: Path) -> Path:
        """Create a valid YAML config and split CSV."""
        data_dir = tmp_path / "data"
        process_dir = data_dir / "processed_flights"
        models_dir = data_dir / "models"
        process_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
  - B738
"""
        )

        # Create a split CSV
        split_df = pl.DataFrame(
            {
                "filepath": ["/fake/flight1.parquet", "/fake/flight2.parquet"],
                "icao": ["abc123", "def456"],
                "split": ["train", "val"],
                "aircraft_type": ["A320", "A320"],
            }
        )
        split_df.write_csv(process_dir / "dataset_split.csv")

        return config

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_train_config_from_cli(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CLI args override YAML defaults in TrainingConfig."""
        config = self._make_config(tmp_path)

        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_get_data.return_value = (mock_train_ds, mock_val_ds)

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            run_training(
                arch="opensky",
                config=config,
                typecode="A320",
                epochs=10,
                batch_size=32,
                lr=0.01,
                device="cpu",
            )

        # Verify trainer was constructed and called
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args
        training_config = call_kwargs.kwargs["config"]

        assert training_config.epochs == 10
        assert training_config.batch_size == 32
        assert training_config.lr == 0.01
        mock_trainer.train.assert_called_once()

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_train_single_typecode(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When --typecode is given, only that typecode is trained."""
        config = self._make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            run_training(
                arch="opensky",
                config=config,
                typecode="A320",
                device="cpu",
            )

        # Only one call to trainer (A320, not B738)
        assert mock_trainer_cls.call_count == 1

    def test_train_missing_split(self, tmp_path: Path) -> None:
        """Raises SystemExit when dataset_split.csv is missing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
        )

        with (
            patch("node_fdm_pipeline.commands.train.importlib.import_module"),
            pytest.raises(SystemExit),
        ):
            run_training(arch="opensky", config=config, device="cpu")

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_train_empty_dataset(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Empty dataset for typecode → warning, skip, no crash."""
        config = self._make_config(tmp_path)

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            # Train for B738 which has no entries in split CSV
            run_training(
                arch="opensky",
                config=config,
                typecode="B738",
                device="cpu",
            )

        # No trainer created for empty dataset
        mock_trainer_cls.assert_not_called()

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_epoch_adjustment_default(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Epochs adjusted by dataset-size coefficient when not overridden."""
        config = self._make_config(tmp_path)

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=5120)
        mock_get_data.return_value = (mock_train_ds, MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            run_training(
                arch="opensky",
                config=config,
                typecode="A320",
                device="cpu",
            )

        # batch_size=512, n_step=10, coeff=5.0, adjusted=4000
        training_config = mock_trainer_cls.call_args.kwargs["config"]
        assert training_config.epochs == 4000

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_epoch_adjustment_capped(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Coefficient is capped at 10x for very small datasets."""
        config = self._make_config(tmp_path)

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)
        mock_get_data.return_value = (mock_train_ds, MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            run_training(
                arch="opensky",
                config=config,
                typecode="A320",
                device="cpu",
            )

        # batch_size=512, len=10, n_step=max(0,1)=1, coeff=min(50,10)=10, adjusted=8000
        training_config = mock_trainer_cls.call_args.kwargs["config"]
        assert training_config.epochs == 8000
