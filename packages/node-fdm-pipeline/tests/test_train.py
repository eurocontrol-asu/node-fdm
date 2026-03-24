"""Tests for the training command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from node_fdm_pipeline.commands.train import run_training


def _make_delta_table(path: Path, *, typecodes: list[str] | None = None) -> None:
    """Create a minimal Delta Table with required columns."""
    typecodes = typecodes or ["A320"]
    rng = np.random.default_rng(42)
    rows = []
    for acft in typecodes:
        for i in range(4):
            split = ["train", "train", "val", "test"][i]
            fid = f"abc123_{acft}_{i:02d}_s0"
            for _j in range(20):
                rows.append(
                    {
                        "meta_flight_id": fid,
                        "meta_split": split,
                        "meta_aircraft_type": acft,
                        "fdm_flag_valid": True,
                        "fdm_flag_distance_ok": True,
                        "raw_alt_m": float(rng.uniform(300, 10000)),
                        "era_tas_ms": float(rng.uniform(100, 250)),
                    }
                )
    df = pl.DataFrame(rows)
    df.write_delta(str(path), mode="overwrite")


class TestRunTraining:
    """Tests for ``run_training``."""

    def _make_config(self, tmp_path: Path) -> Path:
        """Create a valid YAML config and Delta Table."""
        data_dir = tmp_path / "data"
        models_dir = data_dir / "models"
        data_dir.mkdir(parents=True)
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

        # Create Delta Table
        _make_delta_table(data_dir / "flights.delta", typecodes=["A320"])

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

    def test_train_missing_delta(self, tmp_path: Path) -> None:
        """Raises SystemExit when Delta Table doesn't exist."""
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
            # Train for B738 which has no entries in Delta Table
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

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_cli_seq_len_param(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--seq-len CLI arg propagates to TrainingConfig.seq_len."""
        config = self._make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            run_training(
                arch="opensky",
                config=config,
                typecode="A320",
                epochs=1,
                seq_len=200,
                device="cpu",
            )

        training_config = mock_trainer_cls.call_args.kwargs["config"]
        assert training_config.seq_len == 200

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_shift_defaults_to_seq_len(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When --shift is not given, shift defaults to seq_len."""
        config = self._make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            run_training(
                arch="opensky",
                config=config,
                typecode="A320",
                seq_len=200,
                device="cpu",
            )

        training_config = mock_trainer_cls.call_args.kwargs["config"]
        assert training_config.shift == 200

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_shift_explicit(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Explicit --shift overrides the seq_len default."""
        config = self._make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            run_training(
                arch="opensky",
                config=config,
                typecode="A320",
                seq_len=200,
                shift=100,
                device="cpu",
            )

        training_config = mock_trainer_cls.call_args.kwargs["config"]
        assert training_config.shift == 100

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_train_e1_cols_in_stats(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """E1 columns passed to get_train_val_data so stats include them.

        When e1_cols are forwarded, the trainer computes stats for them and
        they appear in the saved meta.json stats_dict.
        """
        from node_fdm_data.schemas.opensky import E1_COLS

        config = self._make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with patch("node_fdm_pipeline.commands.train.importlib.import_module"):
            run_training(
                arch="opensky",
                config=config,
                typecode="A320",
                epochs=1,
                device="cpu",
            )

        # get_train_val_data must receive e1_cols kwarg matching the architecture
        data_kwargs = mock_get_data.call_args.kwargs
        assert "e1_cols" in data_kwargs, "e1_cols not passed to get_train_val_data"
        assert data_kwargs["e1_cols"] == E1_COLS
        assert "fdm_alt_diff_m" in data_kwargs["e1_cols"]
