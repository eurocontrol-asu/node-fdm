"""Tests for the training command."""

from __future__ import annotations

from contextlib import nullcontext
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


def _make_config(tmp_path: Path) -> Path:
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
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01          # opensky26 tbl.3
    n_passes: 2
    slope_tol: 3.0e-4      # opensky26 tbl.3
    flat_tol: 5.0e-2
    min_len: 15
  cas:
    cutoff_s: 180.0
    sigma_s: 8.0
    sigma_r: 3.0           # opensky26 tbl.3
    n_passes: 2
    slope_tol: 0.09        # opensky26 tbl.3
    flat_tol: 20.0
    min_len: 5
  vz:
    sigma_s: 6.0
    sigma_r: 100.0         # opensky26 tbl.3
    slope_tol: 50.0        # opensky26 tbl.3
    flat_tol: 100.0
    min_len: 10
  alt:
    sigma_s: 6.0
    sigma_r: 20.0          # opensky26 tbl.3
    n_passes: 2
    tol_ftmin: 150.0
    min_len: 6
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002         # opensky26 tbl.3
    slope_tol: 1.2e-3      # opensky26 tbl.3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
"""
    )

    _make_delta_table(data_dir / "flights.delta", typecodes=["A320"])

    return config


class TestRunTraining:
    """Tests for ``run_training`` with trainer/loader mocked, real config + Delta on disk."""

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_train_config_from_cli(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CLI args override YAML defaults in TrainingConfig."""
        config = _make_config(tmp_path)

        mock_train_ds = MagicMock()
        mock_val_ds = MagicMock()
        mock_get_data.return_value = (mock_train_ds, mock_val_ds)

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        with nullcontext():
            run_training(
                arch="adsb",
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
        config = _make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with nullcontext():
            run_training(
                arch="adsb",
                config=config,
                typecode="A320",
                device="cpu",
            )

        # Only one call to trainer (A320, not B738)
        assert mock_trainer_cls.call_count == 1

    @pytest.mark.parametrize(
        "train_len, expected_epochs",
        [
            # batch_size=512, n_step=10, coeff=5.0, adjusted=4000
            pytest.param(5120, 4000, id="default_coefficient"),
            # batch_size=512, n_step=max(0,1)=1, coeff=min(50,10)=10, adjusted=8000
            pytest.param(10, 8000, id="capped_at_10x"),
        ],
    )
    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_epoch_adjustment(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
        train_len: int,
        expected_epochs: int,
    ) -> None:
        """Epochs scale by dataset-size coefficient (capped at 10x for tiny datasets)."""
        config = _make_config(tmp_path)

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=train_len)
        mock_get_data.return_value = (mock_train_ds, MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with nullcontext():
            run_training(
                arch="adsb",
                config=config,
                typecode="A320",
                device="cpu",
            )

        training_config = mock_trainer_cls.call_args.kwargs["config"]
        assert training_config.epochs == expected_epochs

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_cli_seq_len_param(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--seq-len CLI arg propagates to TrainingConfig.seq_len."""
        config = _make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with nullcontext():
            run_training(
                arch="adsb",
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
        config = _make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with nullcontext():
            run_training(
                arch="adsb",
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
        config = _make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with nullcontext():
            run_training(
                arch="adsb",
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
    def test_train_forwards_e1_cols(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run_training forwards the architecture's E1_COLS (incl. diff features) to the loader."""
        from node_fdm_models.schemas.adsb import E1_COLS

        config = _make_config(tmp_path)

        mock_get_data.return_value = (MagicMock(), MagicMock())
        mock_trainer_cls.return_value = MagicMock()

        with nullcontext():
            run_training(
                arch="adsb",
                config=config,
                typecode="A320",
                epochs=1,
                device="cpu",
            )

        data_kwargs = mock_get_data.call_args.kwargs
        assert data_kwargs.get("e1_cols") == E1_COLS
        assert "fdm_alt_diff_m" in data_kwargs["e1_cols"]
        assert "fdm_gamma_diff_rad" in data_kwargs["e1_cols"]


class TestRunTrainingMissingData:
    """Tests for ``run_training`` covering missing/empty data paths."""

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
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01          # opensky26 tbl.3
    n_passes: 2
    slope_tol: 3.0e-4      # opensky26 tbl.3
    flat_tol: 5.0e-2
    min_len: 15
  cas:
    cutoff_s: 180.0
    sigma_s: 8.0
    sigma_r: 3.0           # opensky26 tbl.3
    n_passes: 2
    slope_tol: 0.09        # opensky26 tbl.3
    flat_tol: 20.0
    min_len: 5
  vz:
    sigma_s: 6.0
    sigma_r: 100.0         # opensky26 tbl.3
    slope_tol: 50.0        # opensky26 tbl.3
    flat_tol: 100.0
    min_len: 10
  alt:
    sigma_s: 6.0
    sigma_r: 20.0          # opensky26 tbl.3
    n_passes: 2
    tol_ftmin: 150.0
    min_len: 6
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002         # opensky26 tbl.3
    slope_tol: 1.2e-3      # opensky26 tbl.3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
"""
        )

        with (
            pytest.raises(SystemExit),
        ):
            run_training(arch="adsb", config=config, device="cpu")

    @patch("node_fdm.trainer.ODETrainer")
    @patch("node_fdm.loader.get_train_val_data")
    def test_train_empty_dataset(
        self,
        mock_get_data: MagicMock,
        mock_trainer_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Empty dataset for typecode → warning, skip, no crash."""
        config = _make_config(tmp_path)

        with nullcontext():
            run_training(
                arch="adsb",
                config=config,
                typecode="B738",
                device="cpu",
            )

        mock_trainer_cls.assert_not_called()
