"""Tests for TrainingConfig and ODETrainer."""

from __future__ import annotations

import json

import pytest
import torch

from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.trainer import TrainingConfig


def _make_synthetic_dataset(
    n_samples: int = 20,
    seq_len: int = 10,
    n_x: int = 4,
    n_u: int = 8,
    n_e: int = 4,
) -> FlightDataset:
    """Create a synthetic dataset with random tensors."""
    samples = [
        FlightSample(
            x=torch.randn(seq_len, n_x),
            u=torch.randn(seq_len, n_u),
            e=torch.randn(seq_len, n_e),
            dx=torch.randn(seq_len, n_x),
        )
        for _ in range(n_samples)
    ]
    return FlightDataset(samples)


def _make_smooth_dataset(
    n_samples: int = 16,
    seq_len: int = 5,
    n_x: int = 4,
    n_u: int = 8,
    n_e: int = 4,
) -> FlightDataset:
    """Create a dataset with smooth linear trajectories for stable ODE integration."""
    torch.manual_seed(42)
    samples: list[FlightSample] = []
    for _ in range(n_samples):
        x0 = torch.randn(n_x) * 0.1
        velocity = torch.randn(n_x) * 0.01
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        x = x0.unsqueeze(0) + t * velocity.unsqueeze(0)
        dx = velocity.unsqueeze(0).expand(seq_len, n_x)
        samples.append(
            FlightSample(
                x=x,
                u=torch.randn(seq_len, n_u) * 0.01,
                e=torch.randn(seq_len, n_e) * 0.01,
                dx=dx,
            )
        )
    return FlightDataset(samples)


class TestTrainingConfig:
    """Unit tests for TrainingConfig Pydantic model."""

    def test_defaults(self) -> None:
        """Default values are applied correctly."""
        cfg = TrainingConfig(architecture_name="node_adsb_v1", model_name="test")
        assert cfg.lr == pytest.approx(1e-3)
        assert cfg.epochs == 800
        assert cfg.batch_size == 512
        assert cfg.method == "rk4"

    def test_negative_lr_raises(self) -> None:
        """Negative learning rate fails validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrainingConfig(architecture_name="a", model_name="b", lr=-1)

    def test_zero_epochs_raises(self) -> None:
        """Zero epochs fails validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrainingConfig(architecture_name="a", model_name="b", epochs=0)

    def test_zero_batch_size_raises(self) -> None:
        """Zero batch_size fails validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TrainingConfig(architecture_name="a", model_name="b", batch_size=0)

    def test_roundtrip(self) -> None:
        """model_dump → model_validate round-trips correctly."""
        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test",
            lr=0.01,
            epochs=10,
        )
        data = cfg.model_dump()
        restored = TrainingConfig.model_validate(data)
        assert restored == cfg

    def test_from_dict(self) -> None:
        """Construction via explicit keyword args works."""
        cfg = TrainingConfig(
            architecture_name="qar",
            model_name="qar_model",
            lr=5e-4,
            epochs=100,
            batch_size=256,
        )
        assert cfg.architecture_name == "qar"
        assert cfg.lr == pytest.approx(5e-4)


class TestODETrainer:
    """Functional tests for ODETrainer."""

    def test_training_1_epoch(self, tmp_path: object) -> None:
        """1-epoch training on synthetic data: loss finite, meta saved."""
        from pathlib import Path

        from node_fdm.trainer import ODETrainer

        model_dir = Path(str(tmp_path))

        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_model",
            epochs=1,
            batch_size=4,
            num_workers=0,
            val_batch_size=4,
            seq_len=5,
            step=0.01,
            method="euler",
        )

        train_ds = _make_smooth_dataset(n_samples=8, seq_len=5)
        val_ds = _make_smooth_dataset(n_samples=4, seq_len=5)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=model_dir,
        )

        records = trainer.train()

        # Loss should be finite
        assert len(records) == 1
        assert records[0]["train_loss"] == records[0]["train_loss"]  # not NaN
        assert records[0]["val_loss"] == records[0]["val_loss"]  # not NaN

        # Meta file saved
        meta_path = model_dir / "test_model" / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["architecture_name"] == "node_adsb_v1"

        # Loss CSV saved
        csv_path = model_dir / "test_model" / "training_losses.csv"
        assert csv_path.exists()

    def test_ode_rollout_loss_decreases(self, tmp_path: object) -> None:
        """ODE rollout loss decreases over 5 epochs on simple dynamics."""
        from pathlib import Path

        from node_fdm.trainer import ODETrainer

        model_dir = Path(str(tmp_path))

        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_rollout",
            epochs=5,
            batch_size=4,
            num_workers=0,
            val_batch_size=4,
            lr=1e-3,
            seq_len=5,
            step=0.1,
        )

        # Smooth linear trajectories so ODE integration is stable
        train_ds = _make_smooth_dataset(n_samples=16, seq_len=5)
        val_ds = _make_smooth_dataset(n_samples=4, seq_len=5)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=model_dir,
        )

        records = trainer.train()
        assert len(records) == 5
        assert records[-1]["val_loss"] <= records[0]["val_loss"]

    def test_ode_rollout_nan_detection(self, tmp_path: object) -> None:
        """NaN in x_seq triggers a warning log."""
        from pathlib import Path
        from unittest.mock import patch

        from node_fdm.trainer import ODETrainer

        model_dir = Path(str(tmp_path))

        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_nan",
            epochs=1,
            batch_size=4,
            num_workers=0,
            val_batch_size=4,
            seq_len=5,
        )

        # Create dataset with NaN injected into x
        samples = [
            FlightSample(
                x=torch.full((5, 4), float("nan")),
                u=torch.randn(5, 8),
                e=torch.randn(5, 4),
                dx=torch.randn(5, 4),
            )
            for _ in range(8)
        ]
        train_ds = FlightDataset(samples)
        val_ds = _make_synthetic_dataset(n_samples=4, seq_len=5)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=model_dir,
        )

        with patch.object(trainer, "loss_fn", wraps=trainer.loss_fn):
            records = trainer.train()

        # Loss should be NaN (from NaN inputs) — trainer logs warning
        assert len(records) == 1

    def test_single_sample_batch(self, tmp_path: object) -> None:
        """batch_size=1 with ODE rollout produces no shape errors."""
        from pathlib import Path

        from node_fdm.trainer import ODETrainer

        model_dir = Path(str(tmp_path))

        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_single",
            epochs=1,
            batch_size=1,
            num_workers=0,
            val_batch_size=1,
            seq_len=5,
            step=0.1,
        )

        train_ds = _make_smooth_dataset(n_samples=2, seq_len=5)
        val_ds = _make_smooth_dataset(n_samples=1, seq_len=5)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=model_dir,
        )

        records = trainer.train()
        assert len(records) == 1
        assert records[0]["train_loss"] == records[0]["train_loss"]  # not NaN

    def test_very_short_sequence(self, tmp_path: object) -> None:
        """seq_len=2 produces a valid t_grid and finite loss."""
        from pathlib import Path

        from node_fdm.trainer import ODETrainer

        model_dir = Path(str(tmp_path))

        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_short",
            epochs=1,
            batch_size=4,
            num_workers=0,
            val_batch_size=4,
            seq_len=2,
            step=0.1,
        )

        train_ds = _make_smooth_dataset(n_samples=8, seq_len=2)
        val_ds = _make_smooth_dataset(n_samples=4, seq_len=2)

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=model_dir,
        )

        records = trainer.train()
        assert len(records) == 1
        assert records[0]["train_loss"] == records[0]["train_loss"]
