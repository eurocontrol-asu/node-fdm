"""Tests for optimizer state checkpoint save/load and ModelMeta.optimizer_saved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.predictor import ModelMeta
from node_fdm.trainer import ODETrainer, TrainingConfig


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


def _make_trainer(tmp_path: Path) -> ODETrainer:
    """Create a minimal ODETrainer for testing."""
    cfg = TrainingConfig(
        architecture_name="node_adsb_v1",
        model_name="test_optim",
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
    return ODETrainer(
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestOptimizerCheckpoint:
    """Unit tests for optimizer state save/load."""

    def test_save_model_creates_optimizer_checkpoint(self, tmp_path: Path) -> None:
        """Train 1 epoch, call save_model(0) → optimizer.pt exists in model dir."""
        trainer = _make_trainer(tmp_path)
        trainer.train()
        trainer.save_model(0)

        optimizer_path = trainer.model_dir / "optimizer.pt"
        assert optimizer_path.exists(), "optimizer.pt should be created by save_model"

    def test_load_optimizer_state_restores(self, tmp_path: Path) -> None:
        """Save optimizer, create new trainer, load state → state dicts match."""
        trainer1 = _make_trainer(tmp_path)
        trainer1.train()
        trainer1.save_model(0)

        # Capture original optimizer state
        original_state = trainer1.optimizer.state_dict()

        # Create a fresh trainer (same config, same model_dir)
        trainer2 = _make_trainer(tmp_path)
        trainer2.load_optimizer_state()

        restored_state = trainer2.optimizer.state_dict()

        # Compare param_groups (hyperparams)
        assert len(restored_state["param_groups"]) == len(original_state["param_groups"])
        for orig_pg, rest_pg in zip(
            original_state["param_groups"], restored_state["param_groups"], strict=True
        ):
            assert orig_pg["lr"] == pytest.approx(rest_pg["lr"])

        # Compare state keys exist
        assert set(restored_state["state"].keys()) == set(original_state["state"].keys())

    def test_load_optimizer_state_missing_file(self, tmp_path: Path) -> None:
        """Call load_optimizer_state with no optimizer.pt → logs warning, does not raise."""
        trainer = _make_trainer(tmp_path)

        # No optimizer.pt exists — should not raise
        trainer.load_optimizer_state()

    def test_model_meta_optimizer_saved_field(self, tmp_path: Path) -> None:
        """Save meta after training → meta.json contains 'optimizer_saved': true."""
        trainer = _make_trainer(tmp_path)
        trainer.train()
        trainer.save_model(0)

        meta_path = trainer.model_dir / "meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta.get("optimizer_saved") is True


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestOptimizerFunctional:
    """Functional tests for optimizer checkpoint integration."""

    def test_existing_train_unchanged(self, tmp_path: Path) -> None:
        """Run ODETrainer.train() for 2 epochs → meta.json + layer checkpoints + optimizer.pt."""
        cfg = TrainingConfig(
            architecture_name="node_adsb_v1",
            model_name="test_func",
            epochs=2,
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
            model_dir=tmp_path,
        )
        records = trainer.train()

        model_dir = tmp_path / "test_func"
        assert len(records) == 2

        # meta.json exists
        assert (model_dir / "meta.json").exists()

        # Layer checkpoints exist
        layer_pts = list(model_dir.glob("*.pt"))
        assert len(layer_pts) >= 1, "At least one layer .pt file expected"

        # optimizer.pt exists
        assert (model_dir / "optimizer.pt").exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestOptimizerEdgeCases:
    """Edge-case tests for backward compatibility."""

    def test_legacy_checkpoint_no_optimizer(self, tmp_path: Path) -> None:
        """Load old model dir without optimizer.pt → load_optimizer_state returns gracefully."""
        trainer = _make_trainer(tmp_path)

        # Simulate legacy checkpoint: create model dir with meta but no optimizer.pt
        trainer.model_dir.mkdir(parents=True, exist_ok=True)

        # Should not raise — optimizer starts fresh
        trainer.load_optimizer_state()

    def test_model_meta_backward_compat(self, tmp_path: Path) -> None:
        """Load old meta.json without optimizer_saved → defaults to False."""
        meta_data = {
            "architecture_name": "node_adsb_v1",
            "model_params": [2, 1, 48],
            "step": 1.0,
            "shift": 60,
            "lr": 0.001,
            "seq_len": 60,
            "batch_size": 512,
            "method": "rk4",
            "stats_dict": {
                "col1": {"mean": 0.0, "std": 1.0, "max": 3.0},
            },
        }
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps(meta_data))

        meta = ModelMeta.from_json(meta_path)
        assert meta.optimizer_saved is False
