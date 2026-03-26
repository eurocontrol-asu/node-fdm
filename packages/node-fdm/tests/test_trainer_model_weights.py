"""Tests for model weights checkpoint load on resume (AXM-780).

The ``load_model_weights`` method mirrors ``load_optimizer_state``:
iterate over layer ``.pt`` files written by ``save_layer_checkpoint``,
restore each layer's ``state_dict``, and update ``best_val_loss``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.trainer import ODETrainer, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers (same pattern as test_trainer_optimizer)
# ---------------------------------------------------------------------------


def _make_smooth_dataset(
    n_samples: int = 16,
    seq_len: int = 5,
    n_x: int = 4,
    n_u: int = 4,
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


def _make_trainer(tmp_path: Path, *, epochs: int = 1) -> ODETrainer:
    """Create a minimal ODETrainer for testing."""
    cfg = TrainingConfig(
        architecture_name="opensky_2025",
        model_name="test_weights",
        epochs=epochs,
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


class TestLoadModelWeights:
    """Unit tests for load_model_weights — the missing counterpart to save_layer_checkpoint."""

    def test_load_model_weights_restores_state(self, tmp_path: Path) -> None:
        """Train 1 epoch, save, create new trainer, load weights → layer state dicts match."""
        trainer1 = _make_trainer(tmp_path)
        trainer1.train()
        trainer1.save_model(0)

        # Capture original layer states
        original_states: dict[str, dict[str, torch.Tensor]] = {}
        for name in trainer1.model.layers_name:
            layer = trainer1.model.layers_dict[name]
            original_states[name] = {k: v.clone() for k, v in layer.state_dict().items()}

        # Fresh trainer — weights are randomly initialised, different from trained
        trainer2 = _make_trainer(tmp_path)
        trainer2.load_model_weights()

        for name in trainer2.model.layers_name:
            layer = trainer2.model.layers_dict[name]
            restored = layer.state_dict()
            for key, orig_tensor in original_states[name].items():
                assert torch.equal(restored[key], orig_tensor), (
                    f"Layer {name!r} param {key!r} not restored correctly"
                )

    def test_load_model_weights_missing_checkpoint(self, tmp_path: Path) -> None:
        """Call load_model_weights with no .pt files → raises FileNotFoundError or logs warning."""
        trainer = _make_trainer(tmp_path)

        # Remove any layer .pt files that __init__/save_meta may have created
        for pt in trainer.model_dir.glob("*.pt"):
            if pt.name != "optimizer.pt":
                pt.unlink()

        with pytest.raises((FileNotFoundError, SystemExit)):
            trainer.load_model_weights()


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


class TestModelWeightsFunctional:
    """Functional tests for weight loading during resume."""

    def test_resume_loss_continuity(self, tmp_path: Path) -> None:
        """Train 2 epochs (val_loss=L), save, resume 1 epoch → first val_loss ≈ L."""
        trainer1 = _make_trainer(tmp_path, epochs=2)
        records1 = trainer1.train()
        last_val_loss = records1[-1]["val_loss"]
        trainer1.save_model(1)

        # Resume: new trainer, load both weights and optimizer
        trainer2 = _make_trainer(tmp_path, epochs=1)
        trainer2.load_model_weights()
        trainer2.load_optimizer_state()
        records2 = trainer2.train()

        resumed_val_loss = records2[0]["val_loss"]

        # The resumed loss should be in the same ballpark -- not 100x higher
        # (which would indicate weights were not loaded).
        assert resumed_val_loss < last_val_loss * 10, (
            f"Resumed val_loss {resumed_val_loss:.4f} is >10x the saved "
            f"val_loss {last_val_loss:.4f} -- weights were likely not loaded"
        )

    def test_train_unchanged(self, tmp_path: Path) -> None:
        """Run train 2 epochs → same artifacts as before (no regression)."""
        cfg = TrainingConfig(
            architecture_name="opensky_2025",
            model_name="test_unchanged",
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

        model_dir = tmp_path / "test_unchanged"
        assert len(records) == 2
        assert (model_dir / "meta.json").exists()

        layer_pts = list(model_dir.glob("*.pt"))
        assert len(layer_pts) >= 1, "At least one layer .pt file expected"
        assert (model_dir / "optimizer.pt").exists()
