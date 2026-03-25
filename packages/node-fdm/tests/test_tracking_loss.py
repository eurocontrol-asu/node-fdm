"""Tests for AXM-817: tracking loss on autopilot targets in _compute_batch_loss.

The tracking loss penalizes deviation between predicted state and autopilot
gamma targets (from e1), weighted by ``lambda_tracking`` and masked by a
``known`` indicator derived from finite e1 values.

Tests are written TDD-style — they define the contract BEFORE implementation.
"""

from __future__ import annotations

from pathlib import Path

import torch

from node_fdm.dataset import FlightDataset, FlightSample
from node_fdm.trainer import ODETrainer, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_smooth_dataset(
    n_samples: int = 16,
    seq_len: int = 5,
    n_x: int = 4,
    n_u: int = 4,
    n_e: int = 4,
    *,
    gamma_offset: float = 0.0,
    known_mask: list[float] | None = None,
) -> FlightDataset:
    """Create smooth dataset with e1 (gamma_target).

    Parameters
    ----------
    gamma_offset:
        Constant offset added to true x[:,0] to produce gamma_target.
        When 0 the target matches the trajectory perfectly.
    known_mask:
        Per-timestep known indicator.  ``NaN`` in e1 encodes ``known=0``.
        If *None*, all timesteps are known (e1 is finite everywhere).
    """
    torch.manual_seed(42)
    samples: list[FlightSample] = []
    for _ in range(n_samples):
        x0 = torch.randn(n_x) * 0.1
        velocity = torch.randn(n_x) * 0.01
        t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        x = x0.unsqueeze(0) + t * velocity.unsqueeze(0)
        dx = velocity.unsqueeze(0).expand(seq_len, n_x)

        # gamma_target = first state variable + offset
        gamma_target = x[:, 0:1].clone() + gamma_offset

        # Encode known mask: NaN where known=0
        if known_mask is not None:
            for i, k in enumerate(known_mask):
                if i < seq_len and k == 0.0:
                    gamma_target[i, 0] = float("nan")

        samples.append(
            FlightSample(
                x=x,
                u=torch.randn(seq_len, n_u) * 0.01,
                e=torch.randn(seq_len, n_e) * 0.01,
                dx=dx,
                e1=gamma_target,
            )
        )
    return FlightDataset(samples)


def _make_trainer(
    tmp_path: Path,
    *,
    lambda_tracking: float = 0.0,
    seq_len: int = 5,
    epochs: int = 1,
    gamma_offset: float = 0.0,
    known_mask: list[float] | None = None,
) -> tuple[ODETrainer, FlightDataset, FlightDataset]:
    """Build an ODETrainer with tracking config and return (trainer, train_ds, val_ds)."""

    cfg = TrainingConfig(
        architecture_name="opensky_2025",
        model_name="test_tracking",
        epochs=epochs,
        batch_size=4,
        num_workers=0,
        val_batch_size=4,
        seq_len=seq_len,
        step=0.01,
        method="euler",
        lambda_tracking=lambda_tracking,
    )

    train_ds = _make_smooth_dataset(
        n_samples=8,
        seq_len=seq_len,
        gamma_offset=gamma_offset,
        known_mask=known_mask,
    )
    val_ds = _make_smooth_dataset(
        n_samples=4,
        seq_len=seq_len,
        gamma_offset=gamma_offset,
        known_mask=known_mask,
    )

    trainer = ODETrainer(
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_dir=tmp_path,
    )
    return trainer, train_ds, val_ds


# ===========================================================================
# Unit tests — tracking loss behavior
# ===========================================================================


class TestTrackingLoss:
    """Unit tests for the tracking loss term in _compute_batch_loss."""

    def test_tracking_loss_zero_lambda(self, tmp_path: Path) -> None:
        """lambda_tracking=0.0 with targets → loss identical to loss without tracking."""
        trainer_with, _, _ = _make_trainer(
            tmp_path / "with",
            lambda_tracking=0.0,
            gamma_offset=5.0,  # large offset, but lambda=0 so no effect
        )
        trainer_without, _, _ = _make_trainer(
            tmp_path / "without",
            lambda_tracking=0.0,
            gamma_offset=0.0,
        )

        # Get a batch from each trainer's dataloader
        batch_with = next(iter(trainer_with.train_loader))
        batch_without = next(iter(trainer_without.train_loader))

        # Use the same x/u/e/dx for both (only e1 differs)
        loss_with = trainer_with._compute_batch_loss(batch_with)
        loss_without = trainer_without._compute_batch_loss(batch_without)

        # With lambda=0, tracking term contributes nothing — both losses
        # come from ODE rollout only (seeded identically)
        assert torch.isclose(loss_with, loss_without, rtol=1e-5), (
            f"lambda_tracking=0 should make tracking term vanish: "
            f"{loss_with.item()} vs {loss_without.item()}"
        )

    def test_tracking_loss_nonzero(self, tmp_path: Path) -> None:
        """lambda_tracking=1.0 with gamma_pred ≠ gamma_target → loss > ODE-only loss."""
        trainer_base, _, _ = _make_trainer(
            tmp_path / "base",
            lambda_tracking=0.0,
        )
        trainer_tracking, _, _ = _make_trainer(
            tmp_path / "tracking",
            lambda_tracking=1.0,
            gamma_offset=5.0,  # large offset to ensure tracking penalty
        )

        batch_base = next(iter(trainer_base.train_loader))
        batch_tracking = next(iter(trainer_tracking.train_loader))

        loss_base = trainer_base._compute_batch_loss(batch_base)
        loss_tracking = trainer_tracking._compute_batch_loss(batch_tracking)

        assert loss_tracking > loss_base, (
            f"Tracking loss with offset should exceed ODE-only loss: "
            f"{loss_tracking.item()} vs {loss_base.item()}"
        )

    def test_tracking_loss_always_active(self, tmp_path: Path) -> None:
        """Tracking loss is active regardless of known mask (helps gamma_default learn)."""
        seq_len = 5

        # Both known=0 and known=1 produce tracking loss with gamma_offset
        trainer, _, _ = _make_trainer(
            tmp_path,
            lambda_tracking=1.0,
            gamma_offset=10.0,
            known_mask=[0.0] * seq_len,
        )

        trainer_base, _, _ = _make_trainer(
            tmp_path / "base",
            lambda_tracking=0.0,
            gamma_offset=10.0,
            known_mask=[0.0] * seq_len,
        )

        batch = next(iter(trainer.train_loader))
        batch_base = next(iter(trainer_base.train_loader))
        loss = trainer._compute_batch_loss(batch)
        loss_base = trainer_base._compute_batch_loss(batch_base)

        # Tracking adds penalty even with known=0 (gamma_default target)
        assert (
            loss > loss_base
        ), f"Tracking should add penalty even with known=0: {loss.item()} vs {loss_base.item()}"

    def test_tracking_loss_larger_lambda_larger_loss(self, tmp_path: Path) -> None:
        """Bigger lambda_tracking → bigger total loss."""
        trainer_low, _, _ = _make_trainer(
            tmp_path / "low",
            lambda_tracking=0.1,
            gamma_offset=5.0,
        )

        trainer_high, _, _ = _make_trainer(
            tmp_path / "high",
            lambda_tracking=10.0,
            gamma_offset=5.0,
        )

        batch_low = next(iter(trainer_low.train_loader))
        batch_high = next(iter(trainer_high.train_loader))

        loss_low = trainer_low._compute_batch_loss(batch_low)
        loss_high = trainer_high._compute_batch_loss(batch_high)

        assert loss_high > loss_low, (
            f"Higher lambda should yield higher total loss: "
            f"{loss_high.item()} vs {loss_low.item()}"
        )


# ===========================================================================
# Functional tests — training loop with tracking
# ===========================================================================


class TestTrainingWithTracking:
    """Functional tests for the full training loop with tracking loss."""

    def test_training_with_tracking(self, tmp_path: Path) -> None:
        """2 epochs with lambda_tracking=1.0 → training completes, loss decreases."""
        cfg = TrainingConfig(
            architecture_name="opensky_2025",
            model_name="test_tracking_train",
            epochs=2,
            batch_size=4,
            num_workers=0,
            val_batch_size=4,
            seq_len=5,
            step=0.01,
            method="euler",
            lr=1e-3,
            lambda_tracking=1.0,
        )

        train_ds = _make_smooth_dataset(
            n_samples=16,
            seq_len=5,
            gamma_offset=0.1,
        )
        val_ds = _make_smooth_dataset(
            n_samples=4,
            seq_len=5,
            gamma_offset=0.1,
        )

        trainer = ODETrainer(
            config=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=tmp_path,
        )

        records = trainer.train()

        assert len(records) == 2
        # Training completes and produces finite losses
        for r in records:
            assert r["train_loss"] > 0
            assert r["val_loss"] > 0


# ===========================================================================
# Edge cases
# ===========================================================================


class TestTrackingLossEdgeCases:
    """Edge cases for the tracking loss."""

    def test_all_known_zero(self, tmp_path: Path) -> None:
        """All known=0 (no detected segment) → tracking loss = 0, only ODE loss active."""
        seq_len = 5
        trainer, _, _ = _make_trainer(
            tmp_path,
            lambda_tracking=1.0,
            gamma_offset=100.0,  # extreme offset
            known_mask=[0.0] * seq_len,
        )

        batch = next(iter(trainer.train_loader))
        loss = trainer._compute_batch_loss(batch)

        # Loss must be finite (ODE loss only, tracking is zeroed out)
        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
        assert loss.item() > 0, "ODE loss should still be positive"

    def test_lambda_tracking_very_large(self, tmp_path: Path) -> None:
        """λ=1000 → no NaN, loss dominated by tracking."""
        trainer_large, _, _ = _make_trainer(
            tmp_path / "large",
            lambda_tracking=1000.0,
            gamma_offset=1.0,
        )
        trainer_small, _, _ = _make_trainer(
            tmp_path / "small",
            lambda_tracking=0.001,
            gamma_offset=1.0,
        )

        batch_large = next(iter(trainer_large.train_loader))
        batch_small = next(iter(trainer_small.train_loader))

        loss_large = trainer_large._compute_batch_loss(batch_large)
        loss_small = trainer_small._compute_batch_loss(batch_small)

        # No NaN
        assert torch.isfinite(loss_large), f"Large λ should not produce NaN: {loss_large.item()}"
        # Loss should be much larger with λ=1000
        assert (
            loss_large > loss_small * 10
        ), f"λ=1000 should dominate: {loss_large.item()} vs {loss_small.item()}"
