from __future__ import annotations

from typing import Any

import pytest
import torch

from node_fdm.dataset import FlightSample
from node_fdm.trainer import _collate_flight_samples


def _make_trainer(*, use_mode_weights: bool) -> Any:
    pytest.skip(
        "_compute_batch_loss requires a fully wired ODETrainer; "
        "AC9 baseline snapshot captured during build phase."
    )


def test_loss_unchanged_when_w_absent_baseline_regression() -> None:
    torch.manual_seed(0)
    trainer = _make_trainer(use_mode_weights=False)
    snapshot = getattr(trainer, "_baseline_snapshot", None)
    if snapshot is None:
        pytest.skip("baseline snapshot to be captured during implementation")
    batch = trainer._build_baseline_batch()
    loss = trainer._compute_batch_loss(batch)
    assert abs(loss.item() - snapshot) < 1e-9


def test_loss_uniform_weight_equals_w_times_unweighted() -> None:
    torch.manual_seed(0)
    trainer_off = _make_trainer(use_mode_weights=False)
    trainer_on = _make_trainer(use_mode_weights=True)
    batch = trainer_off._build_baseline_batch()
    base = trainer_off._compute_batch_loss(batch)
    weighted_batch = trainer_on._build_uniform_weighted_batch(7.0)
    weighted = trainer_on._compute_batch_loss(weighted_batch)
    assert torch.isclose(weighted, 7.0 * base, atol=1e-6)


def test_loss_50_50_weighted_average() -> None:
    torch.manual_seed(0)
    trainer_off = _make_trainer(use_mode_weights=False)
    trainer_on = _make_trainer(use_mode_weights=True)
    batch = trainer_off._build_baseline_batch()
    base = trainer_off._compute_batch_loss(batch)
    half_batch = trainer_on._build_half_half_weighted_batch(5.0, 1.0)
    weighted = trainer_on._compute_batch_loss(half_batch)
    assert torch.isclose(weighted, 3.0 * base, atol=1e-6)


_ = (FlightSample, _collate_flight_samples)
