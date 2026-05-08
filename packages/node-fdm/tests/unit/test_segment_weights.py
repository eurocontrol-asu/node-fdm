from __future__ import annotations

import torch

from node_fdm.training.weighting import compute_segment_weights


def test_compute_segment_weights_returns_per_batch_mean() -> None:
    w = torch.tensor([[1.0] * 4 + [3.0] * 2, [2.0] * 6])
    out = compute_segment_weights(w)
    assert out.shape == (2,)
    expected = torch.tensor([5 / 3, 2.0])
    assert torch.allclose(out, expected, atol=1e-6)


def test_compute_segment_weights_uniform_segment_returns_that_constant() -> None:
    w = torch.full((4, 60), 7.0)
    out = compute_segment_weights(w)
    assert torch.allclose(out, torch.full((4,), 7.0))


def test_compute_segment_weights_50_50_split_returns_arithmetic_mean() -> None:
    w = torch.cat([torch.full((1, 30), 5.0), torch.full((1, 30), 1.0)], dim=1)
    out = compute_segment_weights(w)
    assert torch.allclose(out, torch.tensor([3.0]))
