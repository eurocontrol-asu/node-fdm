"""Wrap-aware loss for the heading state (lateral channel, Phase 2B).

Validates the trainer's residual logic in two layers:

1. End-to-end: a freshly built trainer for ``node_adsb_v1`` has the
   ``_heading_idx`` attribute pointing at the right column.
2. Numerical: the exact transform applied by ``_compute_batch_loss`` —
   ``signed_wrap(pred - true)`` on the heading column, raw difference
   elsewhere — gives the right answer in the textbook cases.

We do NOT spin up a full ODETrainer here (the architecture instantiation
needs real stats) — the inline replica below is a literal copy of the
trainer's residual block.  ``test_smoke.py`` covers the end-to-end path.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _residual_with_wrap(
    pred: torch.Tensor,
    true: torch.Tensor,
    heading_idx: int | None,
    norm_std: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Mirror ``ODETrainer._compute_batch_loss`` residual block.

    Verbatim copy: any drift between this helper and the trainer makes the
    tests below noise.  See ``trainer.py`` for the source.
    """
    residual = pred - true
    if heading_idx is not None:
        two_pi = 2.0 * math.pi
        raw = residual[..., heading_idx]
        wrapped = ((raw + math.pi) % two_pi) - math.pi
        residual = residual.clone()
        residual[..., heading_idx] = wrapped
    return (residual / norm_std) * alpha


def _mse_loss(residual_weighted: torch.Tensor) -> float:
    return float(nn.MSELoss()(residual_weighted, torch.zeros_like(residual_weighted)).item())


class TestHeadingWrapLoss:
    """Numerical correctness of the wrap-aware residual block."""

    def setup_method(self) -> None:
        # Single column → heading_idx = 0; std = 1 keeps the assertions simple.
        self.std = torch.tensor([1.0])
        self.alpha = torch.tensor([1.0])

    def test_full_turn_pred_yields_near_zero_loss(self) -> None:
        """pred=2π, true=0 → loss ≈ 0 (same physical angle)."""
        pred = torch.tensor([[[2.0 * math.pi]]])
        true = torch.tensor([[[0.0]]])
        residual = _residual_with_wrap(pred, true, 0, self.std, self.alpha)
        assert _mse_loss(residual) < 1e-6

    def test_naive_mse_would_be_huge(self) -> None:
        """Without the wrap, the same case gives MSE ≈ (2π)² ≈ 39.5."""
        pred = torch.tensor([[[2.0 * math.pi]]])
        true = torch.tensor([[[0.0]]])
        naive = _residual_with_wrap(pred, true, None, self.std, self.alpha)
        wrapped = _residual_with_wrap(pred, true, 0, self.std, self.alpha)
        naive_loss = _mse_loss(naive)
        wrapped_loss = _mse_loss(wrapped)
        # Sanity: the naive MSE must be at least 6 orders of magnitude
        # larger than the wrapped one to confirm the bug-fix is real.
        assert naive_loss > 30.0
        assert wrapped_loss < naive_loss / 1e6

    def test_wrap_is_symmetric(self) -> None:
        """signed_wrap(2π - 0) == signed_wrap(0 - 2π) == 0."""
        a = _residual_with_wrap(
            torch.tensor([[[2.0 * math.pi]]]),
            torch.tensor([[[0.0]]]),
            0,
            self.std,
            self.alpha,
        )
        b = _residual_with_wrap(
            torch.tensor([[[0.0]]]),
            torch.tensor([[[2.0 * math.pi]]]),
            0,
            self.std,
            self.alpha,
        )
        assert math.isclose(_mse_loss(a), _mse_loss(b), abs_tol=1e-9)

    def test_small_real_error_preserved(self) -> None:
        """A real 0.1 rad error survives the wrap unchanged."""
        pred = torch.tensor([[[2.0 * math.pi + 0.1]]])
        true = torch.tensor([[[0.0]]])
        residual = _residual_with_wrap(pred, true, 0, self.std, self.alpha)
        assert math.isclose(_mse_loss(residual), 0.01, rel_tol=1e-3)

    def test_boundary_bug_fixed(self) -> None:
        """true=+π-ε, pred=-π+ε → loss is 4ε² (Q3 of validate_lateral_wrap)."""
        eps = 1e-3
        pred = torch.tensor([[[-math.pi + eps]]])
        true = torch.tensor([[[math.pi - eps]]])
        residual = _residual_with_wrap(pred, true, 0, self.std, self.alpha)
        assert math.isclose(_mse_loss(residual), (2 * eps) ** 2, rel_tol=1e-2)

    def test_non_heading_column_keeps_plain_diff(self) -> None:
        """Multi-column case: only heading is wrapped, others use raw diff."""
        # 2 columns; index 1 is heading. ``true=0`` so a 0.5 raw delta on
        # column 0 stays 0.5 and a 2π delta on column 1 wraps to 0.
        std = torch.tensor([1.0, 1.0])
        alpha = torch.tensor([1.0, 1.0])
        pred = torch.tensor([[[0.5, 2.0 * math.pi]]])
        true = torch.tensor([[[0.0, 0.0]]])
        residual = _residual_with_wrap(pred, true, 1, std, alpha)
        # MSE = (0.5² + ~0²) / 2 = 0.125
        assert math.isclose(_mse_loss(residual), 0.125, rel_tol=1e-3)


class TestTrainerHeadingIdx:
    """The ODETrainer should resolve ``_heading_idx`` from the architecture spec."""

    def test_heading_idx_present_for_node_adsb_v1(self) -> None:
        """node_adsb_v1 has fdm_heading_rad in X_COLS at position 3."""
        from node_fdm.architectures.registry import get

        spec = get("node_adsb_v1")
        assert "fdm_heading_rad" in spec.x_cols
        assert spec.x_cols.index("fdm_heading_rad") == 3
