"""Batch-compatible Neural ODE wrapper that interpolates inputs over time."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "BatchNeuralODE",
]


class BatchNeuralODE(nn.Module):
    """Wrap a neural ODE model with batched control and environment inputs.

    Interpolates ``u_seq`` and ``e_seq`` at arbitrary time ``t`` using
    linear interpolation on the provided ``t_grid``.
    """

    def __init__(
        self,
        model: nn.Module,
        u_seq: torch.Tensor,
        e_seq: torch.Tensor,
        t_grid: torch.Tensor,
    ) -> None:
        """Initialize the ODE wrapper and reset model history.

        Args:
            model: Base FDM model taking ``(x, u_t, e_t)``.
            u_seq: Control inputs ``(batch, time, n_u)``.
            e_seq: Environment inputs ``(batch, time, n_e)``.
            t_grid: Monotonic time grid ``(time,)``.
        """
        super().__init__()
        self.model = model
        if hasattr(self.model, "reset_history"):
            self.model.reset_history()  # type: ignore[operator]
        self.u_seq = u_seq
        self.e_seq = e_seq
        self.t_grid = t_grid

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluate dynamics at time ``t`` with linear interpolation.

        Args:
            t: Scalar tensor with the evaluation time.
            x: Current state tensor ``(batch, n_x)``.

        Returns:
            State derivatives at time ``t``.
        """
        t_val = float(t.item())
        idx_tensor = torch.searchsorted(
            self.t_grid, torch.tensor(t_val, device=self.t_grid.device)
        )
        idx = int(idx_tensor.item())
        idx0 = max(0, idx - 1)
        idx1 = min(idx, self.t_grid.shape[0] - 1)

        t0 = float(self.t_grid[idx0].item())
        t1 = float(self.t_grid[idx1].item())
        alpha = 0.0 if t1 == t0 else (t_val - t0) / (t1 - t0)

        u0, u1 = self.u_seq[:, idx0, :], self.u_seq[:, idx1, :]
        e0, e1 = self.e_seq[:, idx0, :], self.e_seq[:, idx1, :]

        u_t = (1 - alpha) * u0 + alpha * u1
        e_t = (1 - alpha) * e0 + alpha * e1

        result: torch.Tensor = self.model(x, u_t, e_t)
        return result
