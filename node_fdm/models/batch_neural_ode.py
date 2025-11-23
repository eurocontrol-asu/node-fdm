#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-compatible Neural ODE wrapper that interpolates inputs over time."""

from typing import Any

import torch
import torch.nn as nn


class BatchNeuralODE(nn.Module):
    """Wrap a neural ODE with batched control and environment inputs."""

    def __init__(
        self,
        model: nn.Module,
        u_seq: torch.Tensor,
        e_seq: torch.Tensor,
        t_grid: torch.Tensor,
    ) -> None:
        """Initialize the ODE wrapper and reset model history.

        Args:
            model: Base neural ODE model taking `(x, u_t, e_t)`.
            u_seq: Control inputs over time with shape `(batch, time, features)`.
            e_seq: Environment inputs over time with shape `(batch, time, features)`.
            t_grid: Monotonic time grid corresponding to `u_seq` and `e_seq`.
        """
        super().__init__()
        self.model = model
        self.model.reset_history()
        self.u_seq = u_seq
        self.e_seq = e_seq
        self.t_grid = t_grid

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> Any:
        """Evaluate the ODE dynamics at time `t` with linear interpolation.

        Args:
            t: Scalar tensor containing the evaluation time.
            x: Current state tensor.

        Returns:
            Model output of the wrapped dynamics at time `t`.
        """
        t = t.item()
        idx = torch.searchsorted(self.t_grid, torch.tensor(t, device=self.t_grid.device)).item()
        idx0 = max(0, idx - 1)
        idx1 = min(idx, self.t_grid.shape[0] - 1)

        t0, t1 = self.t_grid[idx0].item(), self.t_grid[idx1].item()
        alpha = 0 if t1 == t0 else (t - t0) / (t1 - t0)

        u0, u1 = self.u_seq[:, idx0, :], self.u_seq[:, idx1, :]
        e0, e1 = self.e_seq[:, idx0, :], self.e_seq[:, idx1, :]

        u_t = (1 - alpha) * u0 + alpha * u1
        e_t = (1 - alpha) * e0 + alpha * e1

        return self.model(x, u_t, e_t)
