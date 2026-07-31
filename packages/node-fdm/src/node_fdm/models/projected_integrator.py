"""Projected fixed-step integrator — state-clamped ODE solvers.

Subclasses torchdiffeq's Euler and RK4 solvers to apply a projection
function after each integration step, keeping the state within
physically admissible bounds.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Any

import torch
from torchdiffeq._impl.fixed_grid import RK4, Euler
from torchdiffeq._impl.misc import _null_callback

__all__ = [
    "ClampedEuler",
    "ClampedRK4",
    "_clamp_columns",
    "_euler_clamped_integrate",
    "_soft_clamp_columns",
]


def _clamp_columns(
    x: torch.Tensor,
    bounds: dict[int, tuple[float, float]],
) -> torch.Tensor:
    """Hard-clamp selected columns of ``x`` to ``[lo, hi]``.

    Args:
        x: Tensor of shape ``(batch, dim)``.
        bounds: Mapping from column index to ``(lo, hi)`` bounds.

    Returns:
        Clamped tensor with the same shape as ``x``.
    """
    if not bounds:
        return x
    cols = []
    for c in range(x.shape[1]):
        if c in bounds:
            lo, hi = bounds[c]
            cols.append(x[:, c].clamp(min=lo, max=hi))
        else:
            cols.append(x[:, c])
    return torch.stack(cols, dim=1)


def _soft_clamp_columns(
    x: torch.Tensor,
    bounds: dict[int, tuple[float, float]],
) -> torch.Tensor:
    """Tanh-based soft clamp on selected columns.

    Maps values smoothly into the open interval ``(lo, hi)`` so that
    gradients never vanish (unlike hard clamp).

    Args:
        x: Tensor of shape ``(batch, dim)``.
        bounds: Mapping from column index to ``(lo, hi)`` bounds.

    Returns:
        Soft-clamped tensor with the same shape as ``x``.
    """
    if not bounds:
        return x
    cols = []
    for c in range(x.shape[1]):
        if c in bounds:
            lo, hi = bounds[c]
            mid = (lo + hi) / 2.0
            half_range = (hi - lo) / 2.0
            # Scale output by (1 - eps) so tanh saturation never reaches bounds.
            cols.append(mid + (half_range - 1e-7) * torch.tanh((x[:, c] - mid) / half_range))
        else:
            cols.append(x[:, c])
    return torch.stack(cols, dim=1)


class ClampedEuler(Euler):  # type: ignore[misc]
    """Euler solver with state projection after each step."""

    def __init__(
        self,
        func: Any,
        y0: torch.Tensor,
        *,
        project_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(func=func, y0=y0, **kwargs)
        self.project_fn = project_fn
        if not hasattr(self.func, "callback_step"):
            self.func.callback_step = _null_callback

    def _step_func(
        self,
        func: Any,
        t0: torch.Tensor,
        dt: torch.Tensor,
        t1: torch.Tensor,
        y0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Euler step without ``perturb`` kwarg (not needed for direct use)."""
        f0: torch.Tensor = func(t0, y0)
        return dt * f0, f0

    def integrate(self, t: torch.Tensor) -> torch.Tensor:
        """Integrate with optional projection after each step."""
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        y0 = self.project_fn(self.y0) if self.project_fn is not None else self.y0
        solution: list[torch.Tensor] = [y0]

        j = 1
        for t0, t1 in pairwise(time_grid):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            dy, _f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            if self.project_fn is not None:
                y1 = self.project_fn(y1)

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                else:
                    solution.append(y1)
                j += 1
            y0 = y1

        return torch.stack(solution)


class ClampedRK4(RK4):  # type: ignore[misc]
    """RK4 solver with state projection after each step."""

    def __init__(
        self,
        func: Any,
        y0: torch.Tensor,
        *,
        project_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(func=func, y0=y0, **kwargs)
        self.project_fn = project_fn
        if not hasattr(self.func, "callback_step"):
            self.func.callback_step = _null_callback

    def _step_func(
        self,
        func: Any,
        t0: torch.Tensor,
        dt: torch.Tensor,
        t1: torch.Tensor,
        y0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RK4 step without ``perturb`` kwarg (not needed for direct use)."""
        f0: torch.Tensor = func(t0, y0)
        half_dt = dt / 2
        k1 = f0
        k2 = func(t0 + half_dt, y0 + half_dt * k1)
        k3 = func(t0 + half_dt, y0 + half_dt * k2)
        k4 = func(t1, y0 + dt * k3)
        return dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), f0

    def integrate(self, t: torch.Tensor) -> torch.Tensor:
        """Integrate with optional projection after each step."""
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        y0 = self.project_fn(self.y0) if self.project_fn is not None else self.y0
        solution: list[torch.Tensor] = [y0]

        j = 1
        for t0, t1 in pairwise(time_grid):
            dt = t1 - t0
            self.func.callback_step(t0, y0, dt)
            dy, _f0 = self._step_func(self.func, t0, dt, t1, y0)
            y1 = y0 + dy

            if self.project_fn is not None:
                y1 = self.project_fn(y1)

            while j < len(t) and t1 >= t[j]:
                if self.interp == "linear":
                    solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                else:
                    solution.append(y1)
                j += 1
            y0 = y1

        return torch.stack(solution)


def _euler_clamped_integrate(
    func: Any,
    y0: torch.Tensor,
    t: torch.Tensor,
    project_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Convenience wrapper: Euler integration with optional state projection.

    Args:
        func: ODE right-hand side ``f(t, y)``.
        y0: Initial state.
        t: Time grid.
        project_fn: Optional projection applied after each step.
        **kwargs: Forwarded to :class:`ClampedEuler`.

    Returns:
        Solution tensor of shape ``(len(t), *y0.shape)``.
    """
    solver = ClampedEuler(func, y0, project_fn=project_fn, **kwargs)
    return solver.integrate(t)
