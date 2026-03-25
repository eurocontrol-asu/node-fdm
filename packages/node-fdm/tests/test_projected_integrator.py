"""Tests for projected fixed-step integrator — clamp utilities and projected solvers."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from node_fdm.models.projected_integrator import (
    ClampedEuler,
    ClampedRK4,
    _clamp_columns,
    _soft_clamp_columns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstDeriv(nn.Module):
    """Model that returns a constant derivative, ignoring inputs."""

    def __init__(self, dx: torch.Tensor) -> None:
        super().__init__()
        self.dx = dx

    def forward(self, t: torch.Tensor, y: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.dx.expand_as(y)

    def callback_step(self, t0: torch.Tensor, y0: torch.Tensor, dt: torch.Tensor) -> None:
        pass


class _LinearDeriv(nn.Module):
    """Model whose derivative is proportional to state: f(t, y) = scale * y."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, t: torch.Tensor, y: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.scale * y

    def callback_step(self, t0: torch.Tensor, y0: torch.Tensor, dt: torch.Tensor) -> None:
        pass


def _make_solver(
    func: nn.Module,
    y0: torch.Tensor,
    *,
    method: str = "euler",
    project_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> ClampedEuler | ClampedRK4:
    """Instantiate a projected solver directly."""
    cls = ClampedEuler if method == "euler" else ClampedRK4
    return cls(func=func, y0=y0, project_fn=project_fn, atol=1e-9)


# ---------------------------------------------------------------------------
# Unit tests — _clamp_columns
# ---------------------------------------------------------------------------


class TestClampColumns:
    def test_clamp_columns_basic(self) -> None:
        """Tensor with values outside bounds -> output within bounds, shape preserved."""
        x = torch.tensor([[-10.0, 50.0, 0.5], [20.0, -3.0, 1.5]])
        bounds = {0: (-5.0, 5.0), 1: (0.0, 10.0)}

        out = _clamp_columns(x, bounds)

        assert out.shape == x.shape
        assert out[:, 0].min() >= -5.0
        assert out[:, 0].max() <= 5.0
        assert out[:, 1].min() >= 0.0
        assert out[:, 1].max() <= 10.0
        # Column 2 has no bounds -> unchanged
        assert torch.equal(out[:, 2], x[:, 2])

    def test_clamp_columns_no_bounds(self) -> None:
        """Empty bounds dict -> output equals input."""
        x = torch.randn(4, 3)
        out = _clamp_columns(x, {})
        assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# Unit tests — _soft_clamp_columns
# ---------------------------------------------------------------------------


class TestSoftClampColumns:
    def test_soft_clamp_columns_range(self) -> None:
        """Output strictly within (lo, hi), never exactly at bound."""
        x = torch.tensor([[-1000.0, 1000.0], [0.0, 0.0]])
        bounds = {0: (-1.0, 1.0), 1: (-1.0, 1.0)}

        out = _soft_clamp_columns(x, bounds)

        assert (out[:, 0] > -1.0).all()
        assert (out[:, 0] < 1.0).all()
        assert (out[:, 1] > -1.0).all()
        assert (out[:, 1] < 1.0).all()

    def test_soft_clamp_columns_gradient(self) -> None:
        """Tensor at bounds, .backward() -> gradient is non-zero everywhere."""
        x = torch.tensor([[-1.0, 1.0], [0.0, 0.5]], requires_grad=True)
        bounds = {0: (-1.0, 1.0), 1: (-1.0, 1.0)}

        out = _soft_clamp_columns(x, bounds)
        out.sum().backward()  # type: ignore[no-untyped-call]

        assert x.grad is not None
        assert (x.grad != 0).all(), "Gradient must be non-zero everywhere for soft clamp"


# ---------------------------------------------------------------------------
# Unit tests — Projected integration
# ---------------------------------------------------------------------------


class TestProjectedIntegration:
    def _integrate(
        self,
        func: nn.Module,
        y0: torch.Tensor,
        t: torch.Tensor,
        *,
        method: str = "euler",
        project_fn: Callable[[torch.Tensor], torch.Tensor] | None | str = "default",
    ) -> torch.Tensor:
        """Run projected integration via direct solver instantiation."""
        bounds = {0: (-1.0, 1.0), 1: (-2.0, 2.0), 2: (-3.0, 3.0)}

        actual_fn: Callable[[torch.Tensor], torch.Tensor] | None
        if project_fn == "default":
            actual_fn = lambda y: _clamp_columns(y, bounds)  # noqa: E731
        elif isinstance(project_fn, str):
            actual_fn = None
        else:
            actual_fn = project_fn

        solver = _make_solver(func, y0, method=method, project_fn=actual_fn)
        return solver.integrate(t)

    def test_projected_euler_stays_bounded(self) -> None:
        """Random model, 60 steps, tight bounds -> all states in trajectory within bounds."""
        torch.manual_seed(42)
        func = _LinearDeriv(scale=5.0)  # aggressive derivative to force divergence
        y0 = torch.randn(4, 3)  # batch=4, state_dim=3
        t = torch.linspace(0, 1, 61)  # 60 steps

        traj = self._integrate(func, y0, t, method="euler")

        # traj shape: (time, batch, state_dim)
        assert (traj[..., 0] >= -1.0 - 1e-6).all()
        assert (traj[..., 0] <= 1.0 + 1e-6).all()
        assert (traj[..., 1] >= -2.0 - 1e-6).all()
        assert (traj[..., 1] <= 2.0 + 1e-6).all()
        assert (traj[..., 2] >= -3.0 - 1e-6).all()
        assert (traj[..., 2] <= 3.0 + 1e-6).all()

    def test_projected_rk4_stays_bounded(self) -> None:
        """Same as above with RK4 -> all states within bounds."""
        torch.manual_seed(42)
        func = _LinearDeriv(scale=5.0)
        y0 = torch.randn(4, 3)
        t = torch.linspace(0, 1, 61)

        traj = self._integrate(func, y0, t, method="rk4")

        assert (traj[..., 0] >= -1.0 - 1e-6).all()
        assert (traj[..., 0] <= 1.0 + 1e-6).all()
        assert (traj[..., 1] >= -2.0 - 1e-6).all()
        assert (traj[..., 1] <= 2.0 + 1e-6).all()
        assert (traj[..., 2] >= -3.0 - 1e-6).all()
        assert (traj[..., 2] <= 3.0 + 1e-6).all()

    def test_projected_euler_no_project(self) -> None:
        """project_fn=None -> identical to standard Euler."""
        from torchdiffeq._impl.fixed_grid import Euler

        torch.manual_seed(7)
        func = _LinearDeriv(scale=0.5)
        y0 = torch.randn(2, 3)
        t = torch.linspace(0, 1, 11)

        traj_projected = self._integrate(func, y0, t, method="euler", project_fn=None)

        standard = Euler(func=func, y0=y0, atol=1e-9)
        traj_standard = standard.integrate(t)

        assert torch.allclose(traj_projected, traj_standard, atol=1e-6)

    def test_autograd_through_projection(self) -> None:
        """Forward + backward through projected Euler -> no RuntimeError, finite gradients."""

        class _DiffFunc(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(3, 3, bias=False)

            def forward(self, t: torch.Tensor, y: torch.Tensor, **kwargs: object) -> torch.Tensor:
                result: torch.Tensor = self.linear(y)
                return result

            def callback_step(self, t0: torch.Tensor, y0: torch.Tensor, dt: torch.Tensor) -> None:
                pass

        func = _DiffFunc()
        y0 = torch.randn(2, 3, requires_grad=True)
        t = torch.linspace(0, 1, 11)

        bounds = {0: (-5.0, 5.0), 1: (-5.0, 5.0), 2: (-5.0, 5.0)}
        project_fn = lambda y: _clamp_columns(y, bounds)  # noqa: E731

        solver = _make_solver(func, y0, method="euler", project_fn=project_fn)
        traj = solver.integrate(t)
        loss = traj.sum()
        loss.backward()  # type: ignore[no-untyped-call]

        assert y0.grad is not None
        assert torch.isfinite(y0.grad).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestProjectedEdgeCases:
    def test_state_already_in_bounds(self) -> None:
        """State already within [lo, hi] everywhere -> project_fn is identity, no effect."""
        x = torch.tensor([[0.0, 0.0, 0.0]])
        bounds = {0: (-1.0, 1.0), 1: (-2.0, 2.0), 2: (-3.0, 3.0)}

        out = _clamp_columns(x, bounds)
        assert torch.equal(out, x)

    def test_single_step_integration(self) -> None:
        """seq_len=2, one step -> projection applied once, correct output."""
        dx = torch.tensor([10.0, 0.0, 0.0])  # large step in col 0 only
        func = _ConstDeriv(dx)
        y0 = torch.zeros(1, 3)
        t = torch.tensor([0.0, 1.0])  # single step

        bounds = {0: (-1.0, 1.0)}
        project_fn = lambda y: _clamp_columns(y, bounds)  # noqa: E731

        solver = _make_solver(func, y0, method="euler", project_fn=project_fn)
        traj = solver.integrate(t)

        # After one Euler step: y = 0 + 1.0 * 10 = 10 -> clamped to 1.0
        assert traj.shape == (2, 1, 3)
        assert torch.isclose(traj[1, 0, 0], torch.tensor(1.0), atol=1e-6)

    def test_batch_size_one(self) -> None:
        """Single sample batch -> works without dimension errors."""
        func = _LinearDeriv(scale=1.0)
        y0 = torch.randn(1, 3)
        t = torch.linspace(0, 1, 11)

        bounds = {0: (-1.0, 1.0), 1: (-1.0, 1.0), 2: (-1.0, 1.0)}
        project_fn = lambda y: _clamp_columns(y, bounds)  # noqa: E731

        solver = _make_solver(func, y0, method="euler", project_fn=project_fn)
        traj = solver.integrate(t)

        assert traj.shape == (11, 1, 3)
        assert (traj[..., 0] >= -1.0 - 1e-6).all()
        assert (traj[..., 0] <= 1.0 + 1e-6).all()
