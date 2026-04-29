"""Tests for BatchNeuralODE — time interpolation and ODE wrapper."""

from __future__ import annotations

import torch

from node_fdm.models.batch_neural_ode import BatchNeuralODE


class TestBatchNeuralODE:
    """Tests for the batched Neural ODE wrapper."""

    def _dummy_model(self) -> torch.nn.Module:
        """Create a simple model: f(x, u, e) = x (ignores u, e)."""

        class DummyModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
                return x

        return DummyModel()

    def _model_with_history(self) -> torch.nn.Module:
        """Model with reset_history method."""

        class HistoryModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.history_reset = False

            def reset_history(self) -> None:
                self.history_reset = True

            def forward(self, x: torch.Tensor, u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
                return x

        return HistoryModel()

    def test_forward_shape(self) -> None:
        """Output shape matches input batch x n_x."""
        batch, time, n_x, n_u, n_e = 4, 10, 3, 2, 1
        model = self._dummy_model()
        u_seq = torch.randn(batch, time, n_u)
        e_seq = torch.randn(batch, time, n_e)
        t_grid = torch.linspace(0, 1, time)

        ode = BatchNeuralODE(model, u_seq, e_seq, t_grid)
        x = torch.randn(batch, n_x)
        t = torch.tensor(0.5)
        out = ode(t, x)
        assert out.shape == (batch, n_x)

    def test_interpolation_at_grid_point(self) -> None:
        """At a grid point, interpolation returns exact values."""
        batch, time, n_u, n_e = 2, 5, 1, 1
        u_seq = (
            torch.arange(time, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch, -1, n_u)
        )
        e_seq = torch.zeros(batch, time, n_e)
        t_grid = torch.linspace(0, 1, time)

        model = self._dummy_model()
        ode = BatchNeuralODE(model, u_seq, e_seq, t_grid)

        x = torch.zeros(batch, 1)
        # At t=0 (first grid point), u should be 0
        out = ode(torch.tensor(0.0), x)
        assert out.shape == (batch, 1)

    def test_u_zero_order_hold_midpoint(self) -> None:
        """F3: ``u`` is held constant at the left grid point (ZOH).

        Linear interpolation of commanded inputs turned a hard target
        step into a multi-second ramp at inference; ``u`` must now match
        the discrete training semantics. ``e`` keeps linear interp.
        """
        batch, n_e = 1, 1

        class AddUModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
                return u  # return u directly

        u_seq = torch.tensor([[[0.0], [10.0]]])  # (1, 2, 1)
        e_seq = torch.zeros(batch, 2, n_e)
        t_grid = torch.tensor([0.0, 1.0])

        ode = BatchNeuralODE(AddUModel(), u_seq, e_seq, t_grid)
        x = torch.zeros(batch, 1)
        # At t=0.5, u should equal u_seq[:, idx0=0, :] = 0.0 (zero-order-hold).
        out = ode(torch.tensor(0.5), x)
        assert torch.isclose(out, torch.tensor([[0.0]]), atol=1e-5).all()

    def test_e_linear_interpolation_midpoint(self) -> None:
        """Environment ``e`` keeps linear interpolation (continuous physics)."""
        batch, n_u = 1, 1

        class AddEModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
                return e  # return e directly

        u_seq = torch.zeros(batch, 2, n_u)
        e_seq = torch.tensor([[[0.0], [10.0]]])  # (1, 2, 1)
        t_grid = torch.tensor([0.0, 1.0])

        ode = BatchNeuralODE(AddEModel(), u_seq, e_seq, t_grid)
        x = torch.zeros(batch, 1)
        out = ode(torch.tensor(0.5), x)
        assert torch.isclose(out, torch.tensor([[5.0]]), atol=1e-5).all()

    def test_reset_history_called(self) -> None:
        """Model with reset_history has it called on init."""
        model = self._model_with_history()
        assert not model.history_reset
        BatchNeuralODE(
            model,
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 1),
            torch.tensor([0.0, 1.0]),
        )
        assert model.history_reset

    def test_boundary_t_values(self) -> None:
        """At t=0 and t=1 (boundaries), forward works without error."""
        batch, time, n_x, n_u, n_e = 2, 5, 2, 1, 1
        model = self._dummy_model()
        ode = BatchNeuralODE(
            model,
            torch.randn(batch, time, n_u),
            torch.randn(batch, time, n_e),
            torch.linspace(0, 1, time),
        )
        x = torch.randn(batch, n_x)
        # Both boundaries should work
        ode(torch.tensor(0.0), x)
        ode(torch.tensor(1.0), x)

    def test_same_t0_t1(self) -> None:
        """When t0 == t1, alpha defaults to 0 (no division by zero)."""
        batch, n_e = 1, 1

        class AddUModel(torch.nn.Module):
            def forward(self, x: torch.Tensor, u: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
                return u

        ode = BatchNeuralODE(
            AddUModel(),
            torch.tensor([[[5.0]]]),
            torch.zeros(batch, 1, n_e),
            torch.tensor([0.5]),
        )
        x = torch.zeros(batch, 1)
        out = ode(torch.tensor(0.5), x)
        assert torch.isclose(out, torch.tensor([[5.0]]), atol=1e-5).all()
