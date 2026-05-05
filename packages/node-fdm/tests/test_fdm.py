"""Functional tests for FlightDynamicsModel with real architecture specs."""

from __future__ import annotations

import torch

from node_fdm.architectures.registry import get
from node_fdm.models.fdm import FlightDynamicsModel


def _make_stats(cols: list[str]) -> dict[str, dict[str, float]]:
    """Create dummy stats dict for all columns."""
    return {col: {"mean": 0.0, "std": 1.0, "max": 1.0, "p999": 0.8} for col in cols}


class TestFDMOpenSky:
    """FlightDynamicsModel with OpenSky 2025 spec."""

    def test_forward_pass(self) -> None:
        """Forward pass returns tensor of correct shape (AC11)."""
        spec = get("node_adsb_v1")
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols
        all_cols += [col for _, col in spec.dx_cols]
        stats = _make_stats(all_cols)

        model = FlightDynamicsModel(spec, stats)
        batch = 4
        x = torch.randn(batch, len(spec.x_cols))
        u = torch.randn(batch, len(spec.u_cols))
        e = torch.randn(batch, len(spec.e0_cols))

        model.reset_history()
        out = model(x, u, e)
        assert out.shape == (batch, len(spec.dx_cols))

    def test_no_nan_output(self) -> None:
        """Forward pass produces no NaN values."""
        spec = get("node_adsb_v1")
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols
        all_cols += [col for _, col in spec.dx_cols]
        stats = _make_stats(all_cols)

        model = FlightDynamicsModel(spec, stats)
        x = torch.randn(4, len(spec.x_cols))
        u = torch.randn(4, len(spec.u_cols))
        e = torch.randn(4, len(spec.e0_cols))

        model.reset_history()
        out = model(x, u, e)
        assert not torch.isnan(out).any()


class TestFDMQAR:
    """FlightDynamicsModel with QAR spec."""

    def test_forward_pass(self) -> None:
        """Forward pass returns tensor of correct shape (AC11)."""
        spec = get("qar")
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols
        all_cols += [col for _, col in spec.dx_cols]
        stats = _make_stats(all_cols)

        model = FlightDynamicsModel(spec, stats)
        batch = 4
        x = torch.randn(batch, len(spec.x_cols))
        u = torch.randn(batch, len(spec.u_cols))
        e = torch.randn(batch, len(spec.e0_cols))

        model.reset_history()
        out = model(x, u, e)
        assert out.shape == (batch, len(spec.dx_cols))

    def test_no_nan_output(self) -> None:
        """Forward pass produces no NaN values."""
        spec = get("qar")
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols
        all_cols += [col for _, col in spec.dx_cols]
        stats = _make_stats(all_cols)

        model = FlightDynamicsModel(spec, stats)
        x = torch.randn(4, len(spec.x_cols))
        u = torch.randn(4, len(spec.u_cols))
        e = torch.randn(4, len(spec.e0_cols))

        model.reset_history()
        out = model(x, u, e)
        assert not torch.isnan(out).any()


class TestFDMEdgeCases:
    """Edge case tests for FlightDynamicsModel."""

    def test_empty_architecture(self) -> None:
        """Spec with 0 layers: model instantiates, forward is identity-ish."""
        from node_fdm.architectures.registry import ArchitectureSpec

        spec = ArchitectureSpec(
            name="empty",
            x_cols=["x1", "x2"],
            u_cols=["u1"],
            e0_cols=["e1"],
            e1_cols=[],
            dx_cols=[(1, "x1"), (1, "x2")],
            layers=[],
        )
        stats = _make_stats(["x1", "x2", "u1", "e1"])
        model = FlightDynamicsModel(spec, stats)
        model.reset_history()

        x = torch.randn(2, 2)
        u = torch.randn(2, 1)
        e = torch.randn(2, 1)
        out = model(x, u, e)
        # With no layers, dx output just reads from vect_dict (the raw input)
        assert out.shape == (2, 2)
