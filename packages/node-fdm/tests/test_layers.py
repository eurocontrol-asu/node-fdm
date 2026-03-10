"""Tests for neural network layers."""

from __future__ import annotations

import torch

from node_fdm.layers.blocks import Backbone, Head, MLPBlock


class TestMLPBlock:
    """MLPBlock forward pass tests."""

    def test_forward_shape(self) -> None:
        """Output shape matches (batch, output_dim)."""
        block = MLPBlock(input_dim=10, hidden_dim=32, output_dim=5, num_layers=2)
        x = torch.randn(8, 10)
        out = block(x)
        assert out.shape == (8, 5)


class TestBackbone:
    """Backbone forward pass tests."""

    def test_forward_shape(self) -> None:
        """Backbone output dim equals hidden_dim."""
        bb = Backbone(input_dim=10, hidden_dim=48, num_layers=2)
        x = torch.randn(4, 10)
        out = bb(x)
        assert out.shape == (4, 48)


class TestHead:
    """Head forward pass tests."""

    def test_forward_shape(self) -> None:
        """Head output is scalar per sample."""
        head = Head(input_dim=48, hidden_dim=24, output_dim=1, num_layers=1)
        x = torch.randn(4, 48)
        out = head(x)
        assert out.shape == (4, 1)

    def test_with_activation(self) -> None:
        """Head with sigmoid activation clamps to [0, 1]."""
        head = Head(
            input_dim=10,
            hidden_dim=8,
            output_dim=1,
            num_layers=1,
            last_activation=torch.nn.Sigmoid,
        )
        x = torch.randn(4, 10)
        out = head(x)
        assert (out >= 0).all()
        assert (out <= 1).all()
