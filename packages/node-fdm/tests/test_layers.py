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


class TestInputNormalizer:
    """Tests for InputNormalizer."""

    def test_passthrough_mode(self) -> None:
        """Column with None mode returns input unchanged."""
        from node_fdm.layers.normalizers import InputNormalizer

        norm = InputNormalizer(
            mean_dict={"alt": 5000.0},
            std_dict={"alt": 2000.0},
            modes={"alt": None},
        )
        x = torch.tensor([1.0, 2.0, 3.0])
        out = norm(x, "alt")
        assert torch.allclose(out, x)


class TestOutputDenormalizer:
    """Tests for OutputDenormalizer."""

    def test_max_mode(self) -> None:
        """'max' mode scales by max value."""
        from node_fdm.layers.normalizers import OutputDenormalizer

        denorm = OutputDenormalizer(
            mean_dict={"speed": 200.0},
            std_dict={"speed": 50.0},
            max_dict={"speed": 300.0},
            modes={"speed": "max"},
        )
        x = torch.tensor([0.5])
        out = denorm(x, "speed")
        assert torch.isclose(out, torch.tensor([150.0])).all()

    def test_passthrough_mode(self) -> None:
        """None mode returns input unchanged."""
        from node_fdm.layers.normalizers import OutputDenormalizer

        denorm = OutputDenormalizer(
            mean_dict={"alt": 5000.0},
            std_dict={"alt": 2000.0},
            max_dict={"alt": 10000.0},
            modes={"alt": None},
        )
        x = torch.tensor([42.0])
        out = denorm(x, "alt")
        assert torch.isclose(out, torch.tensor([42.0])).all()


class TestEngineLayer:
    """Tests for EngineLayer — N1 and non-N1 branches."""

    def test_fuel_flow_uses_denormalizer(self) -> None:
        """Non-N1 column is denormalized via standard OutputDenormalizer."""
        from node_fdm.layers.engine import EngineLayer

        layer = EngineLayer(
            input_cols=["alt"],
            input_stats=({"alt": 5000.0}, {"alt": 2000.0}),
            output_cols=["fuel_flow"],
            output_stats=({"fuel_flow": 1.0}, {"fuel_flow": 0.5}, {"fuel_flow": 3.0}),
            backbone_dim=16,
            backbone_depth=1,
            head_dim=16,
            head_depth=1,
        )
        x_dict = {"alt": torch.randn(4)}
        out = layer(x_dict)
        assert "fuel_flow" in out
        assert out["fuel_flow"].shape == (4,)

    def test_n1_column_scaled(self) -> None:
        """N1 column is multiplied by _N1_MAX (100)."""
        from node_fdm.layers.engine import EngineLayer

        layer = EngineLayer(
            input_cols=["alt"],
            input_stats=({"alt": 5000.0}, {"alt": 2000.0}),
            output_cols=["N1"],
            output_stats=({"N1": 80.0}, {"N1": 10.0}, {"N1": 100.0}),
            backbone_dim=16,
            backbone_depth=1,
            head_dim=16,
            head_depth=1,
        )
        x_dict = {"alt": torch.randn(4)}
        out = layer(x_dict)
        assert "N1" in out
        assert out["N1"].shape == (4,)
