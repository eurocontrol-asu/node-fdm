"""MLP building blocks for structured flight dynamics models.

Provides ``MLPBlock``, ``Backbone`` (shared trunk), ``Head`` (per-output),
and ``MultiLayerDict`` (module-dict keyed by column name).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

__all__ = [
    "Backbone",
    "Head",
    "MLPBlock",
    "MultiLayerDict",
]


class MLPBlock(nn.Module):
    """Simple configurable multi-layer perceptron block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        last_activation: type[nn.Module] | None = None,
    ) -> None:
        """Initialize the MLP block.

        Args:
            input_dim: Size of input feature dimension.
            hidden_dim: Hidden layer width.
            output_dim: Output feature dimension.
            num_layers: Number of hidden layers.
            last_activation: Optional module class applied after the final linear.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        if last_activation is not None:
            layers.append(last_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP on the input tensor.

        Args:
            x: Input tensor of shape ``(batch, features)``.

        Returns:
            Output tensor.
        """
        result: torch.Tensor = self.net(x)
        return result


class Backbone(MLPBlock):
    """Backbone MLP used as shared trunk before output heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 48,
        num_layers: int = 2,
        last_activation: type[nn.Module] | None = None,
    ) -> None:
        """Initialize backbone with symmetric hidden dimensions."""
        super().__init__(
            input_dim,
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            last_activation=last_activation,
        )


class Head(MLPBlock):
    """Head MLP producing final scalar outputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 24,
        output_dim: int = 1,
        num_layers: int = 1,
        last_activation: type[nn.Module] | None = None,
    ) -> None:
        """Initialize head with optional activation."""
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers=num_layers,
            last_activation=last_activation,
        )


class MultiLayerDict(nn.Module):
    """Module dictionary that builds one sub-module per output column.

    Keys are plain column name strings (not ``Column`` objects).
    """

    def __init__(
        self,
        output_cols: list[str],
        layer_factory: Callable[[str], nn.Module],
    ) -> None:
        """Initialize with one layer per output column.

        Args:
            output_cols: Column names to produce.
            layer_factory: Factory receiving a column name and returning a module.
        """
        super().__init__()
        self.output_cols = output_cols
        col_dict = {col: layer_factory(col) for col in self.output_cols}
        self.layer_dict = nn.ModuleDict(col_dict)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply each sub-layer and return outputs keyed by column name.

        Args:
            x: Input tensor forwarded to each sub-layer.

        Returns:
            Dictionary mapping column names to layer outputs.
        """
        return {col: self.layer_dict[col](x) for col in self.output_cols}
