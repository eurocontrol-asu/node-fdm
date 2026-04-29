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
    "GammaDefaultNet",
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
        """Initialize head with optional activation.

        The last linear layer is zero-initialized so that a fresh
        Neural ODE predicts dx ≈ 0 (identity dynamics).  This is
        critical for stable training: large initial derivatives cause
        the ODE to diverge within the first integration window.
        """
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            num_layers=num_layers,
            last_activation=last_activation,
        )
        # Zero-init last linear so fresh model predicts dx ≈ 0
        last_linear = self.net[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)


class GammaDefaultNet(nn.Module):
    """Context-aware default gamma predictor.

    Takes altitude, TAS, vertical speed, and altitude difference
    (alt_target - alt) as inputs and produces a bounded default gamma
    per batch element.  Output is clamped to ±``max_gamma_rad``
    (default 0.18 rad ≈ 10°) via tanh.

    ``alt_diff`` carries the climb/descend intent: negative means
    "above target → descend", positive means "below target → climb".
    Without it the net cannot determine flight direction from state
    alone (an aircraft at 5 000 m could be climbing or descending).

    Gamma is intentionally excluded from inputs to avoid a feedback
    loop (the net's output influences gamma via gamma_diff).  Vertical
    speed (``vz = tas * sin(gamma)``) is used instead to convey
    current climb/descent rate without creating a direct feedback path.

    When ``input_stats`` is provided, inputs are z-score normalized
    before the MLP to prevent tanh saturation on raw physical values
    (alt ~10 000, TAS ~230).  Without normalization the MLP output
    is O(10³), tanh saturates to ±1, and gradients vanish.
    """

    _INPUT_DIM: int = 4  # alt, tas, vz, alt_diff

    _MAX_GAMMA_RAD: float = 0.18  # ≈ 10°, physical upper bound

    _mean_alt: torch.Tensor
    _std_alt: torch.Tensor
    _mean_tas: torch.Tensor
    _std_tas: torch.Tensor
    _mean_vz: torch.Tensor
    _std_vz: torch.Tensor
    _mean_alt_diff: torch.Tensor
    _std_alt_diff: torch.Tensor
    _scale: torch.Tensor

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 1,
        input_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize the gamma default network.

        Args:
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
            input_stats: Optional mapping ``{"alt": {"mean": ..., "std": ...},
                "tas": ..., "vz": ..., "alt_diff": ...}`` for input
                z-score normalization.  Keys are canonical names.
        """
        super().__init__()
        self.register_buffer("_scale", torch.tensor(self._MAX_GAMMA_RAD))

        # Register normalization buffers (default: no-op identity)
        _keys = ("alt", "tas", "vz", "alt_diff")
        for key in _keys:
            mean = 0.0
            std = 1.0
            if input_stats and key in input_stats:
                mean = input_stats[key].get("mean", 0.0)
                std = input_stats[key].get("std", 1.0)
            self.register_buffer(f"_mean_{key}", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer(f"_std_{key}", torch.tensor(std, dtype=torch.float32))

        # Custom MLP with SiLU activation (smooth, no dead neurons unlike ReLU)
        # and Xavier init for balanced gradients.  No small-init needed since
        # inputs are z-score normalized and tanh bounds the output.
        layers: list[nn.Module] = []
        prev_dim = self._INPUT_DIM
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        alt: torch.Tensor,
        tas: torch.Tensor,
        vz: torch.Tensor,
        alt_diff: torch.Tensor,
    ) -> torch.Tensor:
        """Predict default gamma from flight context.

        Args:
            alt: Altitude tensor of shape ``(batch,)``.
            tas: True airspeed tensor of shape ``(batch,)``.
            vz: Vertical speed tensor of shape ``(batch,)``.
            alt_diff: Altitude difference (target - current) ``(batch,)``.

        Returns:
            Scalar gamma per sample, shape ``(batch,)``.
        """
        alt_n = (alt - self._mean_alt) / self._std_alt
        tas_n = (tas - self._mean_tas) / self._std_tas
        vz_n = (vz - self._mean_vz) / self._std_vz
        alt_diff_n = (alt_diff - self._mean_alt_diff) / self._std_alt_diff
        x = torch.stack([alt_n, tas_n, vz_n, alt_diff_n], dim=-1)  # (batch, 4)
        out: torch.Tensor = torch.tanh(self.mlp(x).squeeze(-1)) * self._scale
        return out


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
