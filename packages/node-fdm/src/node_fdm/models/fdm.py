"""Neural flight dynamics model assembled from architecture specs.

Replaces the legacy ``FlightDynamicsModel`` that accepted ``Sequence[Any]``
with a typed version driven by ``ArchitectureSpec`` and ``LayerSpec``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from node_fdm.architectures.registry import ArchitectureSpec, resolve_layer_class

__all__ = [
    "FlightDynamicsModel",
]

_DEFAULT_BACKBONE_DEPTH: int = 2
_DEFAULT_HEAD_DEPTH: int = 1
_DEFAULT_NEURONS: int = 48


class FlightDynamicsModel(nn.Module):
    """Compute state derivatives using a layered flight dynamics architecture.

    The model iterates over ``spec.layers``, resolving each layer class at
    construction time.  Trainable layers get normalization stats; non-trainable
    layers are instantiated directly.
    """

    def __init__(
        self,
        spec: ArchitectureSpec,
        stats_dict: dict[str, dict[str, float]],
        model_params: tuple[int, int, int] = (
            _DEFAULT_BACKBONE_DEPTH,
            _DEFAULT_HEAD_DEPTH,
            _DEFAULT_NEURONS,
        ),
    ) -> None:
        """Initialize the model from an architecture specification.

        Args:
            spec: Typed architecture specification.
            stats_dict: Mapping ``column_name → {"mean": ..., "std": ..., "max": ...}``.
            model_params: Tuple of ``(backbone_depth, head_depth, hidden_width)``.
        """
        super().__init__()
        self.spec = spec
        self.stats_dict = stats_dict
        self.backbone_depth, self.head_depth, self.neurons_num = model_params
        self.layers_dict = nn.ModuleDict()
        self.layers_name: list[str] = []
        self.history: dict[str, Any] = {}

        for layer_spec in spec.layers:
            self.layers_name.append(layer_spec.name)
            layer_cls = resolve_layer_class(layer_spec.layer_class)

            if layer_spec.trainable:
                layer = self._create_structured_layer(layer_spec, layer_cls)
            else:
                layer = layer_cls(**layer_spec.config)

            self.layers_dict[layer_spec.name] = layer

    def _create_structured_layer(
        self,
        layer_spec: Any,
        layer_cls: type[nn.Module],
    ) -> nn.Module:
        """Build a structured (trainable) layer with normalization stats.

        Args:
            layer_spec: Layer specification with input/output columns.
            layer_cls: The ``StructuredLayer`` subclass to instantiate.

        Returns:
            Configured structured layer instance.
        """
        input_mean = {
            col: self.stats_dict[col]["mean"]
            for col in layer_spec.input_cols
            if col in self.stats_dict
        }
        input_std = {
            col: self.stats_dict[col]["std"]
            for col in layer_spec.input_cols
            if col in self.stats_dict
        }
        output_mean = {
            col: self.stats_dict[col]["mean"]
            for col in layer_spec.output_cols
            if col in self.stats_dict
        }
        output_std = {
            col: self.stats_dict[col]["std"]
            for col in layer_spec.output_cols
            if col in self.stats_dict
        }
        output_max = {
            col: self.stats_dict[col]["max"]
            for col in layer_spec.output_cols
            if col in self.stats_dict
        }

        return layer_cls(
            input_cols=layer_spec.input_cols,
            input_stats=(input_mean, input_std),
            output_cols=layer_spec.output_cols,
            output_stats=(output_mean, output_std, output_max),
            backbone_dim=self.neurons_num,
            backbone_depth=self.backbone_depth,
            head_dim=self.neurons_num // 2,
            head_depth=self.head_depth,
        )

    def reset_history(self) -> None:
        """Clear stored layer outputs between runs."""
        self.history = {}

    def forward(self, x: torch.Tensor, u_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """Compute state derivatives for the current batch.

        Args:
            x: State tensor of shape ``(batch, n_x)``.
            u_t: Control tensor interpolated at current time ``(batch, n_u)``.
            e_t: Environment tensor interpolated at current time ``(batch, n_e0)``.

        Returns:
            Tensor of state derivatives ``(batch, n_dx)`` assembled from
            architecture outputs.
        """
        vects = torch.cat([x, u_t, e_t], dim=1)
        vect_dict: dict[str, torch.Tensor] = {}
        all_cols = self.spec.x_cols + self.spec.u_cols + self.spec.e0_cols
        for i, col in enumerate(all_cols):
            vect_dict[col] = vects[..., i]

        for name in self.layers_name:
            vect_dict = vect_dict | self.layers_dict[name](vect_dict)

        ode_output = torch.stack(
            [coeff * vect_dict[col] for coeff, col in self.spec.dx_cols],
            dim=1,
        )

        # Store history for debugging/analysis (detached to avoid breaking
        # autograd when the model is called repeatedly inside odeint).
        for col, vect in vect_dict.items():
            vect_d = vect.detach()
            if col in self.history:
                self.history[col] = torch.cat([self.history[col], vect_d.unsqueeze(1)], dim=1)
            else:
                self.history[col] = vect_d.unsqueeze(1)

        return ode_output
