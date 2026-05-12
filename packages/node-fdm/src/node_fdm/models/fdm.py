"""Neural flight dynamics model assembled from architecture specs.

Replaces the legacy ``FlightDynamicsModel`` that accepted ``Sequence[Any]``
with a typed version driven by ``ArchitectureSpec`` and ``LayerSpec``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from node_fdm.architectures.registry import ArchitectureSpec, resolve_layer_class
from node_fdm.layers.activations import resolve_activation

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
        activation: str = "silu",
    ) -> None:
        """Initialize the model from an architecture specification.

        Args:
            spec: Typed architecture specification.
            stats_dict: Mapping ``column_name → {"mean": ..., "std": ..., "max": ...}``.
            model_params: Tuple of ``(backbone_depth, head_depth, hidden_width)``.
            activation: Hidden-layer activation identifier (``"silu"`` /
                ``"relu"`` / ``"gelu"`` / ``"tanh"``). Persisted in
                ``meta.json`` so inference rebuilds the same graph.
        """
        super().__init__()
        self.spec = spec
        self.stats_dict = stats_dict
        self.backbone_depth, self.head_depth, self.neurons_num = model_params
        self.activation = activation
        self._activation_cls: type[nn.Module] = resolve_activation(activation)
        self.layers_dict = nn.ModuleDict()
        self.layers_name: list[str] = []
        self.history: dict[str, Any] = {}

        for layer_spec in spec.layers:
            self.layers_name.append(layer_spec.name)
            layer_cls = resolve_layer_class(layer_spec.layer_class)

            if layer_spec.trainable:
                layer = self._create_structured_layer(layer_spec, layer_cls)
            else:
                # Pass input_stats for layers that support normalization
                # (e.g. TrajectoryLayer → GammaDefaultNet).
                import inspect

                sig = inspect.signature(layer_cls.__init__)
                if "input_stats" in sig.parameters:
                    layer = layer_cls(**layer_spec.config, input_stats=stats_dict)
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

        denormalize_modes: dict[str, str | None] = layer_spec.config.get("denormalize_modes", {})
        scale_overrides: dict[str, float] = layer_spec.config.get("scale_overrides", {})
        cap_overrides: dict[str, float] = layer_spec.config.get("cap_overrides", {})
        nn_output_caps: dict[str, float] = self.spec.nn_output_caps
        scale_dict: dict[str, float] = {}
        cap_dict: dict[str, float] = {}
        for col, mode in denormalize_modes.items():
            if mode == "scaled":
                # Scale = natural unit of the signal (data-driven).
                # Precedence: layer scale_overrides > stats p999.
                # ``nn_output_caps`` deliberately does NOT feed scale: it
                # represents a hard regulatory/physical bound that should
                # leave room for the data-driven scale to set the gradient
                # sensitivity (cap > scale ⇒ live tanh gradient up to cap).
                if col in scale_overrides:
                    scale_dict[col] = scale_overrides[col]
                else:
                    col_stats = self.stats_dict.get(col, {})
                    if "p999" not in col_stats:
                        msg = f"Scaled mode for '{col}' requires 'p999' in stats_dict"
                        raise ValueError(msg)
                    scale_dict[col] = col_stats["p999"]
                # Cap = hard bound (regulatory / physical / numerical).
                # Precedence: layer cap_overrides > spec.nn_output_caps >
                # dx_bounds > scale (cap≡scale fallback preserves legacy
                # behavior when no explicit physical bound is declared).
                if col in cap_overrides:
                    cap_dict[col] = cap_overrides[col]
                elif col in nn_output_caps:
                    cap_dict[col] = nn_output_caps[col]
                elif col in self.spec.dx_bounds:
                    cap_dict[col] = self.spec.dx_bounds[col][1]
                else:
                    cap_dict[col] = scale_dict[col]

        output_init_biases: dict[str, float] = layer_spec.config.get("output_init_biases", {})

        return layer_cls(
            input_cols=layer_spec.input_cols,
            input_stats=(input_mean, input_std),
            output_cols=layer_spec.output_cols,
            output_stats=(output_mean, output_std, output_max),
            backbone_dim=self.neurons_num,
            backbone_depth=self.backbone_depth,
            head_dim=self.neurons_num // 2,
            head_depth=self.head_depth,
            denormalize_modes=denormalize_modes if denormalize_modes else None,
            scale_dict=scale_dict if scale_dict else None,
            cap_dict=cap_dict if cap_dict else None,
            output_init_biases=output_init_biases if output_init_biases else None,
            activation=self._activation_cls,
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
