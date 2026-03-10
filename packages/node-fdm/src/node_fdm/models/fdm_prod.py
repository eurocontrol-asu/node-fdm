"""Production-ready flight dynamics model loader.

Replaces legacy ``FlightDynamicsModelProd`` that used
``get_architecture_params_from_meta`` with the typed
``ArchitectureSpec`` / ``resolve_layer_class`` from the registry.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
import torch.nn as nn

from node_fdm.architectures.registry import ArchitectureSpec, resolve_layer_class

__all__ = [
    "FlightDynamicsModelProd",
]

log = structlog.get_logger("node_fdm.models.fdm_prod")


class FlightDynamicsModelProd(nn.Module):  # type: ignore[misc]
    """Load pretrained flight dynamics layers and run inference.

    Unlike the training :class:`FlightDynamicsModel`, this class loads
    layer weights from saved checkpoints at construction time.

    Args:
        spec: Architecture specification.
        stats_dict: Per-column normalization statistics.
        model_params: Tuple of ``(backbone_depth, head_depth, hidden_width)``.
        model_path: Directory containing ``.pt`` checkpoint files.
    """

    def __init__(
        self,
        spec: ArchitectureSpec,
        stats_dict: dict[str, dict[str, float]],
        model_params: tuple[int, int, int],
        model_path: Path,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.stats_dict = stats_dict
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_depth, head_depth, neurons_num = model_params

        self.layers_dict = nn.ModuleDict()
        self.layers_name: list[str] = []

        for layer_spec in spec.layers:
            self.layers_name.append(layer_spec.name)
            layer_cls = resolve_layer_class(layer_spec.layer_class)

            if layer_spec.trainable:
                layer = self._create_structured_layer(
                    layer_spec.input_cols,
                    layer_spec.output_cols,
                    layer_cls,
                    backbone_depth=backbone_depth,
                    head_depth=head_depth,
                    neurons_num=neurons_num,
                )
            else:
                layer = layer_cls(**layer_spec.config)

            self.layers_dict[layer_spec.name] = layer

            # Load checkpoint for trainable layers
            if layer_spec.trainable:
                checkpoint_path = self.model_path / f"{layer_spec.name}.pt"
                if checkpoint_path.exists():
                    checkpoint = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                    layer.load_state_dict(checkpoint["layer_state"], strict=False)
                    log.debug("checkpoint_loaded", layer=layer_spec.name)
                else:
                    log.warning(
                        "no_checkpoint",
                        layer=layer_spec.name,
                        path=str(checkpoint_path),
                    )
            layer.eval()

    def _create_structured_layer(
        self,
        input_cols: list[str],
        output_cols: list[str],
        layer_cls: type[nn.Module],
        *,
        backbone_depth: int,
        head_depth: int,
        neurons_num: int,
    ) -> nn.Module:
        """Build a structured layer with normalization stats.

        Args:
            input_cols: Columns consumed by the layer.
            output_cols: Columns produced by the layer.
            layer_cls: Layer class to instantiate.
            backbone_depth: Backbone network depth.
            head_depth: Head network depth.
            neurons_num: Hidden dimension width.

        Returns:
            Configured layer instance.
        """
        input_mean = {
            col: self.stats_dict[col]["mean"] for col in input_cols if col in self.stats_dict
        }
        input_std = {
            col: self.stats_dict[col]["std"] for col in input_cols if col in self.stats_dict
        }
        output_mean = {
            col: self.stats_dict[col]["mean"] for col in output_cols if col in self.stats_dict
        }
        output_std = {
            col: self.stats_dict[col]["std"] for col in output_cols if col in self.stats_dict
        }
        output_max = {
            col: self.stats_dict[col]["max"] for col in output_cols if col in self.stats_dict
        }

        return layer_cls(
            input_cols=input_cols,
            input_stats=(input_mean, input_std),
            output_cols=output_cols,
            output_stats=(output_mean, output_std, output_max),
            backbone_dim=neurons_num,
            backbone_depth=backbone_depth,
            head_dim=neurons_num // 2,
            head_depth=head_depth,
        )

    def reset_history(self) -> None:
        """Clear stored layer outputs between runs (delegates to inner model)."""
        # FlightDynamicsModelProd has no history — noop for interface compat
        pass

    def forward(
        self,
        x: torch.Tensor,
        u_t: torch.Tensor,
        e_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state derivatives for the current batch.

        Args:
            x: State tensor ``(batch, n_x)``.
            u_t: Control tensor ``(batch, n_u)``.
            e_t: Environment tensor ``(batch, n_e0)``.

        Returns:
            Tensor of state derivatives ``(batch, n_dx)``.
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
        return ode_output
