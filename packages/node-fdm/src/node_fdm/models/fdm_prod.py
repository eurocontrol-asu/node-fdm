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

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, resolve_layer_class

__all__ = [
    "FlightDynamicsModelProd",
]

log = structlog.get_logger("node_fdm.models.fdm_prod")


class FlightDynamicsModelProd(nn.Module):
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
                    layer_spec,
                    layer_cls,
                    backbone_depth=backbone_depth,
                    head_depth=head_depth,
                    neurons_num=neurons_num,
                )
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

            # Load checkpoint for all layers that have one
            if True:
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
        layer_spec: LayerSpec,
        layer_cls: type[nn.Module],
        *,
        backbone_depth: int,
        head_depth: int,
        neurons_num: int,
    ) -> nn.Module:
        """Build a structured layer with normalization stats.

        Mirrors :meth:`FlightDynamicsModel._create_structured_layer` to
        ensure the production model has identical denormalization config.

        Args:
            layer_spec: Layer specification with input/output columns and config.
            layer_cls: Layer class to instantiate.
            backbone_depth: Backbone network depth.
            head_depth: Head network depth.
            neurons_num: Hidden dimension width.

        Returns:
            Configured layer instance.
        """
        input_cols = layer_spec.input_cols
        output_cols = layer_spec.output_cols

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

        # Scaled denormalization precedence (mirrors training-side fdm.py):
        #   scale: layer scale_overrides > spec.nn_output_caps > stats p999
        #   cap:   layer cap_overrides   > spec.nn_output_caps > |dx_bounds|
        # ``nn_output_caps`` carries hard physics caps (e.g. phi_bank=1.0 rad)
        # that override the data-driven p999 fallback; ``scale_overrides`` /
        # ``cap_overrides`` remain as a per-layer escape hatch for legacy
        # checkpoints.
        raw_modes = layer_spec.config.get("denormalize_modes", {})
        denormalize_modes: dict[str, str] = dict(raw_modes) if isinstance(raw_modes, dict) else {}
        raw_scale_overrides = layer_spec.config.get("scale_overrides", {})
        scale_overrides: dict[str, float] = (
            dict(raw_scale_overrides) if isinstance(raw_scale_overrides, dict) else {}
        )
        raw_cap_overrides = layer_spec.config.get("cap_overrides", {})
        cap_overrides: dict[str, float] = (
            dict(raw_cap_overrides) if isinstance(raw_cap_overrides, dict) else {}
        )
        nn_output_caps: dict[str, float] = self.spec.nn_output_caps
        scale_dict: dict[str, float] = {}
        cap_dict: dict[str, float] = {}
        for col, mode in denormalize_modes.items():
            if mode != "scaled":
                continue
            # Scale = data-driven natural unit (mirrors fdm.py — see notes
            # there). nn_output_caps does NOT feed scale; it only caps.
            if col in scale_overrides:
                scale_dict[col] = scale_overrides[col]
            elif col in self.stats_dict:
                scale_dict[col] = self.stats_dict[col].get("p999", self.stats_dict[col]["std"])
            # Cap = hard regulatory/physical bound (or fallback to scale).
            if col in cap_overrides:
                cap_dict[col] = cap_overrides[col]
            elif col in nn_output_caps:
                cap_dict[col] = nn_output_caps[col]
            elif col in self.spec.dx_bounds:
                cap_dict[col] = max(abs(v) for v in self.spec.dx_bounds[col])
            elif col in scale_dict:
                cap_dict[col] = scale_dict[col]

        return layer_cls(
            input_cols=input_cols,
            input_stats=(input_mean, input_std),
            output_cols=output_cols,
            output_stats=(output_mean, output_std, output_max),
            backbone_dim=neurons_num,
            backbone_depth=backbone_depth,
            head_dim=neurons_num // 2,
            head_depth=head_depth,
            denormalize_modes=denormalize_modes or None,
            scale_dict=scale_dict or None,
            cap_dict=cap_dict or None,
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
