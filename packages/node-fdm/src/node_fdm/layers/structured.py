"""Structured layer combining normalization, shared backbone, and per-column heads.

Replaces legacy ``StructuredLayer`` that used ``Column`` objects with a version
using plain string column names and explicit config dictionaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from node_fdm.layers.blocks import Backbone, Head, MultiLayerDict
from node_fdm.layers.normalizers import InputNormalizer, OutputDenormalizer

__all__ = [
    "StructuredLayer",
]


class StructuredLayer(nn.Module):
    """Structured layer: normalize → backbone → per-column heads → denormalize."""

    def __init__(
        self,
        input_cols: list[str],
        input_stats: tuple[dict[str, float], dict[str, float]],
        output_cols: list[str],
        output_stats: tuple[dict[str, float], dict[str, float], dict[str, float]],
        backbone_dim: int = 48,
        backbone_depth: int = 2,
        head_dim: int = 24,
        head_depth: int = 1,
        activations: dict[str, type[nn.Module] | None] | None = None,
        normalize_modes: dict[str, str | None] | None = None,
        denormalize_modes: dict[str, str | None] | None = None,
        scale_dict: dict[str, float] | None = None,
        cap_dict: dict[str, float] | None = None,
        output_init_biases: dict[str, float] | None = None,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        """Initialize structured layer.

        Args:
            input_cols: Column names consumed as inputs.
            input_stats: Tuple of ``(mean_dict, std_dict)`` for inputs.
            output_cols: Column names produced by the layer.
            output_stats: Tuple of ``(mean_dict, std_dict, max_dict)`` for outputs.
            backbone_dim: Hidden dimension for backbone MLP.
            backbone_depth: Number of layers in backbone.
            head_dim: Hidden dimension for head MLPs.
            head_depth: Number of layers in each head.
            activations: Optional mapping of output column → activation class.
            normalize_modes: Optional mapping of input column → normalize mode.
            denormalize_modes: Optional mapping of output column → denormalize mode.
        """
        super().__init__()
        input_dim = len(input_cols)
        input_mean_dict, input_std_dict = input_stats
        output_mean_dict, output_std_dict, output_max_dict = output_stats

        self.input_cols = input_cols
        self.output_cols = output_cols

        self.normalizer = InputNormalizer(input_mean_dict, input_std_dict, modes=normalize_modes)
        self.backbone = Backbone(
            input_dim,
            hidden_dim=backbone_dim,
            num_layers=backbone_depth,
            activation=activation,
        )

        _activations = activations or {}
        _init_biases = output_init_biases or {}

        def head_factory(col: str) -> Head:
            """Build a Head module for the given output column."""
            return Head(
                backbone_dim,
                hidden_dim=head_dim,
                output_dim=1,
                num_layers=head_depth,
                last_activation=_activations.get(col),
                output_init_bias=_init_biases.get(col, 0.0),
                activation=activation,
            )

        self.heads = MultiLayerDict(self.output_cols, head_factory)

        self.denormalizer = OutputDenormalizer(
            output_mean_dict,
            output_std_dict,
            output_max_dict,
            modes=denormalize_modes,
            scale_dict=scale_dict,
            cap_dict=cap_dict,
        )

    def normalize_input(self, x_dict: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Normalize input tensors by column.

        Args:
            x_dict: Mapping from column names to input tensors.

        Returns:
            List of normalized tensors ready for concatenation.
        """
        out_list: list[torch.Tensor] = []
        for col in self.input_cols:
            norm_vect = self.normalizer(x_dict[col], col)
            if len(norm_vect.shape) == 1:
                norm_vect = norm_vect.unsqueeze(1)
            out_list.append(norm_vect)
        return out_list

    def denormalize_output(
        self, out_norm_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Denormalize outputs from heads back to physical scale.

        Args:
            out_norm_dict: Mapping from column names to normalized outputs.

        Returns:
            Dictionary of denormalized predictions keyed by column name.
        """
        out_pred_dict: dict[str, torch.Tensor] = {}
        for col in self.output_cols:
            out_norm = out_norm_dict[col]
            out_pred_dict[col] = self.denormalizer(out_norm.squeeze(-1), col)
        return out_pred_dict

    def forward_trunk_head(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run normalization, trunk, and heads to produce normalized outputs.

        Args:
            x_dict: Mapping of input tensors keyed by column name.

        Returns:
            Dictionary of normalized outputs keyed by column name.
        """
        out_list = self.normalize_input(x_dict)
        x_norm = torch.cat(out_list, dim=1)
        features = self.backbone(x_norm)
        out_norm_dict: dict[str, torch.Tensor] = self.heads(features)
        return out_norm_dict

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute denormalized predictions from input mapping.

        Args:
            x_dict: Mapping from column names to input tensors.

        Returns:
            Dictionary of denormalized predictions keyed by column name.
        """
        out_norm_dict = self.forward_trunk_head(x_dict)
        return self.denormalize_output(out_norm_dict)
