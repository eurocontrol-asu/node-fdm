#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured layer combining normalization, shared trunk, and per-column heads."""

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from utils.learning.base.blocks import Backbone, Head
from utils.learning.base.normalizers import InputNormalizer, OutputDenormalizer
from utils.learning.base.multi_layer_dict import MultiLayerDict


class StructuredLayer(nn.Module):
    """Structured layer handling normalization, trunk, heads, and denormalization."""

    def __init__(
        self,
        input_cols: Sequence[Any],
        input_stats: Sequence[Any],
        output_cols: Sequence[Any],
        output_stats: Sequence[Any],
        backbone_dim: int = 48,
        backbone_depth: int = 2,
        head_dim: int = 24,
        head_depth: int = 1,
    ):
        """Initialize structured layer with normalization, trunk, and heads.

        Args:
            input_cols: Columns used as inputs.
            input_stats: Tuple of (mean dict, std dict) for inputs.
            output_cols: Columns produced by the layer.
            output_stats: Tuple of (mean dict, std dict, max dict) for outputs.
            backbone_dim: Hidden dimension for backbone MLP.
            backbone_depth: Number of layers in backbone.
            head_dim: Hidden dimension for head MLPs.
            head_depth: Number of layers in each head.
        """
        super().__init__()
        input_dim = len(input_cols)
        input_mean_dict, input_std_dict = input_stats
        output_mean_dict, output_std_dict, output_max_dict = output_stats

        self.input_cols = input_cols
        self.output_cols = output_cols

        self.normalizer = InputNormalizer(input_mean_dict, input_std_dict)
        self.backbone = Backbone(
            input_dim, hidden_dim=backbone_dim, num_layers=backbone_depth
        )

        def head_factory(col):
            return Head(
                backbone_dim,
                hidden_dim=head_dim,
                output_dim=1,
                num_layers=head_depth,
                last_activation=col.last_activation_fn,
            )

        self.heads = MultiLayerDict(self.output_cols, head_factory)

        self.denormalizer = OutputDenormalizer(
            output_mean_dict, output_std_dict, output_max_dict
        )

    def normalize_input(
        self, x_dict: Dict[Any, torch.Tensor]
    ) -> Sequence[torch.Tensor]:
        """Normalize input tensors according to column stats.

        Args:
            x_dict: Mapping from column identifiers to input tensors.

        Returns:
            List of normalized tensors ready for concatenation.
        """
        out_list = []
        for col in self.input_cols:
            norm_vect = self.normalizer(x_dict[col], col)
            if len(norm_vect.shape) == 1:
                norm_vect = norm_vect.unsqueeze(1)
            out_list.append(norm_vect)
        return out_list

    def denormalize_output(
        self, out_norm_dict: Dict[Any, torch.Tensor]
    ) -> Dict[Any, torch.Tensor]:
        """Denormalize outputs from heads back to physical scale.

        Args:
            out_norm_dict: Mapping from column identifiers to normalized outputs.

        Returns:
            Dictionary of denormalized predictions keyed by column identifiers.
        """
        out_pred_dict = dict()
        for col in self.output_cols:
            out_norm = out_norm_dict[col]
            out_pred_dict[col] = self.denormalizer(out_norm.squeeze(-1), col)
        return out_pred_dict

    def forward_trunk_head(
        self, x_dict: Dict[Any, torch.Tensor]
    ) -> Dict[Any, torch.Tensor]:
        """Run normalization, trunk, and heads to produce normalized outputs.

        Args:
            x_dict: Mapping of input tensors keyed by column identifiers.

        Returns:
            Dictionary of normalized outputs keyed by column identifiers.
        """
        out_list = self.normalize_input(x_dict)
        x_norm = torch.cat(out_list, dim=1)
        features = self.backbone(x_norm)
        out_norm_dict = self.heads(features)
        return out_norm_dict

    def forward(self, x_dict: Dict[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
        """Compute denormalized predictions from input mapping.

        Args:
            x_dict: Mapping from column identifiers to input tensors.

        Returns:
            Dictionary of denormalized predictions keyed by column identifiers.
        """
        out_norm_dict = self.forward_trunk_head(x_dict)
        out_pred_dict = self.denormalize_output(out_norm_dict)
        return out_pred_dict
