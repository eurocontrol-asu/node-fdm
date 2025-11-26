#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural flight dynamics model assembled from architecture layers."""

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from node_fdm.utils.learning.base.structured_layer import StructuredLayer


class FlightDynamicsModel(nn.Module):
    """Compute state derivatives using a layered flight dynamics architecture."""

    def __init__(
        self,
        architecture: Sequence[Any],
        stats_dict: Dict[Any, Dict[str, float]],
        model_cols: Tuple[Any, Any, Any, Any, Any],
        model_params: Sequence[int] = (2, 1, 48),
    ) -> None:
        """Initialize the model with architecture definition and statistics.

        Args:
            architecture: Iterable of layer definitions `(name, class, inputs, outputs, structured_flag)`.
            stats_dict: Mapping from column to normalization/denormalization statistics.
            model_cols: Tuple of model column groups (state, control, env, env_extra, derivatives).
            model_params: Sequence defining backbone depth, head depth, and hidden width.
        """
        super().__init__()
        self.architecture = architecture
        self.stats_dict = stats_dict
        self.x_cols, self.u_cols, self.e0_cols, self.e_cols, self.dx_cols = model_cols
        self.backbone_depth, self.head_depth, self.neurons_num = model_params
        self.layers_dict = nn.ModuleDict({})
        self.layers_name = []

        for name, layer_class, input_cols, ouput_cols, structured in self.architecture:
            self.layers_name.append(name)
            if structured:
                self.layers_dict[name] = self.create_structured_layer(
                    input_cols,
                    ouput_cols,
                    layer_class=layer_class,
                )
            else:
                self.layers_dict[name] = layer_class()

    def reset_history(self):
        """Reset internal history buffers.

        Clears stored layer outputs used for debugging or analysis between runs.
        """
        self.history = {}

    def create_structured_layer(
        self,
        input_cols: Sequence[Any],
        output_cols: Sequence[Any],
        layer_class: Any = StructuredLayer,
    ) -> nn.Module:
        """Build a structured layer with normalization and denormalization stats.

        Args:
            input_cols: Columns consumed by the layer.
            output_cols: Columns produced by the layer.
            layer_class: Layer implementation to instantiate.

        Returns:
            Configured structured layer instance.
        """
        input_stats = [
            {
                col.col_name: self.stats_dict[col][metric]
                for col in input_cols
                if col.normalize_mode is not None
            }
            for metric in ["mean", "std"]
        ]
        output_stats = [
            {
                col.col_name: self.stats_dict[col][metric]
                for col in output_cols
                if col.denormalize_mode is not None
            }
            for metric in ["mean", "std", "max"]
        ]

        layer = layer_class(
            input_cols,
            input_stats,
            output_cols,
            output_stats,
            backbone_dim=self.neurons_num,
            backbone_depth=self.backbone_depth,
            head_dim=self.neurons_num // 2,
            head_depth=self.head_depth,
        )

        return layer

    def forward(
        self, x: torch.Tensor, u_t: torch.Tensor, e_t: torch.Tensor
    ) -> torch.Tensor:
        """Compute state derivatives for the current batch.

        Args:
            x: State tensor.
            u_t: Control tensor interpolated at current time.
            e_t: Environment tensor interpolated at current time.

        Returns:
            Tensor of state derivatives assembled from architecture outputs.
        """

        vects = torch.cat([x, u_t, e_t], dim=1)
        vect_dict = dict()
        for i, col in enumerate(self.x_cols + self.u_cols + self.e0_cols):
            vect_dict[col] = vects[..., i]

        for name in self.layers_name:
            vect_dict = vect_dict | self.layers_dict[name](vect_dict)

        ode_output = torch.stack(
            [coeff * vect_dict[col] for coeff, col in self.dx_cols],
            dim=1,
        )

        for col, vect in vect_dict.items():
            if torch.isnan(vect).any():
                pass
            if col in self.history.keys():
                self.history[col] = torch.cat(
                    [self.history[col], vect.unsqueeze(1)], dim=1
                )
            else:
                self.history[col] = vect.unsqueeze(1)

        return ode_output
