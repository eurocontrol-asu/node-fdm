#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Engine layer for QAR-based flight dynamics models.

This module defines the EngineLayer responsible for handling engine-related
signals, including N1 and fuel flow predictions within the structured model
architecture.
"""

from typing import Any, Dict, Mapping

import torch
from node_fdm.architectures.qar.columns import (
    col_n1,
    col_ff,
)
from node_fdm.utils.learning.base.structured_layer import StructuredLayer


class EngineLayer(StructuredLayer):
    """Engine-specific structured layer for predicting normalized outputs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the engine layer with inherited structured components.

        Args:
            *args: Positional arguments forwarded to ``StructuredLayer``; the
                exact structure depends on upstream configuration.
            **kwargs: Keyword arguments forwarded to ``StructuredLayer``; the
                exact structure depends on upstream configuration.
        """
        super().__init__(*args, **kwargs)
        self.n1_max: int = 100

    def forward(self, x_dict: Mapping[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
        """Run a forward pass to produce denormalized engine predictions.

        Args:
            x_dict: Mapping of column identifiers to input tensors. Column
                identifiers are dynamically defined, so ``Any`` is used here
                to reflect the flexible key types.

        Returns:
            Dictionary mapping column identifiers to denormalized prediction
            tensors for N1 and fuel flow.
        """
        out_norm_dict = self.forward_trunk_head(x_dict)

        fuel_flow_norm = out_norm_dict[col_ff].squeeze(-1)
        fuel_flow_pred = self.denormalizer(fuel_flow_norm, col_ff)

        out_pred_dict = dict()
        out_pred_dict[col_n1] = self.n1_max * out_norm_dict[col_n1].squeeze(-1)
        out_pred_dict[col_ff] = fuel_flow_pred

        return out_pred_dict
