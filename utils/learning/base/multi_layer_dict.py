#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ModuleDict wrapper to build per-column heads."""

from typing import Any, Callable, Sequence

import torch.nn as nn


class MultiLayerDict(nn.Module):
    """Create a dictionary of layers keyed by column."""

    def __init__(self, output_cols: Sequence[Any], layer_factory: Callable[[Any], nn.Module]):
        """Initialize layered dictionary.

        Args:
            output_cols: Iterable of column descriptors to key layers.
            layer_factory: Factory callable producing a module per column.
        """
        super().__init__()
        self.output_cols = output_cols
        col_dict = {
            col.col_name : layer_factory(col)
            for col in self.output_cols}
        self.layer_dict = nn.ModuleDict(col_dict)
        
    def forward(self, x: Any) -> dict:
        """Apply each sub-layer and return outputs keyed by column objects.

        Args:
            x: Input tensor forwarded to each sub-layer.

        Returns:
            Dictionary mapping column objects to layer outputs.
        """
        output_dict = {
            col: self.layer_dict[col.col_name](x) 
            for col in self.output_cols
        }
        return output_dict
