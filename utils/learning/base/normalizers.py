#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalization and denormalization helpers for model inputs/outputs."""

from typing import Any, Dict

import torch 
import torch.nn as nn

class InputNormalizer(nn.Module):
    """Normalize inputs based on provided mean/std statistics."""

    def __init__(self, mean_dict: Dict[Any, Any], std_dict: Dict[Any, Any]):
        """Register mean and std buffers for each column key."""
        super().__init__()
        for k in mean_dict:
            k = str(k)
            self.register_buffer(f'mean_{k}', torch.tensor(mean_dict[k], dtype=torch.float32))
            self.register_buffer(f'std_{k}', torch.tensor(std_dict[k], dtype=torch.float32))
    
    def forward(self, x: torch.Tensor, col: Any) -> torch.Tensor:
        """Apply normalization if requested by the column.

        Args:
            x: Input tensor to normalize.
            col: Column metadata indicating normalization mode.

        Returns:
            Normalized tensor or original tensor if no normalization is applied.
        """
        if col.normalize_mode == "normal":
            mean = getattr(self, f'mean_{col}')
            std = getattr(self, f'std_{col}')
            return (x - mean) / std
        return x


class OutputDenormalizer(nn.Module):
    """Denormalize network outputs based on provided statistics."""

    def __init__(self, mean_dict: Dict[Any, Any], std_dict: Dict[Any, Any], max_dict: Dict[Any, Any], max_ratio: float = 1.2):
        """Register normalization statistics and max scaling per column."""
        super().__init__()
        self.max_ratio = max_ratio
        for k in mean_dict:
            k = str(k)
            self.register_buffer(f'mean_{k}', torch.tensor(mean_dict[k], dtype=torch.float32))
            self.register_buffer(f'std_{k}', torch.tensor(std_dict[k], dtype=torch.float32))
            self.register_buffer(f'max_{k}', torch.tensor(max_dict[k], dtype=torch.float32))

    def forward(self, x: torch.Tensor, col: Any) -> torch.Tensor:
        """Denormalize outputs according to column configuration.

        Args:
            x: Normalized tensor to denormalize.
            col: Column metadata indicating denormalization mode.

        Returns:
            Tensor in physical scale.
        """
        if col.denormalize_mode == "normal_clamp":
            mean = getattr(self, f'mean_{col}')
            std = getattr(self, f'std_{col}')
            maxv = getattr(self, f'max_{col}')
            value = mean + x * std
            value_clamped = torch.clamp(value, min=-self.max_ratio * maxv, max=self.max_ratio * maxv)
            return value_clamped
        elif col.denormalize_mode == "max":
            maxv = getattr(self, f'max_{col}')
            value = x * maxv
            return value
        return x
