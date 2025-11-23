#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MLP building blocks for structured models."""

from typing import Any, Callable, Optional

import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    """Simple configurable multi-layer perceptron block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        last_activation: Optional[Callable[[], Any]] = None,
    ):
        """Initialize the MLP block.

        Args:
            input_dim: Size of input feature dimension.
            hidden_dim: Hidden layer width.
            output_dim: Output feature dimension.
            num_layers: Number of hidden layers.
            last_activation: Optional callable producing a final activation module.
        """
        
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        if last_activation is not None:
            layers.append(last_activation())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP block on the input tensor.

        Args:
            x: Input tensor of shape (batch, features).

        Returns:
            Tensor output of the MLP.
        """
        return self.net(x)

class Backbone(MLPBlock):
    """Backbone MLP used before output heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 2, last_activation: Optional[Callable[[], Any]] = None):
        """Initialize backbone with symmetric hidden dimensions."""
        super().__init__(
            input_dim, 
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            last_activation=last_activation
        )

class Head(MLPBlock):
    """Head MLP producing final outputs."""

    def __init__(self, input_dim: int, hidden_dim: int = 24, output_dim: int = 1, num_layers: int = 1, last_activation: Optional[Callable[[], Any]] = None):
        """Initialize head with optional activation and custom sizes."""
        super().__init__(
            input_dim, 
            hidden_dim, 
            output_dim, 
            num_layers=num_layers,
            last_activation=last_activation
        )
