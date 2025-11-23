#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loss factory helpers for model training."""

from typing import Any

import torch.nn as nn

        
def get_loss(loss_name: str) -> Any:
    """Return a loss function instance by name.

    Args:
        loss_name: Identifier for the desired loss (e.g., ``"mse"``, ``"bce"``).

    Returns:
        Instantiated loss function (defaults to MSE).
    """
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "bce":
        return nn.BCELoss()
    else:
        return nn.MSELoss()
