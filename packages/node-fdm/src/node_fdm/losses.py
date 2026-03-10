"""Loss function factory for model training."""

from __future__ import annotations

import torch.nn as nn

__all__ = [
    "get_loss",
]

_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "bce": nn.BCELoss,
    "huber": nn.HuberLoss,
    "l1": nn.L1Loss,
}


def get_loss(name: str) -> nn.Module:
    """Return a loss function instance by name.

    Supported names: ``"mse"``, ``"bce"``, ``"huber"``, ``"l1"``.

    Args:
        name: Case-insensitive loss identifier.

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If *name* is not recognised.
    """
    key = name.lower()
    if key not in _LOSS_REGISTRY:
        available = sorted(_LOSS_REGISTRY)
        msg = f"Unknown loss '{name}'. Available: {available}"
        raise ValueError(msg)
    return _LOSS_REGISTRY[key]()
