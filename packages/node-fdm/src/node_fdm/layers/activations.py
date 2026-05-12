"""Activation function registry — string identifier to ``nn.Module`` class.

Used by ``TrainingConfig``/``ModelMeta`` to persist the chosen activation as
a portable JSON string while keeping the model graph builders typed.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = [
    "ACTIVATIONS",
    "DEFAULT_ACTIVATION",
    "resolve_activation",
]

DEFAULT_ACTIVATION: str = "silu"

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def resolve_activation(name: str | None) -> type[nn.Module]:
    """Map an activation identifier to its ``nn.Module`` subclass.

    Args:
        name: One of ``"silu"``, ``"relu"``, ``"gelu"``, ``"tanh"``. ``None``
            resolves to the default (``DEFAULT_ACTIVATION``).

    Returns:
        The corresponding ``nn.Module`` class.

    Raises:
        ValueError: If ``name`` is not a known activation.
    """
    key = (name or DEFAULT_ACTIVATION).lower()
    if key not in ACTIVATIONS:
        msg = f"Unknown activation '{name}'. Allowed: {sorted(ACTIVATIONS)}"
        raise ValueError(msg)
    return ACTIVATIONS[key]
