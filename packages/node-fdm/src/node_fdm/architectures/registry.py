"""Typed architecture registry with Pydantic specs.

Replaces the legacy ``list[Any]`` architecture definitions with frozen
Pydantic models and a simple name-based registry.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import torch.nn as nn
from pydantic import BaseModel

__all__ = [
    "REGISTRY",
    "ArchitectureSpec",
    "LayerSpec",
    "get",
    "register",
    "resolve_layer_class",
]

REGISTRY: dict[str, ArchitectureSpec] = {}


class LayerSpec(BaseModel, frozen=True):
    """Specification for a single neural network layer in an architecture.

    Attributes:
        name: Human-readable layer identifier (e.g. ``"trajectory"``).
        layer_class: Dotted import path to the ``nn.Module`` class.
        input_cols: Column names consumed by this layer.
        output_cols: Column names produced by this layer.
        trainable: Whether the layer uses a structured (trainable) backbone.
    """

    name: str
    layer_class: str
    input_cols: list[str]
    output_cols: list[str]
    trainable: bool = True
    config: dict[str, object] = {}


class ArchitectureSpec(BaseModel, frozen=True):
    """Full architecture specification for a flight dynamics model.

    Attributes:
        name: Unique architecture name (e.g. ``"opensky_2025"``).
        x_cols: State variable column names.
        u_cols: Control input column names.
        e0_cols: Environment-at-t0 column names.
        e1_cols: Derived environment column names.
        dx_cols: Derivative columns as ``(sign, column_name)`` tuples.
        layers: Ordered list of layer specifications.
        preprocessing_fn: Optional dotted path to preprocessing function.
        segment_filter_fn: Optional dotted path to segment filter function.
    """

    name: str
    x_cols: list[str]
    u_cols: list[str]
    e0_cols: list[str]
    e1_cols: list[str]
    dx_cols: list[tuple[int, str]]
    layers: list[LayerSpec]
    preprocessing_fn: str | None = None
    segment_filter_fn: str | None = None


def register(spec: ArchitectureSpec) -> None:
    """Register an architecture spec in the global registry.

    Args:
        spec: Architecture specification to register.

    Warns:
        UserWarning: If an architecture with the same name is already registered.
    """
    if spec.name in REGISTRY:
        warnings.warn(
            f"Architecture '{spec.name}' already registered, overwriting.",
            UserWarning,
            stacklevel=2,
        )
    REGISTRY[spec.name] = spec


def get(name: str) -> ArchitectureSpec:
    """Retrieve a registered architecture spec by name.

    Args:
        name: Architecture name.

    Returns:
        The registered ``ArchitectureSpec``.

    Raises:
        ValueError: If no architecture with that name is registered.
    """
    if name not in REGISTRY:
        available = list(REGISTRY.keys())
        msg = f"Unknown architecture '{name}'. Available: {available}"
        raise ValueError(msg)
    return REGISTRY[name]


def resolve_layer_class(dotted_path: str) -> type[nn.Module]:
    """Import and return a layer class from a dotted module path.

    Args:
        dotted_path: Full dotted path like ``"node_fdm.layers.structured.StructuredLayer"``.

    Returns:
        The resolved ``nn.Module`` subclass.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls: Any = getattr(module, class_name)
    return cls  # type: ignore[no-any-return]
