"""Architecture contracts and provider discovery for node-fdm."""

from __future__ import annotations

from node_fdm.architectures.registry import (
    ARCHITECTURE_ENTRY_POINT_GROUP,
    ArchitectureOrigin,
    ArchitectureSpec,
    LayerSpec,
    architecture_digest,
    available,
    discover_architectures,
    get,
    get_origin,
    register,
    resolve_layer_class,
)

__all__ = [
    "ARCHITECTURE_ENTRY_POINT_GROUP",
    "ArchitectureOrigin",
    "ArchitectureSpec",
    "LayerSpec",
    "architecture_digest",
    "available",
    "discover_architectures",
    "get",
    "get_origin",
    "register",
    "resolve_layer_class",
]
