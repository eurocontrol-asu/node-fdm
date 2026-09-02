"""Typed architecture registry with Pydantic specs.

Replaces the legacy ``list[Any]`` architecture definitions with frozen
Pydantic models and a simple name-based registry.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import torch.nn as nn
from pydantic import BaseModel, Field

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
        name: Unique architecture name (e.g. ``"node_adsb_v1"``).
        x_cols: State variable column names.
        u_cols: Control input column names.
        e0_cols: Environment-at-t0 column names.
        e1_cols: Derived environment column names.
        dx_cols: Derivative columns as ``(sign, column_name)`` tuples.
        layers: Ordered list of layer specifications.
        preprocessing_fn: Optional dotted path to preprocessing function.
        segment_filter_fn: Optional dotted path to segment filter function.
        x_bounds: Physical bounds for state variables as ``{col: (min, max)}``.
        dx_bounds: Physical bounds for derivatives as ``{col: (min, max)}``.
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
    x_bounds: dict[str, tuple[float, float]] = {}
    dx_bounds: dict[str, tuple[float, float]] = {}
    derived_output_cols: list[str] = Field(default_factory=list)
    """NN-output columns whose stats must be derived analytically.

    Listed columns are computed by ``DERIVED_FEATURES`` from the dataset
    derivatives (e.g. ``fdm_a_spec_ms2`` from ``fdm_d_tas_ms2`` and
    ``fdm_gamma_rad``). The resulting per-column ``p999`` feeds the
    ``OutputDenormalizer`` scale via ``_create_structured_layer``.
    """
    nn_output_caps: dict[str, float] = Field(default_factory=dict)
    """Hard physical caps for NN outputs in ``"scaled"`` denormalize mode.

    Takes priority over the data-driven ``stats_dict[col]["p999"]`` fallback
    for the ``cap`` parameter of ``OutputDenormalizer``. ``scale`` stays
    data-driven (p999 from ``compute_stats``), so ``cap > scale`` keeps the
    tanh gradient alive between ``scale`` and ``cap`` while the cap enforces
    a hard regulatory/physical bound. Itself overridden by
    ``layer_spec.config["cap_overrides"]`` (legacy per-layer overrides).
    """
    nn_output_scale_floor_ratio: float = 0.0
    """Threshold (as fraction of unconditional p999) for the scale percentile.

    When > 0, ``compute_stats`` derives the scale of NN-output columns from
    the *conditional* p99.9 of ``|x|`` restricted to samples where
    ``|x| > floor_ratio * p999_unconditional``. Removes the dilution caused
    by long stretches of near-zero output (cruise for ``a_spec``,
    straight-flight for ``phi_bank``) and yields a scale that reflects the
    natural unit of the *active* signal.
    """


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
    discover_architectures()
    if name not in REGISTRY:
        available = sorted(REGISTRY)
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


ARCHITECTURE_ENTRY_POINT_GROUP = "node_fdm.architectures"
"""Python entry-point group used by architecture provider packages."""


class ArchitectureOrigin(BaseModel, frozen=True):
    """Distribution metadata identifying an architecture provider."""

    provider: str
    distribution: str | None = None
    version: str | None = None
    entry_point: str | None = None


ORIGINS: dict[str, ArchitectureOrigin] = {}
_DISCOVERY_COMPLETE = False


def _coerce_payload(payload: object) -> dict[Any, Any]:
    """Coerce a provider payload to a raw alias-to-spec mapping."""
    from collections.abc import Iterable, Mapping

    if isinstance(payload, Mapping):
        return dict(payload)
    if not isinstance(payload, Iterable) or isinstance(payload, (str, bytes)):
        msg = "Architecture providers must return a mapping or iterable of ArchitectureSpec."
        raise TypeError(msg)

    specs = list(payload)
    entries = {spec.name: spec for spec in specs if isinstance(spec, ArchitectureSpec)}
    if len(entries) != len(specs):
        msg = "Architecture providers must return ArchitectureSpec values."
        raise TypeError(msg)
    return entries


def _validate_entry(alias: object, spec: object) -> tuple[str, ArchitectureSpec]:
    """Type-check one raw provider entry."""
    if not isinstance(alias, str) or not alias:
        msg = "Architecture aliases must be non-empty strings."
        raise TypeError(msg)
    if not isinstance(spec, ArchitectureSpec):
        msg = f"Architecture alias {alias!r} does not reference an ArchitectureSpec."
        raise TypeError(msg)
    return alias, spec


def _assign_key(
    normalized: dict[str, ArchitectureSpec],
    key: str,
    spec: ArchitectureSpec,
    label: str,
) -> None:
    """Bind key to spec, rejecting a conflicting existing binding."""
    existing = normalized.get(key)
    if existing is not None and existing != spec:
        msg = f"Provider defines conflicting architecture {label} {key!r}."
        raise ValueError(msg)
    normalized[key] = spec


def _provider_entries(payload: object) -> dict[str, ArchitectureSpec]:
    """Normalize a provider payload to an alias-to-spec mapping."""
    normalized: dict[str, ArchitectureSpec] = {}
    for raw_alias, raw_spec in _coerce_payload(payload).items():
        alias, spec = _validate_entry(raw_alias, raw_spec)
        _assign_key(normalized, alias, spec, "alias")
        _assign_key(normalized, spec.name, spec, "name")
    return normalized


def discover_architectures() -> None:
    """Load architecture catalogs declared through Python entry points once."""
    global _DISCOVERY_COMPLETE

    if _DISCOVERY_COMPLETE:
        return

    from importlib import metadata

    discovered: list[
        tuple[metadata.EntryPoint, dict[str, ArchitectureSpec], ArchitectureOrigin]
    ] = []
    entry_points = sorted(
        metadata.entry_points(group=ARCHITECTURE_ENTRY_POINT_GROUP),
        key=lambda item: (item.name, item.value),
    )
    for entry_point in entry_points:
        provider = entry_point.load()
        if not callable(provider):
            msg = f"Architecture entry point {entry_point.name!r} is not callable."
            raise TypeError(msg)
        entries = _provider_entries(provider())
        distribution = getattr(entry_point, "dist", None)
        origin = ArchitectureOrigin(
            provider=entry_point.name,
            distribution=getattr(distribution, "name", None),
            version=getattr(distribution, "version", None),
            entry_point=entry_point.value,
        )
        discovered.append((entry_point, entries, origin))

    pending_names: set[str] = set()
    for entry_point, entries, _origin in discovered:
        collisions = (set(entries) & set(REGISTRY)) | (set(entries) & pending_names)
        if collisions:
            names = ", ".join(sorted(collisions))
            msg = f"Architecture provider {entry_point.name!r} collides on: {names}."
            raise ValueError(msg)
        pending_names.update(entries)

    for _entry_point, entries, origin in discovered:
        for name, spec in entries.items():
            REGISTRY[name] = spec
            ORIGINS[name] = origin
    _DISCOVERY_COMPLETE = True


def available() -> tuple[str, ...]:
    """Return all registered canonical names and aliases in stable order."""
    discover_architectures()
    return tuple(sorted(REGISTRY))


def get_origin(name: str) -> ArchitectureOrigin | None:
    """Return provider metadata for an architecture alias, when discoverable."""
    discover_architectures()
    return ORIGINS.get(name)


def architecture_digest(spec: ArchitectureSpec) -> str:
    """Return a deterministic SHA-256 digest of a normalized specification."""
    import hashlib
    import json

    payload = json.dumps(
        spec.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()
