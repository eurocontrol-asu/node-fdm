"""Architecture resolver — dispatches ``--arch`` flag to schema + preprocessing."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ArchitectureInfo",
    "resolve_architecture",
]


@dataclass(frozen=True)
class ArchitectureInfo:
    """Resolved architecture with schema columns and preprocessing functions.

    Attributes:
        name: Architecture registry name (e.g. ``"node_adsb_v1"``).
        x_cols: State column names.
        u_cols: Control column names.
        e0_cols: Environment column names.
        e1_cols: Derived environment column names (computed by physics layers).
        dx_cols: Derivative column specs ``(sign, col_name)``.
        preprocessing_fn: Flight preprocessing function (used by predict/stats).
        segment_filter_fn: Optional segment filter function (used by predict/stats).
    """

    name: str
    x_cols: list[str]
    u_cols: list[str]
    e0_cols: list[str]
    dx_cols: list[tuple[int, str]]
    preprocessing_fn: Any
    segment_filter_fn: Any
    e1_cols: list[str] = field(default_factory=list)


#: Reverse mapping from architecture registry name to CLI arch key.


def _resolve_optional_callable(dotted_path: str | None) -> Any:
    """Resolve an optional dotted callable path declared by an architecture."""
    if dotted_path is None:
        return None
    module_path, attribute = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    resolved = getattr(module, attribute)
    if not callable(resolved):
        msg = f"Architecture hook {dotted_path!r} is not callable."
        raise TypeError(msg)
    return resolved


def resolve_architecture(arch: str) -> ArchitectureInfo:
    """Resolve any installed architecture provider alias."""
    from node_fdm.architectures import available, get

    try:
        spec = get(arch)
    except ValueError as exc:
        supported = ", ".join(available()) or "none installed"
        msg = f"Unknown architecture {arch!r}. Available: {supported}."
        raise ValueError(msg) from exc

    return ArchitectureInfo(
        name=spec.name,
        x_cols=list(spec.x_cols),
        u_cols=list(spec.u_cols),
        e0_cols=list(spec.e0_cols),
        e1_cols=list(spec.e1_cols),
        dx_cols=list(spec.dx_cols),
        preprocessing_fn=_resolve_optional_callable(spec.preprocessing_fn),
        segment_filter_fn=_resolve_optional_callable(spec.segment_filter_fn),
    )
