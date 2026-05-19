"""Architecture resolver — dispatches ``--arch`` flag to schema + preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ARCH_BY_NAME",
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
        architecture_import: Dotted import path to trigger auto-registration.
    """

    name: str
    x_cols: list[str]
    u_cols: list[str]
    e0_cols: list[str]
    dx_cols: list[tuple[int, str]]
    preprocessing_fn: Any
    segment_filter_fn: Any
    architecture_import: str
    e1_cols: list[str] = field(default_factory=list)


#: Reverse mapping from architecture registry name to CLI arch key.
ARCH_BY_NAME: dict[str, str] = {
    "qar": "qar",
    "node_adsb_v1": "adsb",
    "node_adsb_hybrid_v1": "adsb_hybrid",
}


def resolve_architecture(arch: str) -> ArchitectureInfo:
    """Resolve an architecture name to its schema and preprocessing components.

    Args:
        arch: Architecture identifier — ``"qar"``, ``"adsb"``, or ``"adsb_hybrid"``.

    Returns:
        Fully resolved ``ArchitectureInfo``.

    Raises:
        ValueError: If *arch* is not a supported architecture.
    """
    match arch:
        case "qar":
            from node_fdm_data.schemas.qar import DX_COLS, E0_COLS, E1_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="qar",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                e1_cols=E1_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.qar",
            )
        case "adsb":
            from node_fdm_data.schemas.adsb import DX_COLS, E0_COLS, E1_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="node_adsb_v1",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                e1_cols=E1_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.adsb",
            )
        case "adsb_hybrid":
            from node_fdm_data.schemas.adsb_hybrid import (
                DX_COLS,
                E0_COLS,
                E1_COLS,
                U_COLS,
                X_COLS,
            )

            return ArchitectureInfo(
                name="node_adsb_hybrid_v1",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                e1_cols=E1_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.adsb_hybrid",
            )
        case _:
            msg = f"Unknown architecture: {arch!r}. Supported: 'qar', 'adsb', 'adsb_hybrid'."
            raise ValueError(msg)
