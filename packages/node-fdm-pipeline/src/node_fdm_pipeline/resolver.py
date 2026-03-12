"""Architecture resolver — dispatches ``--arch`` flag to schema + preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ArchitectureInfo",
    "resolve_architecture",
]


@dataclass(frozen=True)
class ArchitectureInfo:
    """Resolved architecture with schema columns and preprocessing functions.

    Attributes:
        name: Architecture registry name (e.g. ``"opensky_2025"``).
        x_cols: State column names.
        u_cols: Control column names.
        e0_cols: Environment column names.
        dx_cols: Derivative column specs ``(sign, col_name)``.
        preprocessing_fn: Flight preprocessing function.
        segment_filter_fn: Optional segment filter function.
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


def resolve_architecture(arch: str) -> ArchitectureInfo:
    """Resolve an architecture name to its schema and preprocessing components.

    Args:
        arch: Architecture identifier — ``"opensky"``, ``"opensky_v2"``, or ``"qar"``.

    Returns:
        Fully resolved ``ArchitectureInfo``.

    Raises:
        ValueError: If *arch* is not a supported architecture.
    """
    match arch:
        case "opensky":
            from node_fdm_data.preprocessing.opensky import (
                segment_filtering,
                training_preprocessing,
            )
            from node_fdm_data.schemas.opensky import DX_COLS, E0_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="opensky_2025",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=training_preprocessing,
                segment_filter_fn=segment_filtering,
                architecture_import="node_fdm.architectures.opensky",
            )
        case "opensky_v2":
            from node_fdm_data.preprocessing.opensky import (
                segment_filtering,
                training_preprocessing,
            )
            from node_fdm_data.schemas.opensky_v2 import DX_COLS, E0_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="opensky_v2",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=training_preprocessing,
                segment_filter_fn=segment_filtering,
                architecture_import="node_fdm.architectures.opensky_v2",
            )
        case "qar":
            from node_fdm_data.preprocessing.qar import flight_processing as qar_processing
            from node_fdm_data.schemas.qar import DX_COLS, E0_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="qar",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=qar_processing,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.qar",
            )
        case _:
            msg = f"Unknown architecture: {arch!r}. Supported: 'opensky', 'opensky_v2', 'qar'."
            raise ValueError(msg)
