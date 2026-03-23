"""Architecture resolver — dispatches ``--arch`` flag to schema + preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
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
        name: Architecture registry name (e.g. ``"opensky_2025"``).
        x_cols: State column names.
        u_cols: Control column names.
        e0_cols: Environment column names.
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


#: Reverse mapping from architecture registry name to CLI arch key.
ARCH_BY_NAME: dict[str, str] = {
    "opensky_2025": "opensky",
    "opensky_v2": "opensky_v2",
    "qar": "qar",
    "node_adsb_v1": "adsb",
}


def resolve_architecture(arch: str) -> ArchitectureInfo:
    """Resolve an architecture name to its schema and preprocessing components.

    Args:
        arch: Architecture identifier — ``"opensky"``, ``"opensky_v2"``, ``"qar"``, or ``"adsb"``.

    Returns:
        Fully resolved ``ArchitectureInfo``.

    Raises:
        ValueError: If *arch* is not a supported architecture.
    """
    match arch:
        case "opensky":
            from node_fdm_data.schemas.opensky import DX_COLS, E0_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="opensky_2025",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.opensky",
            )
        case "opensky_v2":
            from node_fdm_data.schemas.opensky_v2 import DX_COLS, E0_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="opensky_v2",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.opensky_v2",
            )
        case "qar":
            from node_fdm_data.schemas.qar import DX_COLS, E0_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="qar",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.qar",
            )
        case "adsb":
            from node_fdm_data.schemas.adsb import DX_COLS, E0_COLS, U_COLS, X_COLS

            return ArchitectureInfo(
                name="node_adsb_v1",
                x_cols=X_COLS,
                u_cols=U_COLS,
                e0_cols=E0_COLS,
                dx_cols=DX_COLS,
                preprocessing_fn=None,
                segment_filter_fn=None,
                architecture_import="node_fdm.architectures.adsb",
            )
        case _:
            msg = (
                f"Unknown architecture: {arch!r}. "
                "Supported: 'opensky', 'opensky_v2', 'qar', 'adsb'."
            )
            raise ValueError(msg)
