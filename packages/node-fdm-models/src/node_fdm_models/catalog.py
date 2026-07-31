"""Entry-point provider for the official node-fdm architecture catalog."""

from __future__ import annotations

from node_fdm.architectures import ArchitectureSpec
from node_fdm_models.architectures import NODE_ADSB_V1, QAR

__all__ = ["architectures"]


def architectures() -> dict[str, ArchitectureSpec]:
    """Return official architectures keyed by stable user-facing aliases."""
    return {
        "adsb": NODE_ADSB_V1,
        NODE_ADSB_V1.name: NODE_ADSB_V1,
        QAR.name: QAR,
    }
