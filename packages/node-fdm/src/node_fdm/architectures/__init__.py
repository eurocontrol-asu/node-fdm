"""Architecture registry and specs (ADS-B v1, QAR)."""

from __future__ import annotations

from node_fdm.architectures import adsb as adsb
from node_fdm.architectures import adsb_hybrid as adsb_hybrid
from node_fdm.architectures import adsb_hybrid_v2 as adsb_hybrid_v2
from node_fdm.architectures import qar as qar

__all__ = [
    "adsb",
    "adsb_hybrid",
    "adsb_hybrid_v2",
    "qar",
    "registry",
]
