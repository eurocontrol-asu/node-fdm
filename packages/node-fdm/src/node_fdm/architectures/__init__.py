"""Architecture registry and specs (OpenSky 2025, ADS-B v1, QAR)."""

from __future__ import annotations

from node_fdm.architectures import adsb as adsb
from node_fdm.architectures import opensky as opensky
from node_fdm.architectures import opensky_v2 as opensky_v2
from node_fdm.architectures import qar as qar

__all__ = [
    "adsb",
    "opensky",
    "opensky_v2",
    "qar",
    "registry",
]
