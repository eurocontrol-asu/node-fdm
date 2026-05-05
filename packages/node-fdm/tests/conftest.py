"""Shared test configuration for node-fdm tests.

Imports architecture modules to ensure they self-register in the
global ``REGISTRY`` before any test accesses ``get("node_adsb_v1")``.
"""

from __future__ import annotations

import node_fdm.architectures.adsb as _adsb  # noqa: F401
import node_fdm.architectures.qar as _qar  # noqa: F401
