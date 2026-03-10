"""Shared test configuration for node-fdm tests.

Imports architecture modules to ensure they self-register in the
global ``REGISTRY`` before any test accesses ``get("opensky_2025")``.
"""

from __future__ import annotations

import node_fdm.architectures.opensky as _opensky  # noqa: F401
import node_fdm.architectures.qar as _qar  # noqa: F401
