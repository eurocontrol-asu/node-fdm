"""node-fdm-bada — BADA 4.2 aircraft performance baseline."""

from __future__ import annotations

from node_fdm_bada.aircraft_mapping import BADA_4_2_MAPPING, get_bada_identifier
from node_fdm_bada.predictor import process_single_flight
from node_fdm_bada.utils import cas_to_mach, get_phase, mach_to_cas, ms_to_kt, tas_to_cas

__all__ = [
    "BADA_4_2_MAPPING",
    "cas_to_mach",
    "get_bada_identifier",
    "get_phase",
    "mach_to_cas",
    "ms_to_kt",
    "process_single_flight",
    "tas_to_cas",
]
