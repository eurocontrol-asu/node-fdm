"""Physics sub-package — ISA atmosphere model and physical constants."""

from __future__ import annotations

from node_fdm_data.physics.constants import (
    A0,
    FT,
    FTMIN,
    GAMMA_AIR,
    KGH,
    KT,
    NM,
    P0,
    RHO0,
    T0,
    T_K_C,
    TROPOPAUSE_ALT_M,
    G,
    L,
    R,
)
from node_fdm_data.physics.isa import isa_density, isa_pressure, isa_pressure_expr, isa_temperature
from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas, vz_to_gamma

__all__ = [
    "A0",
    "FT",
    "FTMIN",
    "GAMMA_AIR",
    "KGH",
    "KT",
    "NM",
    "P0",
    "RHO0",
    "T0",
    "TROPOPAUSE_ALT_M",
    "T_K_C",
    "G",
    "L",
    "R",
    "cas_to_tas",
    "isa_density",
    "isa_pressure",
    "isa_pressure_expr",
    "isa_temperature",
    "mach_to_tas",
    "vz_to_gamma",
]
