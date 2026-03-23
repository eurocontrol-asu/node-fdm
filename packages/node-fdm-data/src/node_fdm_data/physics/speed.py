"""Speed conversions — Mach/CAS → TAS.

Provides ``mach_to_tas`` and ``cas_to_tas`` using the ISA atmosphere model
and isentropic compressible-flow relations for subsonic flight.
"""

from __future__ import annotations

import numpy as np

from node_fdm_data.physics.constants import A0, GAMMA_AIR, P0, R
from node_fdm_data.physics.isa import isa_pressure, isa_temperature

__all__ = ["cas_to_tas", "mach_to_tas"]

_GM1_OVER_2: float = (GAMMA_AIR - 1.0) / 2.0  # 0.2
_G_OVER_GM1: float = GAMMA_AIR / (GAMMA_AIR - 1.0)  # 3.5
_INV_G_OVER_GM1: float = 1.0 / _G_OVER_GM1  # 2/7


def mach_to_tas(mach: float | np.ndarray, h_m: float | np.ndarray) -> float | np.ndarray:
    """Convert Mach number to True Airspeed (m/s) at altitude *h_m* (metres).

    ``TAS = M * sqrt(gamma * R * T)`` where *T* is the ISA temperature at *h_m*.
    """
    m = np.asarray(mach, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    t = np.asarray(isa_temperature(h), dtype=np.float64)
    # Propagate NaN from altitude (isa_temperature maps NaN→stratosphere)
    t = np.where(np.isnan(h), np.nan, t)
    a_local = np.sqrt(GAMMA_AIR * R * t)
    return m * a_local


def cas_to_tas(cas: float | np.ndarray, h_m: float | np.ndarray) -> float | np.ndarray:
    """Convert Calibrated Airspeed (m/s) to True Airspeed (m/s) at altitude *h_m* (metres).

    Uses the isentropic compressible-flow relation (subsonic):

    1. CAS → impact pressure *qc* at sea level.
    2. *qc* + static pressure at altitude → local Mach.
    3. Mach → TAS via ``mach_to_tas``.

    Negative CAS values return NaN.
    """
    cas_arr = np.asarray(cas, dtype=np.float64)
    p = np.asarray(isa_pressure(h_m), dtype=np.float64)

    # CAS → impact pressure (sea-level isentropic relation)
    cas_ratio = cas_arr / A0
    # Guard negative CAS: set to NaN
    cas_ratio = np.where(cas_arr < 0.0, np.nan, cas_ratio)
    qc = P0 * ((1.0 + _GM1_OVER_2 * cas_ratio**2) ** _G_OVER_GM1 - 1.0)

    # Impact pressure → Mach at altitude
    mach = np.sqrt((2.0 / (GAMMA_AIR - 1.0)) * ((qc / p + 1.0) ** _INV_G_OVER_GM1 - 1.0))

    return mach_to_tas(mach, h_m)
