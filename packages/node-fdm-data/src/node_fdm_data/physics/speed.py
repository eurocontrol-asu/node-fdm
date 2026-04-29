"""Speed conversions — Mach/CAS → TAS.

Provides ``mach_to_tas`` and ``cas_to_tas`` using the ISA atmosphere model
and isentropic compressible-flow relations for subsonic flight.
"""

from __future__ import annotations

import numpy as np

from node_fdm_data.physics.constants import A0, GAMMA_AIR, P0, R
from node_fdm_data.physics.isa import isa_pressure, isa_temperature

__all__ = [
    "cas_to_tas",
    "cas_to_tas_real",
    "mach_to_tas",
    "mach_to_tas_real",
    "tas_to_cas",
    "vz_to_gamma",
]

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


def mach_to_tas_real(
    mach: float | np.ndarray,
    temp_k: float | np.ndarray,
) -> float | np.ndarray:
    """Convert Mach to True Airspeed (m/s) using real static temperature *temp_k*.

    Same relation as :func:`mach_to_tas` but with the temperature provided
    explicitly (e.g. from ERA5 reanalysis) instead of derived from ISA.
    """
    m = np.asarray(mach, dtype=np.float64)
    t = np.asarray(temp_k, dtype=np.float64)
    a_local = np.sqrt(GAMMA_AIR * R * t)
    return m * a_local


def cas_to_tas_real(
    cas: float | np.ndarray,
    h_m: float | np.ndarray,
    temp_k: float | np.ndarray,
) -> float | np.ndarray:
    """Convert CAS (m/s) to TAS (m/s) using real temperature *temp_k* at altitude *h_m*.

    Same isentropic compressible-flow chain as :func:`cas_to_tas` but the
    final Mach→TAS step uses the supplied real temperature rather than ISA.
    Static pressure for the impact-pressure step still comes from the ISA model.
    """
    cas_arr = np.asarray(cas, dtype=np.float64)
    p = np.asarray(isa_pressure(h_m), dtype=np.float64)

    cas_ratio = cas_arr / A0
    cas_ratio = np.where(cas_arr < 0.0, np.nan, cas_ratio)
    qc = P0 * ((1.0 + _GM1_OVER_2 * cas_ratio**2) ** _G_OVER_GM1 - 1.0)

    mach = np.sqrt((2.0 / (GAMMA_AIR - 1.0)) * ((qc / p + 1.0) ** _INV_G_OVER_GM1 - 1.0))
    return mach_to_tas_real(mach, temp_k)


def tas_to_cas(tas: float | np.ndarray, h_m: float | np.ndarray) -> float | np.ndarray:
    """Convert True Airspeed (m/s) to Calibrated Airspeed (m/s) at altitude *h_m* (metres).

    Inverse of ``cas_to_tas``: TAS → Mach → impact pressure → CAS.
    """
    tas_arr = np.asarray(tas, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    t = np.asarray(isa_temperature(h), dtype=np.float64)
    t = np.where(np.isnan(h), np.nan, t)
    a_local = np.sqrt(GAMMA_AIR * R * t)

    # TAS → Mach
    mach = tas_arr / a_local

    # Mach → impact pressure at altitude
    p = np.asarray(isa_pressure(h), dtype=np.float64)
    qc = p * ((1.0 + _GM1_OVER_2 * mach**2) ** _G_OVER_GM1 - 1.0)

    # Impact pressure → CAS (sea-level isentropic relation, inverse)
    cas = A0 * np.sqrt((2.0 / (GAMMA_AIR - 1.0)) * ((qc / P0 + 1.0) ** _INV_G_OVER_GM1 - 1.0))
    return np.where(tas_arr < 0.0, np.nan, cas)


def vz_to_gamma(
    vz: float | np.ndarray,
    tas: float | np.ndarray,
) -> float | np.ndarray:
    """Convert vertical speed to flight-path angle (radians).

    ``gamma = arcsin(clamp(vz / tas, -1, 1))``

    Returns NaN when *tas* is zero or any input is NaN.
    Clamps the ratio to [-1, 1] when |vz| > tas.
    """
    vz_arr = np.asarray(vz, dtype=np.float64)
    tas_arr = np.asarray(tas, dtype=np.float64)
    safe_tas = np.where(tas_arr == 0.0, np.nan, tas_arr)
    ratio = np.clip(vz_arr / safe_tas, -1.0, 1.0)
    return np.arcsin(ratio)
