"""Utility functions for BADA-based flight dynamics conversions.

CAS↔Mach and TAS↔CAS conversions using ISA atmospheric model.
Physical constants and ISA functions are imported from ``node_fdm_data``.
"""

from __future__ import annotations

import numpy as np

from node_fdm_data.physics.constants import A0, GAMMA_AIR, P0, R
from node_fdm_data.physics.isa import isa_pressure

__all__ = [
    "cas_to_mach",
    "get_phase",
    "mach_to_cas",
    "ms_to_kt",
    "tas_to_cas",
]

# Shorthand used in conversion formulas
_GM1: float = GAMMA_AIR - 1  # gamma - 1
_EXP: float = GAMMA_AIR / _GM1  # gamma / (gamma - 1)
_INV_EXP: float = _GM1 / GAMMA_AIR  # (gamma - 1) / gamma


def ms_to_kt(speed_ms: float) -> float:
    """Convert metres per second to knots.

    Args:
        speed_ms: Speed in m/s.

    Returns:
        Speed in knots.
    """
    return speed_ms * 3600.0 / 1852.0


def get_phase(hp_init: float, hp_target: float, *, threshold: float = 50.0) -> str:
    """Infer flight phase from initial and target pressure altitudes.

    Args:
        hp_init: Initial pressure altitude in feet.
        hp_target: Target pressure altitude in feet.
        threshold: Altitude difference tolerance in feet (default 50).

    Returns:
        Phase string: ``"Cruise"``, ``"Climb"``, or ``"Descent"``.
    """
    if abs(hp_init - hp_target) < threshold:
        return "Cruise"
    return "Climb" if hp_init < hp_target else "Descent"


def cas_to_mach(cas_ms: float | np.ndarray, h_m: float | np.ndarray) -> np.ndarray:
    """Convert calibrated airspeed to Mach number at altitude (ISA).

    Uses the compressible-flow relation via impact pressure.

    Args:
        cas_ms: Calibrated airspeed in m/s.
        h_m: Altitude in metres.

    Returns:
        Mach number (array).
    """
    cas = np.asarray(cas_ms, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)

    # Impact-pressure ratio at sea level (from CAS)
    qc_p0 = (1.0 + _GM1 / 2.0 * (cas / A0) ** 2) ** _EXP - 1.0

    # Static pressure at altitude
    p = np.asarray(isa_pressure(h), dtype=np.float64)

    # Impact-pressure ratio at altitude
    qc_p = qc_p0 * (P0 / p)

    # Mach from impact-pressure ratio
    mach: np.ndarray = np.sqrt((2.0 / _GM1) * ((qc_p + 1.0) ** _INV_EXP - 1.0))
    return mach


def mach_to_cas(mach: float | np.ndarray, h_m: float | np.ndarray) -> np.ndarray:
    """Convert Mach number to calibrated airspeed at altitude (ISA).

    Inverse of :func:`cas_to_mach`.

    Args:
        mach: Mach number.
        h_m: Altitude in metres.

    Returns:
        Calibrated airspeed in m/s (array).
    """
    m = np.asarray(mach, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)

    p = np.asarray(isa_pressure(h), dtype=np.float64)

    # Total-to-static pressure ratio at altitude
    pt_over_p = (1.0 + _GM1 / 2.0 * m**2) ** _EXP

    # Impact-pressure ratio at sea level
    qc_p0 = (p / P0) * (pt_over_p - 1.0)

    cas: np.ndarray = A0 * np.sqrt((2.0 / _GM1) * ((qc_p0 + 1.0) ** _INV_EXP - 1.0))
    return cas


def tas_to_cas(
    tas_ms: float | np.ndarray,
    h_m: float | np.ndarray,
    temperature_k: float | np.ndarray,
) -> np.ndarray:
    """Convert true airspeed to calibrated airspeed.

    Args:
        tas_ms: True airspeed in m/s.
        h_m: Altitude in metres.
        temperature_k: Actual temperature in Kelvin.

    Returns:
        Calibrated airspeed in m/s (array).
    """
    tas = np.asarray(tas_ms, dtype=np.float64)
    h = np.asarray(h_m, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)

    # Local speed of sound
    a = np.sqrt(GAMMA_AIR * R * t)
    m = tas / a

    p = np.asarray(isa_pressure(h), dtype=np.float64)

    # Total-to-static pressure ratio
    pt_over_p = (1.0 + _GM1 / 2.0 * m**2) ** _EXP

    # Impact-pressure ratio at sea level
    qc_p0 = (p / P0) * (pt_over_p - 1.0)

    cas: np.ndarray = A0 * np.sqrt((2.0 / _GM1) * ((qc_p0 + 1.0) ** _INV_EXP - 1.0))
    return cas
