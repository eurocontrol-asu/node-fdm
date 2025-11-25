#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for BADA-based flight dynamics conversions."""

from typing import Any

import numpy as np
from utils.physics.constants import gamma_ratio, a0, p0, R, T0


def ms_to_kt(el: float) -> float:
    """Convert meters per second to knots.

    Args:
        el: Speed in meters per second.

    Returns:
        Speed in knots.
    """
    return el * 3600 / 1852


def get_phase(hp_init: float, hp_target: float) -> str:
    """Infer flight phase from initial and target pressure altitudes.

    Args:
        hp_init: Initial pressure altitude in feet.
        hp_target: Target pressure altitude in feet.

    Returns:
        Phase string among ``Cruise``, ``Climb``, or ``Descent``.
    """
    if np.abs(hp_init - hp_target) < 50:
        return "Cruise"
    else:
        if hp_init < hp_target:
            return "Climb"
        else:
            return "Descent"


def cas_to_mach(CAS: Any, h: Any) -> np.ndarray:
    """Convert calibrated airspeed to Mach number at altitude [ISA].

    Args:
        CAS: Calibrated airspeed in meters per second.
        h: Altitude in meters.

    Returns:
        Mach number array.
    """
    CAS = np.asarray(CAS)
    h = np.asarray(h)
    qc_p0 = (1 + (gamma_ratio - 1) / 2 * (CAS / a0) ** 2) ** (
        gamma_ratio / (gamma_ratio - 1)
    ) - 1

    p = isa_pressure(h)

    qc_p = qc_p0 * (p0 / p)

    M = np.sqrt(
        (2 / (gamma_ratio - 1))
        * (((qc_p + 1) ** ((gamma_ratio - 1) / gamma_ratio)) - 1)
    )
    return M


def tas_to_cas(TAS: Any, h: Any, T: Any) -> np.ndarray:
    """Convert true airspeed to calibrated airspeed at altitude with temperature.

    Args:
        TAS: True airspeed in meters per second.
        h: Altitude in meters.
        T: Temperature in Kelvin.

    Returns:
        Calibrated airspeed in meters per second.
    """
    a = np.sqrt(gamma_ratio * R * T)
    M = TAS / a

    p = isa_pressure(h)

    pt_over_p = (1 + (gamma_ratio - 1) / 2 * M**2) ** (gamma_ratio / (gamma_ratio - 1))

    qc_p0 = (p / p0) * (pt_over_p - 1)

    CAS = a0 * np.sqrt(
        (2 / (gamma_ratio - 1))
        * (((qc_p0 + 1) ** ((gamma_ratio - 1) / gamma_ratio)) - 1)
    )
    return CAS


def isa_temperature(altitude_m: float) -> float:
    """Compute ISA temperature (K) given altitude in meters.

    Args:
        altitude_m: Altitude in meters.

    Returns:
        Temperature in Kelvin at the specified altitude.
    """

    T0_c = 15.0
    altitude_km = altitude_m / 1000.0
    lapse_rate = -6.5

    if altitude_km <= 11:
        temperature = T0_c + lapse_rate * altitude_km
    elif 11 < altitude_km <= 20:
        temperature = T0_c + lapse_rate * 11
    else:
        raise ValueError(
            "Altitude out of range. This model only supports altitudes up to 20 km."
        )

    return temperature + 273.15


def isa_pressure(h: Any) -> np.ndarray:
    """Compute ISA pressure (Pa) for altitude in meters.

    Args:
        h: Altitude in meters (scalar or array).

    Returns:
        Static pressure in Pascals.
    """
    h = np.asarray(h)
    T_tropo = T0 - 0.0065 * h
    p_tropo = p0 * (T_tropo / T0) ** (9.80665 / (0.0065 * R))
    T_strato = 216.65
    p_strato = 22632.06 * np.exp(-9.80665 * (h - 11000) / (R * T_strato))
    return np.where(h <= 11000, p_tropo, p_strato)
