"""International Standard Atmosphere (ISA) model.

Provides pressure, temperature, and density as functions of geometric
altitude.  Valid for the troposphere (0-11 km) and lower stratosphere.

Numpy variants operate on scalars/arrays; the ``_expr`` variant returns
a ``pl.Expr`` for use inside Polars ``with_columns``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_data.physics.constants import P0, T0, TROPOPAUSE_ALT_M, G, L, R

__all__ = [
    "isa_density",
    "isa_pressure",
    "isa_pressure_expr",
    "isa_temperature",
]

# Pre-computed stratosphere constants
_T_TROPOPAUSE: float = T0 - L * TROPOPAUSE_ALT_M  # ≈ 216.65 K
_P_TROPOPAUSE: float = P0 * (_T_TROPOPAUSE / T0) ** (G / (R * L))  # ≈ 22 632 Pa


def isa_temperature(h_m: float | np.ndarray) -> float | np.ndarray:
    """ISA temperature (K) at geometric altitude *h_m* (metres).

    Troposphere (h <= 11 000 m): linear lapse ``T = T0 - L*h``.
    Stratosphere (h > 11 000 m): isothermal at ≈ 216.65 K.
    """
    h = np.asarray(h_m, dtype=np.float64)
    return np.where(h <= TROPOPAUSE_ALT_M, T0 - L * h, _T_TROPOPAUSE)


def isa_pressure(h_m: float | np.ndarray) -> float | np.ndarray:
    """ISA static pressure (Pa) at geometric altitude *h_m* (metres).

    Troposphere: barometric formula with lapse rate.
    Stratosphere: exponential decay at constant temperature.
    """
    h = np.asarray(h_m, dtype=np.float64)
    t_tropo = T0 - L * h
    p_tropo = P0 * (t_tropo / T0) ** (G / (R * L))
    p_strato = _P_TROPOPAUSE * np.exp(-G * (h - TROPOPAUSE_ALT_M) / (R * _T_TROPOPAUSE))
    return np.where(h <= TROPOPAUSE_ALT_M, p_tropo, p_strato)


def isa_pressure_expr(h_m: str | pl.Expr) -> pl.Expr:
    """ISA static pressure (Pa) as a Polars expression.

    Equivalent to :func:`isa_pressure` but operates lazily inside
    ``with_columns``.  Null altitudes propagate as null.

    Args:
        h_m: Column name or expression giving geometric altitude in
            **metres**.

    Returns:
        A ``pl.Expr`` evaluating to pressure in **Pa**.
    """
    h = pl.col(h_m) if isinstance(h_m, str) else h_m
    t_tropo = T0 - L * h
    p_tropo = P0 * (t_tropo / T0).pow(G / (R * L))
    p_strato = _P_TROPOPAUSE * (-G * (h - TROPOPAUSE_ALT_M) / (R * _T_TROPOPAUSE)).exp()
    return pl.when(h <= TROPOPAUSE_ALT_M).then(p_tropo).otherwise(p_strato)


def isa_density(h_m: float | np.ndarray) -> float | np.ndarray:
    """ISA air density (kg/m³) at geometric altitude *h_m* (metres).

    Derived from ideal-gas law: ``rho = p / (R * T)``.
    """
    p = np.asarray(isa_pressure(h_m), dtype=np.float64)
    t = np.asarray(isa_temperature(h_m), dtype=np.float64)
    return p / (R * t)
