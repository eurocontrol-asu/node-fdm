"""Unit conversion functions as pure Polars expressions.

Each function takes a column name and returns a ``pl.Expr`` that performs
the conversion.  This replaces the legacy ``Column`` / ``Unit`` class
pattern with simple, composable expressions.

Example::

    df.with_columns(ft_to_m("altitude").alias("altitude_m"))
"""

from __future__ import annotations

import math

import polars as pl

__all__ = [
    "celsius_to_kelvin",
    "deg_to_rad",
    "ft_to_m",
    "ftmin_to_ms",
    "kelvin_to_celsius",
    "kgh_to_kgs",
    "kgs_to_kgh",
    "kt_to_ms",
    "m_to_ft",
    "m_to_nm",
    "ms_to_ftmin",
    "ms_to_kt",
    "nm_to_m",
    "rad_to_deg",
]

# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

_FT_TO_M = 0.3048
_NM_TO_M = 1852.0


def ft_to_m(col: str) -> pl.Expr:
    """Convert feet → metres."""
    return pl.col(col) * _FT_TO_M


def m_to_ft(col: str) -> pl.Expr:
    """Convert metres → feet."""
    return pl.col(col) / _FT_TO_M


def nm_to_m(col: str) -> pl.Expr:
    """Convert nautical miles → metres."""
    return pl.col(col) * _NM_TO_M


def m_to_nm(col: str) -> pl.Expr:
    """Convert metres → nautical miles."""
    return pl.col(col) / _NM_TO_M


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

_KT_TO_MS = 0.514444
_FTMIN_TO_MS = _FT_TO_M / 60.0


def kt_to_ms(col: str) -> pl.Expr:
    """Convert knots → m/s."""
    return pl.col(col) * _KT_TO_MS


def ms_to_kt(col: str) -> pl.Expr:
    """Convert m/s → knots."""
    return pl.col(col) / _KT_TO_MS


def ftmin_to_ms(col: str) -> pl.Expr:
    """Convert ft/min → m/s."""
    return pl.col(col) * _FTMIN_TO_MS


def ms_to_ftmin(col: str) -> pl.Expr:
    """Convert m/s → ft/min."""
    return pl.col(col) / _FTMIN_TO_MS


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

_T_K_C = 273.15


def celsius_to_kelvin(col: str) -> pl.Expr:
    """Convert °C → Kelvin."""
    return pl.col(col) + _T_K_C


def kelvin_to_celsius(col: str) -> pl.Expr:
    """Convert Kelvin → °C."""
    return pl.col(col) - _T_K_C


# ---------------------------------------------------------------------------
# Angle
# ---------------------------------------------------------------------------


_DEG_TO_RAD = math.pi / 180.0


def deg_to_rad(col: str) -> pl.Expr:
    """Convert degrees → radians."""
    return pl.col(col) * _DEG_TO_RAD


def rad_to_deg(col: str) -> pl.Expr:
    """Convert radians → degrees."""
    return pl.col(col) / _DEG_TO_RAD


# ---------------------------------------------------------------------------
# Mass flow
# ---------------------------------------------------------------------------

_KGH_TO_KGS = 1.0 / 3600.0


def kgh_to_kgs(col: str) -> pl.Expr:
    """Convert kg/h → kg/s."""
    return pl.col(col) * _KGH_TO_KGS


def kgs_to_kgh(col: str) -> pl.Expr:
    """Convert kg/s → kg/h."""
    return pl.col(col) / _KGH_TO_KGS
