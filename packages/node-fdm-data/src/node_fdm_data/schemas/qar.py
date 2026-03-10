"""Column schemas and conversion registry for the QAR architecture.

QAR data uses raw parameter names from the flight recorder (e.g.
``ALT__STD``, ``SPD__TAS``).  The ``CONVERSIONS`` dictionary normalises
these into SI-compatible names used by the Neural ODE models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from node_fdm_data.conversions import (
    deg_to_rad,
    ft_to_m,
    ftmin_to_ms,
    kgh_to_kgs,
    kt_to_ms,
    nm_to_m,
)

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "CONVERSIONS",
    "DX_COLS",
    "E0_COLS",
    "E1_COLS",
    "U_COLS",
    "X_COLS",
]

# ---------------------------------------------------------------------------
# State variables (x)
# ---------------------------------------------------------------------------
X_COLS: list[str] = [
    "NAV__GND_DIST",
    "ALT__STD",
    "ATT__FPA",
    "SPD__TAS",
]

# ---------------------------------------------------------------------------
# Control inputs (u)
# ---------------------------------------------------------------------------
U_COLS: list[str] = [
    "NAV__ALT_SEL",
    "SPD__SPD_SEL",
    "SPD__VERT_SEL",
]

# ---------------------------------------------------------------------------
# Environment at t=0 (e0)
# ---------------------------------------------------------------------------
E0_COLS: list[str] = [
    "WIND__HEAD_WIND",
    "WIND__CROSS_WIND",
    "TEMP__SAT",
    "TRAJ__DIST_TO_THR",
    "TRAJ__RWY_ELEV",
]

# ---------------------------------------------------------------------------
# Environment at t (e1) — derived quantities
# ---------------------------------------------------------------------------
E1_COLS: list[str] = [
    "ATT__VV",
    "SPD__MACH",
    "SPD__GND",
    "SPD__CAS",
    "SYS__GW",
    "ENG__N1_LEFT",
    "ENG__N1_RIGHT",
    "ATT__PITCH",
    "ATT__AOA_LH",
]

# ---------------------------------------------------------------------------
# Derivatives (dx) — sign and target column
# ---------------------------------------------------------------------------
DX_COLS: list[tuple[int, str]] = [
    (1, "SPD__GND"),
    (1, "ATT__VV"),
    (1, "d_ATT__FPA"),
    (1, "d_SPD__TAS"),
]

# ---------------------------------------------------------------------------
# Raw-column → (conversion_fn, target_name) registry
# ---------------------------------------------------------------------------
CONVERSIONS: dict[str, tuple[Callable[[str], pl.Expr], str]] = {
    "ALT__STD": (ft_to_m, "alt_m"),
    "SPD__TAS": (kt_to_ms, "tas_ms"),
    "SPD__CAS": (kt_to_ms, "cas_ms"),
    "SPD__GND": (kt_to_ms, "gs_ms"),
    "ATT__VV": (ftmin_to_ms, "vz_ms"),
    "ATT__FPA": (deg_to_rad, "fpa_rad"),
    "ATT__PITCH": (deg_to_rad, "pitch_rad"),
    "ATT__AOA_LH": (deg_to_rad, "aoa_rad"),
    "NAV__GND_DIST": (nm_to_m, "dist_m"),
    "TRAJ__DIST_TO_THR": (nm_to_m, "dist_to_thr_m"),
    "TRAJ__RWY_ELEV": (ft_to_m, "rwy_elev_m"),
    "NAV__ALT_SEL": (ft_to_m, "alt_sel_m"),
    "SPD__VERT_SEL": (ftmin_to_ms, "vz_sel_ms"),
    "WIND__HEAD_WIND": (kt_to_ms, "head_wind_ms"),
    "WIND__CROSS_WIND": (kt_to_ms, "cross_wind_ms"),
    "FUEL__FF_LEFT": (kgh_to_kgs, "ff_left_kgs"),
    "FUEL__FF_RIGHT": (kgh_to_kgs, "ff_right_kgs"),
}
