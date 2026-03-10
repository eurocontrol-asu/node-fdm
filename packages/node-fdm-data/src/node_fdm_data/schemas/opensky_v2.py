"""Column schemas for the OpenSky V2 architecture (lateral extension).

Extends the base OpenSky 2025 schema with lateral state variables
(latitude, longitude, track, track_sel) for the longitudinal+lateral
flight dynamics model.

Column names are plain strings.  The ``CONVERSIONS`` dictionary maps
raw column names to ``(conversion_fn, target_name)`` tuples.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from node_fdm_data.conversions import (
    celsius_to_kelvin,
    ft_to_m,
    ftmin_to_ms,
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
# State variables (x) — integrated by the Neural ODE
# Extends v1 with latitude, longitude, track_sel (from lateral computations)
# ---------------------------------------------------------------------------
X_COLS: list[str] = [
    "distance_m",
    "altitude_ft",
    "gamma_rad",
    "tas_kt",
    "latitude",
    "longitude",
    "track_sel",
]

# ---------------------------------------------------------------------------
# Control inputs (u) — selected / target values
# ---------------------------------------------------------------------------
U_COLS: list[str] = [
    "alt_sel_ft",
    "mach_sel",
    "cas_sel_kt",
    "vz_sel_ftmin",
]

# ---------------------------------------------------------------------------
# Environment at t=0 (e0) — fed to trajectory layer
# ---------------------------------------------------------------------------
E0_COLS: list[str] = [
    "long_wind_kt",
    "adep_dist_nm",
    "ades_dist_nm",
    "temperature_K",
]

# ---------------------------------------------------------------------------
# Environment at t (e1) — derived by trajectory layer
# ---------------------------------------------------------------------------
E1_COLS: list[str] = [
    "vz_ftmin",
    "mach",
    "gs_kt",
    "cas_kt",
    "alt_diff_ft",
    "track",
]

# ---------------------------------------------------------------------------
# Derivatives (dx) — sign and target column
# ---------------------------------------------------------------------------
DX_COLS: list[tuple[int, str]] = [
    (1, "gs_kt"),
    (1, "vz_ftmin"),
    (1, "d_gamma_rad"),
    (1, "d_tas_kt"),
    (1, "d_track"),
]

# ---------------------------------------------------------------------------
# Raw-column → (conversion_fn, target_name) registry
# ---------------------------------------------------------------------------
CONVERSIONS: dict[str, tuple[Callable[[str], pl.Expr], str]] = {
    "altitude": (ft_to_m, "altitude_m"),
    "TAS": (kt_to_ms, "tas_ms"),
    "CAS": (kt_to_ms, "cas_ms"),
    "groundspeed": (kt_to_ms, "gs_ms"),
    "vertical_rate": (ftmin_to_ms, "vz_ms"),
    "long_wind": (kt_to_ms, "long_wind_ms"),
    "temperature": (celsius_to_kelvin, "temperature_K"),
    "adep_dist": (nm_to_m, "adep_dist_m"),
    "ades_dist": (nm_to_m, "ades_dist_m"),
    "selected_mcp": (ft_to_m, "alt_sel_m"),
}
