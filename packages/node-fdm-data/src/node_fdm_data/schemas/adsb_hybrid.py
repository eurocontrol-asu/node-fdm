"""Phase 1 extended ADS-B schema with mass encoder support.

This schema keeps the baseline ADS-B controls and environment columns while
adding a fifth state dimension for aircraft mass. It also documents the
MassEncoder flight feature order, expected monotonic signs, and A320 TCDS mass
bounds exposed by this module.
"""

from __future__ import annotations

from node_fdm_data.schemas import adsb

__all__ = [
    "A320_MTOW_KG",
    "A320_OEW_KG",
    "DX_COLS",
    "E0_COLS",
    "E1_COLS",
    "FLIGHT_FEATURE_COLS",
    "FLIGHT_FEATURE_SIGNS",
    "U_COLS",
    "U_ODE_COLS",
    "X_COLS",
]

X_COLS: list[str] = [*adsb.X_COLS, "fdm_mass_kg"]
DX_COLS: list[tuple[int, str]] = [*adsb.DX_COLS, (0, "fdm_d_mass_kgs")]

U_COLS: list[str] = adsb.U_COLS
U_ODE_COLS: list[str] = adsb.U_ODE_COLS
E0_COLS: list[str] = adsb.E0_COLS
E1_COLS: list[str] = adsb.E1_COLS

FLIGHT_FEATURE_COLS: list[str] = [
    "dist_total_flight",
    "dist_adep_at_t0",
    "cruise_alt_max_flight",
    "wind_long_mean_flight",
    "temp_isa_dev_mean_flight",
]
FLIGHT_FEATURE_SIGNS: list[float] = [1.0, -1.0, -1.0, -1.0, 1.0]

A320_OEW_KG: float = 42_600.0
A320_MTOW_KG: float = 77_000.0
