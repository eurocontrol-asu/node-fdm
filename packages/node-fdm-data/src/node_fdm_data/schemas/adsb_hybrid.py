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
    "FLIGHT_FEATURE_COLS_2_LEAN",
    "FLIGHT_FEATURE_COLS_5",
    "FLIGHT_FEATURE_COLS_6",
    "FLIGHT_FEATURE_COLS_9",
    "FLIGHT_FEATURE_SIGNS_2_LEAN",
    "FLIGHT_FEATURE_SIGNS_5",
    "FLIGHT_FEATURE_SIGNS_6",
    "FLIGHT_FEATURE_SIGNS_9",
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

FLIGHT_FEATURE_COLS_5: list[str] = [
    "dist_total_flight",
    "dist_adep_at_t0",
    "cruise_alt_max_flight",
    "wind_long_mean_flight",
    "temp_isa_dev_mean_flight",
]
FLIGHT_FEATURE_SIGNS_5: list[float] = [1.0, -1.0, -1.0, -1.0, 1.0]

FLIGHT_FEATURE_COLS_6: list[str] = [
    *FLIGHT_FEATURE_COLS_5,
    # Pre-flight Mach planned proxy: highest Mach the FMS selected during the
    # flight (clamped to physically plausible cruise range). A heavy aircraft
    # cruises at a lower planned Mach for fuel efficiency → expected sign -1.
    "mach_cruise_planned",
]
FLIGHT_FEATURE_SIGNS_6: list[float] = [*FLIGHT_FEATURE_SIGNS_5, -1.0]

# Phase 2 — dynamic mass-signature features (Experiment 01).
# All three are aggregated per flight from ADS-B columns:
#   - climb_rate_mean_climb : mean fdm_d_alt_ms over low-climb band
#     (raw_alt_m ∈ [1500, 4500] m). Heavier → lower Vz → sign -1.
#   - accel_mean_climb : mean fdm_d_tas_ms2 over same band.
#     Heavier → lower TAS-acceleration → sign -1.
#   - time_to_fl240_s : seconds from first row above 1000 m to first row
#     above 7300 m (FL240). Heavier → longer climb → sign +1.
FLIGHT_FEATURE_COLS_9: list[str] = [
    *FLIGHT_FEATURE_COLS_6,
    "climb_rate_mean_climb",
    "accel_mean_climb",
    "time_to_fl240_s",
]
FLIGHT_FEATURE_SIGNS_9: list[float] = [*FLIGHT_FEATURE_SIGNS_6, -1.0, -1.0, 1.0]

# Phase 3 — lean 2-feature subset (Experiment 04).
# Identified by the LOOCV oracle linear-regression diagnostic on QAR: the
# pair (cruise_alt_max_flight, dist_total_flight) maximises out-of-sample
# corr (0.608) on the 18 validation flights; adding more features only
# adds noise on this small validation set. Used to test whether removing
# multi-collinear / weak features lets a trained encoder reach the 0.5
# success threshold.
FLIGHT_FEATURE_COLS_2_LEAN: list[str] = [
    "dist_total_flight",
    "cruise_alt_max_flight",
]
FLIGHT_FEATURE_SIGNS_2_LEAN: list[float] = [1.0, -1.0]

A320_OEW_KG: float = 42_600.0
A320_MTOW_KG: float = 77_000.0
