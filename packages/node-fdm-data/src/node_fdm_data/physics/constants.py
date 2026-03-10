"""Physical and environmental constants for aviation calculations.

All constants use SI units and UPPER_CASE naming.  Lookup dictionaries
for discrete QAR signals (gear, flap, speed-brake, on-ground, FMA) are
also provided.
"""

from __future__ import annotations

__all__ = [
    "A0",
    "FT",
    "FTMIN",
    "GAMMA_AIR",
    "KGH",
    "KT",
    "NM",
    "P0",
    "RHO0",
    "T0",
    "TROPOPAUSE_ALT_M",
    "T_K_C",
    "G",
    "L",
    "R",
    "flap_dict",
    "fma_2_dict",
    "gear_set_dict",
    "on_ground_dict",
    "spd_brake_dict",
]

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
MINUTE: float = 60.0  # seconds
HOUR: float = 3600.0  # seconds

# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------
NM: float = 1852.0  # Nautical mile → metres
FT: float = 0.3048  # Foot → metres

# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------
KT: float = NM / HOUR  # Knot → m/s
FTMIN: float = FT / MINUTE  # ft/min → m/s

# ---------------------------------------------------------------------------
# Mass flow
# ---------------------------------------------------------------------------
KGH: float = 1.0 / HOUR  # kg/h → kg/s

# ---------------------------------------------------------------------------
# ISA atmosphere
# ---------------------------------------------------------------------------
T_K_C: float = 273.15  # 0 °C in Kelvin
T0: float = T_K_C + 15.0  # Sea-level temperature (K)
P0: float = 101_325.0  # Sea-level pressure (Pa)
GAMMA_AIR: float = 1.4  # Ratio of specific heats Cp/Cv
R: float = 287.05  # Specific gas constant for dry air (J/(kg·K))
A0: float = (GAMMA_AIR * R * T0) ** 0.5  # Speed of sound at sea level (m/s)
L: float = 0.0065  # Temperature lapse rate (K/m)
G: float = 9.80665  # Gravitational acceleration (m/s²)
RHO0: float = P0 / (R * T0)  # Sea-level air density (kg/m³)
TROPOPAUSE_ALT_M: float = 11_000.0  # Tropopause altitude (m)

# ---------------------------------------------------------------------------
# Discrete-signal lookup tables (QAR)
# ---------------------------------------------------------------------------

gear_set_dict: dict[str, int] = {
    "UP": 1,
    "DOWN": 0,
}

flap_dict: dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "FULL": 4,
    "NRD": 5,
}

spd_brake_dict: dict[str, int] = {
    "NOT COMMANDED": 0,
    "COMMANDED": 1,
}

on_ground_dict: dict[str, int] = {
    "AIR": 0,
    "GROUND": 1,
}

fma_2_dict: dict[str, int] = {
    "SRS": 0,
    "CLB": 1,
    "OP CLB": 2,
    "EXP CLB": 3,
    "ALT": 4,
    "ALT*": 5,
    "FINAL": 6,
    "DES": 7,
    "OP DES": 8,
    "EXP DES": 9,
    "V/S": 10,
    "FPA": 11,
    "FLARE": 12,
    "G/S": 13,
    "G/S*": 14,
    "ROLL": 15,
    "nan": 16,
}
