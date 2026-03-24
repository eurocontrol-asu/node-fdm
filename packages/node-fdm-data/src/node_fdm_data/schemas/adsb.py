"""Column schemas for the ADS-B v1 architecture.

Simplified longitudinal model:

* **State (X)**: removes ``fdm_distance_cum_m`` (not needed for dynamics).
* **Control (U)**: replaces ``fdm_mcp_alt_sel_m`` (BDS40, ~50% coverage)
  with ``fdm_alt_target_m`` (backward-fill, never NaN).  Uses unified
  ``fdm_tas_target_ms`` instead of separate Mach/CAS selects.
* **Control ODE subset (U_ODE)**: empty — all controls (altitude,
  TAS, gamma targets) are consumed by the trajectory layer, not the ODE.
* **Environment (E0)**: removes airport distances (not relevant for
  flight dynamics).
* **Environment at t (E1)**: extends opensky with ``fdm_tas_diff_ms``
  and ``fdm_gamma_diff_rad`` (error signals for the structured layer).
* **Derivatives (DX)**: removes ``raw_gs_ms`` (derivative of the
  dropped distance state).

All column names use SI units (m, m/s, rad, K).
"""

from __future__ import annotations

__all__ = [
    "DX_COLS",
    "E0_COLS",
    "E1_COLS",
    "U_COLS",
    "U_ODE_COLS",
    "X_COLS",
]

# ---------------------------------------------------------------------------
# State variables (x) — integrated by the Neural ODE
# ---------------------------------------------------------------------------
X_COLS: list[str] = [
    "raw_alt_m",
    "fdm_gamma_rad",
    "era_tas_ms",
]

# ---------------------------------------------------------------------------
# Control inputs (u) — selected / target values
# ---------------------------------------------------------------------------
U_COLS: list[str] = [
    "fdm_alt_target_m",
    "fdm_tas_target_ms",
    "fdm_gamma_target_rad",
]

# Subset of U_COLS fed to the ODE layer — empty: all controls are
# consumed by the trajectory layer, none feed the ODE directly.
U_ODE_COLS: list[str] = []

# ---------------------------------------------------------------------------
# Environment at t=0 (e0) — fed to trajectory layer
# ---------------------------------------------------------------------------
E0_COLS: list[str] = [
    "fdm_long_wind_ms",
    "era_temp_K",
]

# ---------------------------------------------------------------------------
# Environment at t (e1) — derived by trajectory layer
# ---------------------------------------------------------------------------
E1_COLS: list[str] = [
    "fdm_d_vz_ms",
    "era_mach",
    "raw_gs_ms",
    "bds_ias_ms",
    "fdm_alt_diff_m",
    "fdm_tas_diff_ms",
    "fdm_gamma_diff_rad",
]

# ---------------------------------------------------------------------------
# Derivatives (dx) — sign and target column
# ---------------------------------------------------------------------------
DX_COLS: list[tuple[int, str]] = [
    (1, "fdm_d_vz_ms"),
    (1, "fdm_d_gamma_rads"),
    (1, "fdm_d_tas_ms"),
]
