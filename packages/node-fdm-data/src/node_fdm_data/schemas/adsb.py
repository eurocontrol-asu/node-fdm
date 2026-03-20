"""Column schemas for the ADS-B v1 architecture.

Simplified longitudinal model:

* **State (X)**: removes ``fdm_distance_cum_m`` (not needed for dynamics).
* **Control (U)**: replaces ``fdm_mcp_alt_sel_m`` (BDS40, ~50% coverage)
  with ``fdm_alt_target_m`` (backward-fill, never NaN).
* **Environment (E0)**: removes airport distances (not relevant for
  flight dynamics).
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
    "fdm_mach_sel",
    "fdm_cas_sel_ms",
    "fdm_vz_sel_ms",
]

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
]

# ---------------------------------------------------------------------------
# Derivatives (dx) — sign and target column
# ---------------------------------------------------------------------------
DX_COLS: list[tuple[int, str]] = [
    (1, "fdm_d_vz_ms"),
    (1, "fdm_d_gamma_rads"),
    (1, "fdm_d_tas_ms"),
]
