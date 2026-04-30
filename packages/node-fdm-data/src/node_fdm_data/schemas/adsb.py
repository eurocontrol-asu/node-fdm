"""Column schemas for the ADS-B v1 architecture.

Simplified longitudinal model:

* **State (X)**: removes ``fdm_distance_cum_m`` (not needed for dynamics).
  Adds ``fdm_heading_rad`` (lateral channel state — true heading,
  unsigned-wrapped in ``[0, 2π)`` at the data layer; integrated freely
  by the solver, wrap-aware loss in the trainer).
* **Control (U)**: replaces ``fdm_mcp_alt_sel_m`` (BDS40, ~50% coverage)
  with ``fdm_alt_target_m`` (backward-fill, never NaN).  Uses unified
  ``fdm_tas_target_ms`` instead of separate Mach/CAS selects.  Adds
  ``fdm_heading_target_rad`` (FMS-equivalent heading consigne projected
  through the wind triangle) and the two known-flags
  ``fdm_heading_target_known`` / ``fdm_heading_known``.
* **Control ODE subset (U_ODE)**: empty — all controls (altitude,
  TAS, gamma targets) are consumed by the trajectory layer, not the ODE.
* **Environment (E0)**: removes airport distances (not relevant for
  flight dynamics).  Adds ``era_u_wind_ms`` and ``era_v_wind_ms`` (full
  ERA5 wind vector, consumed by the TrajectoryLayer to recompute
  per-step drift / track / 2D ground speed from the integrated heading).
* **Environment at t (E1)**: extends opensky with ``fdm_tas_diff_ms``
  and ``fdm_gamma_diff_rad`` (error signals for the structured layer).
  Adds the lateral derived signals ``fdm_lat_wind_ms``, ``fdm_drift_rad``,
  ``fdm_track_rad``, ``fdm_heading_diff_rad`` (all computed by the
  TrajectoryLayer from the live heading state).
* **Derivatives (DX)**: removes ``raw_gs_ms`` (derivative of the
  dropped distance state). Adds ``fdm_d_heading_rads``.

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
    "fdm_heading_rad",
]

# ---------------------------------------------------------------------------
# Control inputs (u) — selected / target values
# ---------------------------------------------------------------------------
U_COLS: list[str] = [
    "fdm_alt_target_m",
    "fdm_tas_target_ms",
    "fdm_gamma_target_rad",
    "fdm_gamma_target_known",
    "fdm_tas_target_known",
    "fdm_heading_target_rad",
    "fdm_heading_target_known",
    "fdm_heading_known",
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
    "era_u_wind_ms",
    "era_v_wind_ms",
]

# ---------------------------------------------------------------------------
# Environment at t (e1) — derived by trajectory layer
# ---------------------------------------------------------------------------
E1_COLS: list[str] = [
    "fdm_d_alt_ms",
    "era_mach",
    "raw_gs_ms",
    "fdm_cas_ms",
    "fdm_alt_diff_m",
    "fdm_tas_diff_ms",
    "fdm_gamma_diff_rad",
    "fdm_g_sin_gamma_ms2",
    "fdm_cos_gamma",
    "fdm_q_pa",
    "fdm_g_over_v",
    "fdm_lat_wind_ms",
    "fdm_drift_rad",
    "fdm_track_rad",
    "fdm_heading_diff_rad",
]

# ---------------------------------------------------------------------------
# Derivatives (dx) — sign and target column
# ---------------------------------------------------------------------------
DX_COLS: list[tuple[int, str]] = [
    (1, "fdm_d_alt_ms"),
    (1, "fdm_d_gamma_rads"),
    (1, "fdm_d_tas_ms2"),
    (1, "fdm_d_heading_rads"),
]
