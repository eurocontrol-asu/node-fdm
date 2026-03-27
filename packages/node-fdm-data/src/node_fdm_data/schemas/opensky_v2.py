"""Column schemas for the OpenSky V2 architecture (lateral extension).

Extends the base OpenSky 2025 schema with lateral state variables
(latitude, longitude, track, track_sel) for the longitudinal+lateral
flight dynamics model.

All column names use SI units (m, m/s, rad, K).  Conversions from
raw units (ft, kt, ft/min, °C, NM) are applied in
:func:`~node_fdm_data.preprocessing.opensky.training_preprocessing`.
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
# Extends v1 with latitude, longitude, track_sel (from lateral computations)
# ---------------------------------------------------------------------------
X_COLS: list[str] = [
    "fdm_distance_cum_m",
    "raw_alt_m",
    "fdm_gamma_rad",
    "era_tas_ms",
    "latitude",
    "longitude",
    "track_sel",
]

# ---------------------------------------------------------------------------
# Control inputs (u) — selected / target values
# ---------------------------------------------------------------------------
U_COLS: list[str] = [
    "fdm_mcp_alt_sel_m",
    "fdm_mach_sel",
    "fdm_cas_sel_ms",
    "fdm_vz_sel_ms",
]

# ---------------------------------------------------------------------------
# Environment at t=0 (e0) — fed to trajectory layer
# ---------------------------------------------------------------------------
E0_COLS: list[str] = [
    "fdm_long_wind_ms",
    "fdm_adep_dist_m",
    "fdm_ades_dist_m",
    "era_temp_K",
]

# ---------------------------------------------------------------------------
# Environment at t (e1) — derived by trajectory layer
# ---------------------------------------------------------------------------
E1_COLS: list[str] = [
    "fdm_d_alt_ms",
    "era_mach",
    "raw_gs_ms",
    "bds_ias_ms",
    "fdm_alt_diff_m",
    "track",
]

# ---------------------------------------------------------------------------
# Derivatives (dx) — sign and target column
# ---------------------------------------------------------------------------
DX_COLS: list[tuple[int, str]] = [
    (1, "raw_gs_ms"),
    (1, "fdm_d_alt_ms"),
    (1, "fdm_d_gamma_rads"),
    (1, "fdm_d_tas_ms"),
    (1, "d_track"),
]
