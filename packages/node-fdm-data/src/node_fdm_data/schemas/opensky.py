"""Column schemas for the OpenSky 2025 architecture.

All column names use SI units (m, m/s, rad, K).  Conversions from
raw units (ft, kt, ft/min, °C, NM) are applied by the ``fdm convert``
pipeline step (:func:`~node_fdm_data.preprocessing.convert.convert_si`).
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
    "fdm_distance_cum_m",
    "raw_alt_m",
    "fdm_gamma_rad",
    "era_tas_ms",
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
]

# ---------------------------------------------------------------------------
# Derivatives (dx) — sign and target column
# ---------------------------------------------------------------------------
DX_COLS: list[tuple[int, str]] = [
    (1, "raw_gs_ms"),
    (1, "fdm_d_alt_ms"),
    (1, "fdm_d_gamma_rads"),
    (1, "fdm_d_tas_ms"),
]
