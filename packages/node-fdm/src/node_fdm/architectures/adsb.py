"""ADS-B v1 architecture specification.

Simplified two-layer architecture (trajectory + data_ode) for ADS-B
data.  Compared to ``opensky_2025``:

* Smaller state vector (no cumulative distance).
* Robust altitude control (``fdm_alt_target_m``, never NaN) kept in
  ``U_COLS`` for the ``TrajectoryLayer``; the ``StructuredLayer`` receives
  only ``U_ODE_COLS`` (``U_COLS`` minus ``fdm_alt_target_m``).
* Leaner environment (no airport distances).
* Fewer derivatives (no ground-speed derivative).

Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register
from node_fdm_data.schemas.adsb import DX_COLS, E0_COLS, E1_COLS, U_COLS, U_ODE_COLS, X_COLS

__all__ = [
    "NODE_ADSB_V1",
]

NODE_ADSB_V1 = ArchitectureSpec(
    name="node_adsb_v1",
    x_cols=X_COLS,
    u_cols=U_COLS,
    e0_cols=E0_COLS,
    e1_cols=E1_COLS,
    dx_cols=DX_COLS,
    layers=[
        LayerSpec(
            name="trajectory",
            layer_class="node_fdm.layers.trajectory.TrajectoryLayer",
            input_cols=X_COLS + U_COLS + E0_COLS,
            output_cols=E1_COLS,
            trainable=False,
            config={
                "col_map": {
                    "tas": "era_tas_ms",
                    "gamma": "fdm_gamma_rad",
                    "alt": "raw_alt_m",
                    "wind": "fdm_long_wind_ms",
                    "alt_sel": "fdm_alt_target_m",
                    "vz": "fdm_d_vz_ms",
                    "gs": "raw_gs_ms",
                    "mach": "era_mach",
                    "cas": "fdm_cas_ms",
                    "alt_diff": "fdm_alt_diff_m",
                    "tas_sel": "fdm_tas_target_ms",
                    "tas_diff": "fdm_tas_diff_ms",
                    "gamma_sel": "fdm_gamma_target_rad",
                    "gamma_known": "fdm_gamma_target_known",
                    "gamma_diff": "fdm_gamma_diff_rad",
                },
            },
        ),
        LayerSpec(
            name="data_ode",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=X_COLS + U_ODE_COLS + E0_COLS + E1_COLS,
            output_cols=["fdm_d_gamma_rads", "fdm_d_tas_ms"],
            trainable=True,
        ),
    ],
    preprocessing_fn="node_fdm_data.preprocessing.opensky.flight_processing",
    segment_filter_fn=None,
    x_bounds={
        "raw_alt_m": (0.0, 15000.0),
        "fdm_gamma_rad": (-0.3, 0.3),
        "era_tas_ms": (50.0, 350.0),
    },
    dx_bounds={
        "fdm_d_vz_ms": (-10.0, 10.0),
        "fdm_d_gamma_rads": (-0.01, 0.01),
        "fdm_d_tas_ms": (-5.0, 5.0),
    },
)

register(NODE_ADSB_V1)
