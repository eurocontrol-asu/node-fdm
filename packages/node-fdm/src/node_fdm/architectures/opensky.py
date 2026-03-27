"""OpenSky 2025 architecture specification.

Defines the two-layer architecture (trajectory + data_ode) for OpenSky
ADS-B data.  Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register
from node_fdm_data.schemas.opensky import DX_COLS, E0_COLS, E1_COLS, U_COLS, X_COLS

__all__ = [
    "OPENSKY_2025",
]

OPENSKY_2025 = ArchitectureSpec(
    name="opensky_2025",
    x_cols=X_COLS,
    u_cols=U_COLS,
    e0_cols=E0_COLS,
    e1_cols=E1_COLS,
    dx_cols=DX_COLS,
    layers=[
        LayerSpec(
            name="trajectory",
            layer_class="node_fdm.layers.trajectory.TrajectoryLayer",
            input_cols=X_COLS + E0_COLS,
            output_cols=E1_COLS,
            trainable=False,
            config={
                "col_map": {
                    "tas": "era_tas_ms",
                    "gamma": "fdm_gamma_rad",
                    "alt": "raw_alt_m",
                    "wind": "fdm_long_wind_ms",
                    "alt_sel": "fdm_mcp_alt_sel_m",
                    "vz": "fdm_d_alt_ms",
                    "gs": "raw_gs_ms",
                    "mach": "era_mach",
                    "cas": "bds_ias_ms",
                    "alt_diff": "fdm_alt_diff_m",
                },
            },
        ),
        LayerSpec(
            name="data_ode",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=X_COLS + U_COLS + E0_COLS + E1_COLS,
            output_cols=["fdm_d_gamma_rads", "fdm_d_tas_ms"],
            trainable=True,
        ),
    ],
    preprocessing_fn="node_fdm_data.preprocessing.opensky.flight_processing",
    segment_filter_fn=None,
)

register(OPENSKY_2025)
