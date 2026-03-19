"""OpenSky V2 architecture specification (longitudinal + lateral).

Extends the base OpenSky 2025 architecture with lateral state variables
(latitude, longitude, track_sel) and an extended trajectory layer that
also outputs track bearing.

Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, register
from node_fdm_data.schemas.opensky_v2 import DX_COLS, E0_COLS, E1_COLS, U_COLS, X_COLS

__all__ = [
    "OPENSKY_V2",
]

OPENSKY_V2 = ArchitectureSpec(
    name="opensky_v2",
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
                    "tas": "tas_ms",
                    "gamma": "gamma_rad",
                    "alt": "altitude_m",
                    "wind": "long_wind_ms",
                    "alt_sel": "alt_sel_m",
                    "vz": "vz_ms",
                    "gs": "gs_ms",
                    "mach": "mach",
                    "cas": "cas_ms",
                    "alt_diff": "alt_diff_m",
                    "track": "track",
                    "track_sel": "track_sel",
                },
            },
        ),
        LayerSpec(
            name="data_ode",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=X_COLS + U_COLS + E0_COLS + E1_COLS,
            output_cols=["d_gamma_rads", "d_tas_ms", "d_track"],
            trainable=True,
        ),
    ],
    preprocessing_fn="node_fdm_data.preprocessing.opensky.flight_processing",
    segment_filter_fn=None,
)

register(OPENSKY_V2)
