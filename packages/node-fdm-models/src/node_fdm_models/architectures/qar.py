"""QAR architecture specification.

Defines the four-layer architecture (trajectory + angle + engine + data_ode)
for QAR flight recorder data.  Auto-registers at import time.
"""

from __future__ import annotations

from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec
from node_fdm_models.schemas.qar import DX_COLS, E0_COLS, E1_COLS, U_COLS, X_COLS

__all__ = [
    "QAR",
]

# Split E1_COLS into semantic sub-groups for layer input/output mapping
# E1_COLS from schema: vv, mach, gnd, cas, gw, n1_l, n1_r, pitch, aoa
_TRAJ_OUTPUTS = ["ATT__VV", "SPD__MACH", "SPD__GND", "SPD__CAS"]
_ANGLE_OUTPUTS = ["ATT__PITCH", "ATT__AOA_LH"]
_ENGINE_OUTPUTS = ["ENG__N1_LEFT", "ENG__N1_RIGHT"]
_MASS_COLS = ["SYS__GW"]

QAR = ArchitectureSpec(
    name="qar",
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
            output_cols=_TRAJ_OUTPUTS,
            trainable=False,
            config={
                "col_map": {
                    "tas": "SPD__TAS",
                    "gamma": "ATT__FPA",
                    "alt": "ALT__STD",
                    "wind": "WIND__HEAD_WIND",
                    "alt_sel": "NAV__ALT_SEL",
                    "vz": "ATT__VV",
                    "gs": "SPD__GND",
                    "mach": "SPD__MACH",
                    "cas": "SPD__CAS",
                    "alt_diff": "alt_diff",
                },
            },
        ),
        LayerSpec(
            name="angle",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=X_COLS + E0_COLS + _TRAJ_OUTPUTS,
            output_cols=_ANGLE_OUTPUTS,
            trainable=True,
        ),
        LayerSpec(
            name="engine",
            layer_class="node_fdm.layers.engine.EngineLayer",
            input_cols=X_COLS + E0_COLS + _TRAJ_OUTPUTS,
            output_cols=_ENGINE_OUTPUTS,
            trainable=True,
        ),
        LayerSpec(
            name="data_ode",
            layer_class="node_fdm.layers.structured.StructuredLayer",
            input_cols=(
                X_COLS + U_COLS + E0_COLS + _TRAJ_OUTPUTS + _ANGLE_OUTPUTS + _ENGINE_OUTPUTS
                # TODO: add _MASS_COLS once mass-decay layer is implemented
            ),
            output_cols=["d_ATT__FPA", "d_SPD__TAS"],
            trainable=True,
        ),
    ],
    preprocessing_fn="node_fdm_models.preprocessing.qar.flight_processing",
)
