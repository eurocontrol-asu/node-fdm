from node_fdm.architectures.opensky_2025.trajectory_layer import TrajectoryLayer
from utils.learning.base.structured_layer import StructuredLayer

from node_fdm.architectures.opensky_2025.columns import (
    col_dist,
    col_alt,
    col_gamma,
    col_tas,
    col_long_wind_spd,
    col_adep_dist,
    col_ades_dist,
    col_temp,
    col_vz,
    col_mach,
    col_gs,
    col_cas,
    col_alt_diff,
    col_alt_sel,
    col_mach_sel,
    col_cas_sel,
    col_vz_sel,
)


X_COLS = [
    col_dist,
    col_alt,
    col_gamma,
    col_tas,
]

U_COLS = [
    col_alt_sel,
    col_mach_sel,
    col_cas_sel,
    col_vz_sel,
]

E0_COLS = [
    col_long_wind_spd,
    col_adep_dist,
    col_ades_dist,
    col_temp,
]

DX_COLS = [col.derivative for col in X_COLS]


E1_COLS = [
    col_vz,
    col_mach,
    col_gs,
    col_cas,
    col_alt_diff,
]

MODEL_COLS = X_COLS, U_COLS, E0_COLS, E1_COLS, DX_COLS

TRAJECTORY_LAYER =  [
    "trajectory",
    TrajectoryLayer,
    X_COLS + E0_COLS,
    E1_COLS,
    False
    ]


DATA_ODE_LAYER = [
    "data_ode",
    StructuredLayer,
    X_COLS + U_COLS + E0_COLS + E1_COLS,
    [col_tas.derivative, col_gamma.derivative],
    True
]

ARCHITECTURE = [
    TRAJECTORY_LAYER,
    DATA_ODE_LAYER
]
