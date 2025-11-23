from node_fdm.architectures.qar.trajectory_layer import TrajectoryLayer
from utils.learning.base.structured_layer import StructuredLayer

from node_fdm.architectures.qar.columns import (
    col_dist,
    col_alt,
    col_gamma,
    col_tas,
    col_temp,
    col_vz,
    col_mach,
    col_gs,
    col_cas,
    col_alt_diff,
    col_spd_diff,
    col_alt_sel,
    col_spd_sel,
    col_vz_sel,
    col_mass,
    col_spd_brake_commanded,
    col_gear_up,
    col_flap_setting,
    col_head_wind_spd,
    col_cross_wind_spd,
    col_aoa,
    col_pitch,
    col_n1, 
    col_ff, 
)


X_COLS = [
    col_dist,
    col_alt,
    col_gamma,
    col_tas,
    col_mass,
]

U_COLS = [
    col_alt_sel,
    col_spd_sel,
    col_vz_sel,
    col_gear_up,
    col_spd_brake_commanded,
    col_flap_setting,
]

E0_COLS = [
    col_head_wind_spd,
    col_cross_wind_spd,
    col_temp,
]

DX_COLS = [
    (1, col_gs),
    (1, col_vz),
    (1, col_gamma.derivative),
    (1, col_tas.derivative),
    (-1, col_ff)
]


E1_COLS = [
    col_vz,
    col_mach,
    col_gs,
    col_cas,
    col_alt_diff,
    col_spd_diff,
]

E2_COLS = [
    col_aoa,
    col_pitch,
]

E3_COLS = [
    col_n1, 
    col_ff
]

E_COLS = E1_COLS + E2_COLS + E3_COLS


MODEL_COLS = X_COLS, U_COLS, E0_COLS, E_COLS, DX_COLS

TRAJECTORY_LAYER =  [
    "trajectory",
    TrajectoryLayer,
    X_COLS + E0_COLS,
    E1_COLS,
    False
]

ANGLE_LAYER =  [
    "angle",
    StructuredLayer,
    X_COLS + E0_COLS + E1_COLS,
    E2_COLS,
    True
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
    ANGLE_LAYER,
    DATA_ODE_LAYER
]
