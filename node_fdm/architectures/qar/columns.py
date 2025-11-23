from utils.data.column import Column
from torch import nn

from utils.physics.units import (
    nautical_miles,
    knot,
    feet,
    dimensionless,
    degree,
    kilogram,
    kilogram_per_hour,
    kilogram_per_sec,
    feet_per_minute,
    rpm_percent,
    degree_celsius,
    boolean,
    flap_conf,
    gear_up,
    spd_brake_commanded,
    on_ground,
    fma_col_2,
)

# Loaded columns: columns directly loaded from raw data with units
col_dist = Column("distance", "dist", "NAV__GND_DIST", nautical_miles)
col_dist_to_thr = Column("distance_to_thr", "dist_to_thr", "TRAJ__DIST_TO_THR", nautical_miles)

col_long = Column("longitude", "long", "TRAJ__LON_C", degree)
col_lat = Column("latitude", "lat", "TRAJ__LAT_C", degree)
col_alt = Column("altitude standard", "alt_std", "ALT__STD", feet)
col_tas = Column("true air speed", "tas", "SPD__TAS", knot)
col_cas = Column("calibrated air speed", "cas", "SPD__CAS", knot)
col_gs = Column("ground speed", "gs", "SPD__GND", knot)
col_mach = Column("mach number", "mach", "SPD__MACH", dimensionless, normalize_mode=None)
col_vz = Column("vertical speed", "vz", "ATT__VV", feet_per_minute)
col_gamma = Column("flight path angle", "gamma", "ATT__FPA", degree)
col_pitch = Column("pitch angle", "pitch", "ATT__PITCH", degree)
col_aoa = Column("angle of attack", "aoa", "ATT__AOA_LH", degree)

col_mass = Column("mass", "mass", "SYS__GW", kilogram)
col_ff_left = Column("fuel flow left", "ff_left", "FUEL__FF_LEFT", kilogram_per_hour)
col_ff_right = Column("fuel flow right", "ff_right", "FUEL__FF_RIGHT", kilogram_per_hour)
col_fuel_conso = Column("fuel consumption", "ff_conso", "FUEL__CONS", kilogram)
col_n1_left = Column("n1 left", "n1_left_rpm", "ENG__N1_LEFT", rpm_percent)
col_n1_right = Column("n1 right", "n1_right_rpm", "ENG__N1_RIGHT", rpm_percent)

col_alt_sel = Column("selected altitude", "alt_sel", "NAV__ALT_SEL", feet)
col_vz_sel = Column("selected vertical speed", "vz_sel", "SPD__VERT_SEL", feet_per_minute)
col_spd_sel = Column("selected speed", "spd_sel", "SPD__SPD_SEL", knot)

col_temp = Column("temperature", "temp", "TEMP__SAT", degree_celsius)
col_head_wind_spd = Column("head wind speed", "head_wind_spd", "WIND__HEAD_WIND", knot)
col_cross_wind_spd = Column("cross wind speed", "cross_wind_spd", "WIND__CROSS_WIND", knot)
col_runway_elev = Column("runway elevation", "rwy_elev", "TRAJ__RWY_ELEV", feet)

col_flap_setting = Column(
    "control flap lever", 
    "ctrl_flap_lever", 
    "CTL__FLAP_LEVER", 
    flap_conf,
    normalize_mode=None, 
    denormalize_mode = "max",
    last_activation_fn = nn.Sigmoid,
    loss_name = "focal_mse",
)

col_gear_up = Column(
    "gear up", 
    "gear_up", 
    "GEAR__SEL", 
    gear_up, 
    normalize_mode=None, 
    denormalize_mode = None, 
    last_activation_fn = nn.Sigmoid,
    loss_name="focal"
)

col_spd_brake_commanded = Column(
    "speed brakes commanded", 
    "spd_brake_on", 
    "CTL__SPD_BRAKE", 
    spd_brake_commanded, 
    normalize_mode=None, 
    denormalize_mode = None, 
    last_activation_fn = nn.Sigmoid,
    loss_name="focal"
)

col_on_ground = Column("on ground", "on_grnd", "GEAR__WOW_MAIN", on_ground)
col_fma_2 = Column("FMA column 2", "fma_col_2", "FMA_COL_2", fma_col_2, loss_name="cross_entropy")


# Processed columns: created or computed during processing, may not have raw names
col_ff = Column("fuel flow", "ff", None, kilogram_per_sec, denormalize_mode = "max", last_activation_fn = nn.Sigmoid)
col_n1 = Column("n1", "n1_rpm", None, rpm_percent, denormalize_mode = None, last_activation_fn = nn.Sigmoid)
col_reduce_engine = Column("reduce engine", "reduce_eng", None, boolean, normalize_mode=None)

col_alt_diff = Column("altitude difference from selected", "alt_diff_sel", None, feet)
col_spd_diff = Column("speed difference from selected", "spd_diff_sel", None, knot)