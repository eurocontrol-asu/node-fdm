import numpy as np
from utils.data.unit import Unit
from utils.physics.constants import NM, ft, kt, ftmn, T_k_c, kgh

from utils.data.conversions import (
    LinearUnitConverter,
    AdditionUnitConverter,
)

# Dimensionless units
dimensionless = Unit("dimensionless", None)
dimensionless_str = Unit("dimensionless", None, value_type="str")

# Force unit
newton = Unit("newton", "N")

# Acceleration units
meter_per_second_square = Unit("meter per second squared", "m/s²")
meter_per_second = Unit(
    "meter per second",
    "m/s",
    derivative=meter_per_second_square,
)

# Speed units
knot = Unit(
    "knot",
    "kt",
    si_unit=meter_per_second,
    modifier=LinearUnitConverter(kt),
)

feet_per_minute = Unit(
    "feet per minute",
    "ft/min",
    si_unit=meter_per_second,
    modifier=LinearUnitConverter(ftmn),
)

# Length units
meter = Unit(
    "meter",
    "m",
    derivative=meter_per_second,
)

feet = Unit(
    "feet",
    "ft",
    si_unit=meter,
    modifier=LinearUnitConverter(ft),
)

nautical_miles = Unit(
    "nautical mile",
    "NM",
    si_unit=meter,
    modifier=LinearUnitConverter(NM),
)

# Mass units
kilogram_per_sec = Unit("kilogram per second", "kg/s")
kilogram_per_hour = Unit(
    "kilogram per hour",
    "kg/h",
    si_unit=kilogram_per_sec,
    modifier=LinearUnitConverter(kgh),
)
kilogram = Unit(
    "kilogram",
    "kg",
    derivative=kilogram_per_sec,
)

# Temperature units
kelvin = Unit("kelvin", "K")
degree_celsius = Unit(
    "degree celsius",
    "°C",
    si_unit=kelvin,
    modifier=AdditionUnitConverter(T_k_c),
)

# Angular units
rad_per_sec = Unit("radian per second", "rad/s")
rad = Unit("radian", "rad", derivative=rad_per_sec)
degree = Unit("degree", "°", si_unit=rad, modifier=np.deg2rad)
