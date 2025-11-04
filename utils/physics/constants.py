"""
Physical and environmental constants for aviation calculations.

Units are mostly SI, with aviation-specific conversions.
"""

# Time constants
minute = 60          # seconds
hour = 3600          # seconds

# Distance constants
NM = 1852            # Nautical Mile in meters
ft = 0.3048          # Foot in meters

# Speed constants
kt = NM / hour       # Knot in meters per second
ftmn = ft / minute   # Feet per minute in meters per second

# Mass flow rate conversion
kgh = 1 / hour       # kilograms per second (1/hour)

# Temperature and atmospheric conditions
T_k_c = 273.15       # Zero Celsius in Kelvin
T0 = T_k_c + 15      # Standard temperature at sea level in Kelvin (15°C)
p0 = 101325.0        # Standard pressure at sea level in Pascals
gamma_ratio = 1.4    # Ratio of specific heats (Cp/Cv) for air
R = 287.05           # Specific gas constant for dry air in J/(kg·K)
a0 = (gamma_ratio * R * T0) ** 0.5         # Speed of sound at sea level in meters per second
L = 0.0065           # Temperature lapse rate in K/m
g = 9.80665          # Gravitational acceleration in m/s²
tropopause_alt_m = 11000.0  # Tropopause altitude in meters
lapse_rate = 0.0065          # Temperature lapse rate in K/m (duplicate of L)

T0 = 288.15
