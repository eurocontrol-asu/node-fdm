
import numpy as np
from utils.physics.constants import gamma_ratio, a0, p0, R, T0

import numpy as np


def ms_to_kt(el):
    return el * 3600 / 1852


def get_phase(hp_init, hp_target):
    if np.abs(hp_init - hp_target)<50:
        return "Cruise"
    else:
        if hp_init < hp_target:
            return "Climb"
        else:
            return "Descent"
        

def cas_to_mach(CAS, h):
    """
    CAS (m/s) -> Mach number at altitude h (m) [ISA atmosphere].
    Vectorized.
    """
    CAS = np.asarray(CAS)
    h = np.asarray(h)

    # Step 1: CAS -> impact pressure ratio at sea level
    qc_p0 = (1 + (gamma_ratio-1)/2 * (CAS/a0)**2)**(gamma_ratio/(gamma_ratio-1)) - 1

    # Step 2: Static pressure at altitude
    p = isa_pressure(h)

    # Step 3: Impact pressure ratio at altitude
    qc_p = qc_p0 * (p0/p)

    # Step 4: Solve Mach
    M = np.sqrt((2/(gamma_ratio-1)) * (((qc_p + 1)**((gamma_ratio-1)/gamma_ratio)) - 1))
    return M



def tas_to_cas(TAS, h, T):
    """
    TAS (m/s), altitude h (m), temp T (K) -> CAS (m/s)
    """
    # Mach from TAS & local T
    a = np.sqrt(gamma_ratio * R * T)
    M = TAS / a

    # Static pressure at altitude
    p = isa_pressure(h)

    # Total pressure / static pressure
    pt_over_p = (1 + (gamma_ratio-1)/2 * M**2)**(gamma_ratio/(gamma_ratio-1))

    # Impact pressure ratio relative to sea level
    qc_p0 = (p/p0) * (pt_over_p - 1)

    # Convert to CAS at sea level ISA
    CAS = a0 * np.sqrt((2/(gamma_ratio-1)) * (((qc_p0 + 1)**((gamma_ratio-1)/gamma_ratio)) - 1))
    return CAS


def isa_temperature(altitude_m):
    """
    Convert altitude in meters to temperature in degrees Celsius in the ISA atmosphere.
    Altitudes beyond 20 km are not considered in this simplified model.
    
    Parameters:
    altitude_m (float): Altitude in meters
    
    Returns:
    float: Temperature in degrees Celsius at the given altitude
    """
    # Constants
    T0 = 15.0  # Sea level standard temperature in Celsius
    altitude_km = altitude_m / 1000.0  # Convert meters to kilometers
    lapse_rate = -6.5  # Temperature lapse rate in Celsius per kilometer up to 11km
    
    # Troposphere (up to 11 km)
    if altitude_km <= 11:
        temperature = T0 + lapse_rate * altitude_km
    # Lower Stratosphere (11 km to 20 km), constant temperature
    elif 11 < altitude_km <= 20:
        temperature = T0 + lapse_rate * 11
    else:
        # Altitudes above 20 km are not handled by this simplified model.
        raise ValueError("Altitude out of range. This model only supports altitudes up to 20 km.")
    
    return temperature + 273.15

def isa_pressure(h):
    """Pression ISA (Pa) pour altitude h (m), vectorisée."""
    h = np.asarray(h)
    T_tropo = T0 - 0.0065*h
    p_tropo = p0 * (T_tropo/T0)**(9.80665/(0.0065*R))
    T_strato = 216.65
    p_strato = 22632.06 * np.exp(-9.80665*(h-11000)/(R*T_strato))
    return np.where(h <= 11000, p_tropo, p_strato)

