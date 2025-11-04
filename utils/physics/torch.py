import torch
from utils.physics.constants import R, p0, T0

def isa_pressure_torch(h):
    """ISA static pressure (Pa) for altitude h (m), torch version (safe)."""
    # Clamp altitude to avoid negative or extreme values
    h = torch.nan_to_num(h, nan=0.0, posinf=1e5, neginf=0.0)
    h = torch.clamp(h, 0.0, 20000.0)

    T_tropo = T0 - 0.0065 * h
    p_tropo = p0 * torch.clamp(T_tropo / T0, min=1e-6) ** (9.80665 / (0.0065 * R))

    T_strato = 216.65
    exp_term = -9.80665 * (h - 11000) / (R * T_strato)
    exp_term = torch.clamp(exp_term, min=-100.0, max=0.0)
    p_strato = 22632.06 * torch.exp(exp_term)

    return torch.where(h <= 11000, p_tropo, p_strato)

def isa_temperature_torch(h):
    """ISA temperature (K) for altitude h (m), torch version (safe)."""
    h = torch.nan_to_num(h, nan=0.0, posinf=1e5, neginf=0.0)
    h = torch.clamp(h, 0.0, 20000.0)

    T0_local = 288.15
    lapse = 0.0065

    T_tropo = T0_local - lapse * h
    T_tropo = torch.clamp(T_tropo, min=150.0, max=320.0)

    T_strato = torch.full_like(h, 216.65)

    return torch.where(h <= 11000, T_tropo, T_strato)