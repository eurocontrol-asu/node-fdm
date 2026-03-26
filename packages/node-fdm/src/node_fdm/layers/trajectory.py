"""Trajectory layer computing derived flight quantities.

Computes vertical speed, Mach number, CAS, ground speed, and altitude
difference from state and environment inputs using ISA atmosphere model.

Column names are configurable via ``col_map`` to support both OpenSky
and QAR naming conventions.

Uses torch-native ISA computations (autodiff-compatible) rather than the
NumPy-based ISA in ``node_fdm_data.physics.isa``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from node_fdm_data.physics.constants import A0, GAMMA_AIR, P0, T0, R

__all__ = [
    "TrajectoryLayer",
]

# ISA constants
_LAPSE_RATE: float = 0.0065  # K/m
_TROPO_ALT: float = 11_000.0  # m
_T_STRATO: float = 216.65  # K
_P_TROPO_11KM: float = 22_632.06  # Pa

# Default column mapping (OpenSky naming — SI units)
DEFAULT_COL_MAP: dict[str, str] = {
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
}


def _isa_temperature_torch(h: torch.Tensor) -> torch.Tensor:
    """ISA temperature (K) from altitude (m), torch-native."""
    h = torch.nan_to_num(h, nan=0.0, posinf=1e5, neginf=0.0)
    h = torch.clamp(h, 0.0, 20_000.0)
    t_tropo = torch.clamp(T0 - _LAPSE_RATE * h, min=150.0, max=320.0)
    t_strato = torch.full_like(h, _T_STRATO)
    return torch.where(h <= _TROPO_ALT, t_tropo, t_strato)


def _isa_pressure_torch(h: torch.Tensor) -> torch.Tensor:
    """ISA pressure (Pa) from altitude (m), torch-native."""
    h = torch.nan_to_num(h, nan=0.0, posinf=1e5, neginf=0.0)
    h = torch.clamp(h, 0.0, 20_000.0)
    t_tropo = T0 - _LAPSE_RATE * h
    p_tropo = P0 * torch.clamp(t_tropo / T0, min=1e-6) ** (9.80665 / (_LAPSE_RATE * R))
    exp_term = torch.clamp(-9.80665 * (h - _TROPO_ALT) / (R * _T_STRATO), min=-100.0, max=0.0)
    p_strato = _P_TROPO_11KM * torch.exp(exp_term)
    return torch.where(h <= _TROPO_ALT, p_tropo, p_strato)


class TrajectoryLayer(nn.Module):
    """Compute trajectory outputs: vertical speed, Mach, CAS, ground speed, and error signals.

    Derived quantities include aerodynamic speeds (Mach, CAS), vertical
    speed, ground speed, and optional error signals (altitude diff, TAS
    diff, gamma diff) when the corresponding selected/target columns are
    present in the input.

    Column names are resolved through ``col_map`` so the same layer works
    with different naming conventions (OpenSky vs. QAR).

    NaN handling for error signals:
        - **gamma_diff**: NaN-aware — positions where ``gamma_target`` is
          NaN produce ``gamma_diff = 0`` (no correction signal), rather
          than propagating NaN or computing ``0 - gamma``.
        - **tas_diff** / **alt_diff**: NaN targets are replaced with 0
          via ``nan_to_num`` before differencing.
    """

    def __init__(
        self,
        col_map: dict[str, str] | None = None,
        input_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize the trajectory layer.

        Args:
            col_map: Mapping from canonical names (``"tas"``, ``"gamma"``,
                ``"alt"``, ``"wind"``, ``"alt_sel"``, ``"vz"``, ``"gs"``,
                ``"mach"``, ``"cas"``, ``"alt_diff"``, ``"tas_sel"``,
                ``"tas_diff"``, ``"gamma_sel"``, ``"gamma_diff"``,
                ``"gamma_known"``) to actual column names in the input
                dict.  Missing keys fall back to ``DEFAULT_COL_MAP``.
            input_stats: Optional column statistics (unused, kept for
                backward-compatible call-sites).
        """
        super().__init__()
        self.col_map = {**DEFAULT_COL_MAP, **(col_map or {})}

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute derived trajectory quantities from input mapping.

        Args:
            x: Mapping from column names to input tensors.

        Returns:
            Dictionary of derived tensors keyed by column name.
        """
        c = self.col_map
        output: dict[str, torch.Tensor] = {}
        _x = {k: v.clone() for k, v in x.items()}

        # Sanitize key inputs
        for col_key in ["tas", "gamma", "alt", "wind"]:
            col_name = c.get(col_key, "")
            if col_name in _x:
                _x[col_name] = torch.nan_to_num(_x[col_name], nan=0.0, posinf=1e6, neginf=-1e6)
                _x[col_name] = torch.clamp(_x[col_name], min=-1e6, max=1e6)

        tas = _x[c["tas"]]
        gamma = _x[c["gamma"]]
        alt = _x[c["alt"]]

        # Vertical speed
        output[c["vz"]] = tas * torch.sin(gamma)

        # Ground speed (TAS minus longitudinal wind)
        wind_col = c.get("wind", "")
        long_wind = _x.get(wind_col, torch.zeros_like(tas)) if wind_col else torch.zeros_like(tas)
        output[c["gs"]] = tas - long_wind

        # ISA temperature and speed of sound
        temp = _isa_temperature_torch(alt)
        a = torch.sqrt(torch.clamp(GAMMA_AIR * R * temp, min=1e-6, max=1e8))

        # Mach number
        mach = tas / torch.clamp(a, min=1e-6, max=1e8)
        output[c["mach"]] = mach

        # CAS from compressible flow
        p = _isa_pressure_torch(alt)
        pt_over_p = torch.pow(
            torch.clamp(1 + (GAMMA_AIR - 1) / 2 * mach**2, min=1e-6, max=1e6),
            GAMMA_AIR / (GAMMA_AIR - 1),
        )
        qc_p0 = (torch.clamp(p, min=1.0) / P0) * (pt_over_p - 1.0)
        qc_p0 = torch.clamp(qc_p0, min=-0.999, max=1e6)

        cas_term = torch.clamp(qc_p0 + 1.0, min=1e-8, max=1e6)
        cas = A0 * torch.sqrt(
            (2.0 / (GAMMA_AIR - 1.0)) * (cas_term ** ((GAMMA_AIR - 1.0) / GAMMA_AIR) - 1.0)
        )
        cas = torch.nan_to_num(cas, nan=0.0, posinf=1e4, neginf=0.0)
        output[c["cas"]] = cas

        # Altitude difference from selected altitude
        alt_sel_col = c.get("alt_sel", "")
        if alt_sel_col and alt_sel_col in x:
            alt_diff = x[alt_sel_col] - alt
            output[c["alt_diff"]] = torch.nan_to_num(alt_diff, nan=0.0)

        # TAS difference from selected TAS target
        tas_sel_col = c.get("tas_sel", "")
        if tas_sel_col and tas_sel_col in x:
            tas_target = torch.nan_to_num(x[tas_sel_col], nan=0.0)
            output[c["tas_diff"]] = tas_target - tas

        # Gamma difference from selected gamma target
        # When known=1: gamma_diff = target - gamma (FMS consigne)
        # When known=0: gamma_diff = 0 (no target, StructuredLayer uses
        #   alt_diff + gamma_known flag to decide d_gamma)
        gamma_sel_col = c.get("gamma_sel", "")
        if gamma_sel_col and gamma_sel_col in x:
            target = x[gamma_sel_col]
            gamma_known_col = c.get("gamma_known", "")
            if gamma_known_col and gamma_known_col in x:
                known = x[gamma_known_col]
                gamma_diff = known * (target - gamma)
            else:
                gamma_diff = target - gamma
            output[c["gamma_diff"]] = gamma_diff

        # Pass through gamma_known flag for the StructuredLayer
        gamma_known_col = c.get("gamma_known", "")
        if gamma_known_col and gamma_known_col in x:
            output[gamma_known_col] = x[gamma_known_col]

        return output
