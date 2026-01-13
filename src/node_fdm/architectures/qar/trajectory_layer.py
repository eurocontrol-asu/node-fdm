#!/usr/bin/env python3
"""Torch module for computing basic trajectory-related aerodynamic features."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from node_fdm.architectures.qar.columns import (
    col_alt,
    col_alt_diff,
    col_alt_sel,
    col_cas,
    col_gamma,
    col_gs,
    col_head_wind_spd,
    col_mach,
    col_spd_diff,
    col_spd_sel,
    col_tas,
    col_vz,
)
from node_fdm.utils.physics.constants import R, a0, gamma_ratio, p0
from node_fdm.utils.physics.torch import isa_pressure_torch, isa_temperature_torch


class TrajectoryLayer(nn.Module):
    """Compute trajectory outputs: vertical speed, Mach, and calibrated airspeed."""

    def __init__(self) -> None:
        """Initialize the trajectory layer with base configuration."""
        super().__init__()
        self.alpha: int = 40

    def forward(self, x: Mapping[Any, torch.Tensor]) -> dict[Any, torch.Tensor]:
        """Compute derived trajectory quantities from the provided inputs.

        Args:
            x: Mapping from column identifiers to input tensors.

        Returns:
            Dictionary of derived tensors keyed by their column identifiers.
        """
        output_dict: dict[Any, torch.Tensor] = {}
        _x = {k: v.clone() for k, v in x.items()}
        for col in _x.keys():
            _x[col] = torch.nan_to_num(_x[col], nan=0.0, posinf=1e8, neginf=-1e8)
            _x[col] = torch.clamp(_x[col], min=-1e8, max=1e8)

        tas = _x[col_tas]
        gamma = _x[col_gamma]
        head_wind = _x[col_head_wind_spd]
        alt = _x[col_alt]

        output_dict[col_vz] = tas * torch.sin(gamma)

        temp = isa_temperature_torch(alt)

        a = torch.sqrt(torch.clamp(gamma_ratio * R * temp, min=1e-6, max=1e6))

        mach = tas / torch.clamp(a, min=1e-6, max=1e8)
        output_dict[col_mach] = mach

        output_dict[col_gs] = tas - head_wind  # To correct with correct formula

        p = isa_pressure_torch(alt)

        pt_over_p = torch.pow(
            torch.clamp(1 + (gamma_ratio - 1) / 2 * mach**2, min=1e-6, max=1e6),
            gamma_ratio / (gamma_ratio - 1),
        )

        qc_p0 = (torch.clamp(p, min=1.0) / p0) * (pt_over_p - 1.0)
        qc_p0 = torch.clamp(qc_p0, min=-0.999, max=1e6)

        cas_term = torch.clamp(qc_p0 + 1.0, min=1e-8, max=1e6)
        cas = a0 * torch.sqrt(
            (2.0 / (gamma_ratio - 1.0))
            * (cas_term ** ((gamma_ratio - 1.0) / gamma_ratio) - 1.0)
        )
        cas = torch.nan_to_num(cas, nan=0.0, posinf=1e4, neginf=0.0)
        output_dict[col_cas] = cas

        alt_diff = x[col_alt_sel] - alt
        output_dict[col_alt_diff] = torch.nan_to_num(alt_diff, nan=0.0)

        spd_diff = x[col_spd_sel] - cas
        output_dict[col_spd_diff] = torch.nan_to_num(spd_diff, nan=0.0)

        return output_dict
