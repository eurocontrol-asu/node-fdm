#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch module for computing basic trajectory features for OpenSky 2025 data."""

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
from node_fdm.architectures.opensky_2025.columns import (
    col_tas,
    col_cas,
    col_gs,
    col_gamma,
    col_alt,
    col_long_wind_spd,
    col_vz,
    col_mach,
    col_alt_sel,
    col_alt_diff,
)

from node_fdm.utils.physics.constants import gamma_ratio, R, p0, a0
from node_fdm.utils.physics.torch import isa_temperature_torch, isa_pressure_torch


class TrajectoryLayer(nn.Module):
    """Compute trajectory outputs such as vertical speed, Mach, and calibrated airspeed."""

    def __init__(self) -> None:
        """Initialize the trajectory layer with base configuration."""
        super().__init__()
        self.alpha = 40

    def forward(self, x: Mapping[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
        """Compute derived trajectory quantities from the provided inputs.

        Args:
            x: Mapping from column identifiers to input tensors.

        Returns:
            Dictionary of derived tensors keyed by their column identifiers.
        """
        output_dict = {}
        for col in [col_tas, col_gamma, col_alt, col_long_wind_spd]:
            x[col] = torch.nan_to_num(x[col], nan=0.0, posinf=1e6, neginf=-1e6)
            x[col] = torch.clamp(x[col], min=-1e6, max=1e6)

        tas = x[col_tas]
        gamma = x[col_gamma]
        long_wind = x[col_long_wind_spd]
        alt = x[col_alt]

        output_dict[col_vz] = tas * torch.sin(gamma)

        temp = isa_temperature_torch(alt)

        a = torch.sqrt(torch.clamp(gamma_ratio * R * temp, min=1e-6, max=1e8))

        mach = tas / torch.clamp(a, min=1e-6, max=1e8)
        output_dict[col_mach] = mach

        output_dict[col_gs] = tas - long_wind

        p = isa_pressure_torch(alt)

        pt_over_p = torch.pow(
            torch.clamp(1 + (gamma_ratio - 1) / 2 * mach**2, min=1e-6, max=1e6),
            gamma_ratio / (gamma_ratio - 1),
        )

        qc_p0 = (torch.clamp(p, min=1.0) / p0) * (pt_over_p - 1.0)
        qc_p0 = torch.clamp(qc_p0, min=-0.999, max=1e6)

        CAS_term = torch.clamp(qc_p0 + 1.0, min=1e-8, max=1e6)
        CAS = a0 * torch.sqrt(
            (2.0 / (gamma_ratio - 1.0))
            * (CAS_term ** ((gamma_ratio - 1.0) / gamma_ratio) - 1.0)
        )
        CAS = torch.nan_to_num(CAS, nan=0.0, posinf=1e4, neginf=0.0)
        output_dict[col_cas] = CAS

        # --- Reference differences ---
        ref_alt = x[col_alt_sel]

        alt_diff = ref_alt - alt

        output_dict[col_alt_diff] = torch.nan_to_num(alt_diff, nan=0.0)

        return output_dict
