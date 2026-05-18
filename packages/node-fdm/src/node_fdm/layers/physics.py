"""Analytical physics layer applying gravity to NN-predicted aerodynamic terms.

The trainable ``StructuredLayer`` predicts purely aerodynamic / propulsive
quantities :

* ``fdm_a_spec_ms2 = (T - D) / m``  — specific longitudinal acceleration
* ``fdm_n_z        = L / (m g)``    — vertical load factor

This non-trainable layer applies the gravity terms analytically to recover
the ODE state derivatives consumed by ``ArchitectureSpec.dx_cols`` :

    d_tas   = a_spec - g * sin(gamma)
    d_gamma = (g / max(tas, V_MIN)) * (n_z - cos(gamma))

Pulling gravity out of the network removes a known physics term from the
learning target, so the NN concentrates on what actually depends on the
aircraft (thrust, drag polar, lift) rather than re-discovering ``g``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "V_MIN_CLAMP",
    "G",
    "PhysicsLayer",
]

# Standard gravity used by both longitudinal and lateral dynamics equations.
G: float = 9.80665
"""Standard gravity (m/s²)."""

# Minimum true airspeed used to keep angular-rate divisions numerically stable.
V_MIN_CLAMP: float = 50.0
"""Lower bound on TAS for the ``1/V`` term (m/s).

Sized to the typical aircraft stall speed: a transport aircraft never flies
below ~55 m/s in any phase. The clamp protects ``d_gamma = (g/V)·(...)``
from blowing up when an intermediate ODE substep produces a corrupted TAS
(negative or near-zero) outside the physical envelope. Without this bound
``g/V`` reaches ``g/V_MIN_CLAMP_OLD ≈ 3.5`` which immediately exits
``dx_bounds`` and triggers a NaN cascade through the integrator.
"""


class PhysicsLayer(nn.Module):
    """Apply analytic flight dynamics in legacy or Newton force mode."""

    def __init__(
        self,
        input_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize the physics layer.

        Args:
            input_stats: Optional column statistics; unused, accepted only to
                keep the non-trainable layer instantiation contract uniform
                with ``TrajectoryLayer``.
        """
        super().__init__()
        del input_stats  # unused

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert model outputs into ODE derivatives.

        Newton mode is selected when ``fdm_t_minus_d_N`` is present. It
        consumes thrust-minus-drag, lift, mass, true airspeed and flight path
        angle, then emits translational, angular and mass derivatives.

        Legacy mode consumes ``fdm_a_spec_ms2`` and ``fdm_n_z_residual``. The
        NN emits ``fdm_n_z_residual = n_z - 1`` (a quantity centered on zero)
        rather than ``n_z`` itself, so symmetric denormalize modes
        (``"scaled"``, ``"normal_clamp"``) behave correctly. This layer adds
        ``1`` back before applying the dynamics equation.

        When ``fdm_phi_bank_rad`` is present in the input mapping (lateral
        channel), this layer also emits ``fdm_d_heading_rads`` from the
        coordinated-turn equation ``d_heading = (g/V)·tan(phi_bank)``.
        ``phi_bank`` is hard-capped upstream (StructuredLayer cap=1.0 rad);
        the resulting rate can exceed ±0.1 rad/s and is bounded again by the
        projected integrator's ``dx_bounds``.

        Args:
            x: Mapping containing either Newton-mode force columns or legacy
                acceleration/load-factor columns, plus ``era_tas_ms``,
                ``fdm_gamma_rad`` and optional ``fdm_phi_bank_rad``.

        Returns:
            Dictionary with ODE derivatives for the selected mode and, when
            phi_bank is provided, ``fdm_d_heading_rads``.
        """
        tas = x["era_tas_ms"]
        gamma = x["fdm_gamma_rad"]
        tas_safe = torch.clamp(tas, min=V_MIN_CLAMP)

        if "fdm_t_minus_d_N" in x:
            t_minus_d = x["fdm_t_minus_d_N"]
            lift = x["fdm_lift_N"]
            mass = x["fdm_mass_kg"]
            out: dict[str, torch.Tensor] = {
                "fdm_d_tas_ms2": t_minus_d / mass - G * torch.sin(gamma),
                "fdm_d_gamma_rads": (lift / mass - G * torch.cos(gamma)) / tas_safe,
                "fdm_d_mass_kgs": torch.zeros_like(mass),
            }
        else:
            a_spec = x["fdm_a_spec_ms2"]
            n_z = x["fdm_n_z_residual"] + 1.0
            out = {
                "fdm_d_tas_ms2": a_spec - G * torch.sin(gamma),
                "fdm_d_gamma_rads": (G / tas_safe) * (n_z - torch.cos(gamma)),
                "fdm_n_z": n_z,
            }

        if "fdm_phi_bank_rad" in x:
            phi_bank = x["fdm_phi_bank_rad"]
            d_heading = (G / tas_safe) * torch.tan(phi_bank)
            out["fdm_d_heading_rads"] = d_heading

        return out
