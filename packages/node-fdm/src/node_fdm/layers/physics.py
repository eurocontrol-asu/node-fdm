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

G: float = 9.80665
"""Standard gravity (m/s²)."""

V_MIN_CLAMP: float = 10.0 / 3.6
"""Lower bound on TAS for the ``1/V`` term (~2.78 m/s).

Far below any realistic flight speed; only triggers on corrupted inputs and
prevents a division-by-zero from poisoning the gradient.
"""


class PhysicsLayer(nn.Module):
    """Apply gravity analytically to ``(a_spec, n_z)`` and emit ``(d_tas, d_gamma)``."""

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
        """Convert aerodynamic outputs into ODE derivatives.

        The NN emits ``fdm_n_z_residual = n_z - 1`` (a quantity centered on
        zero) rather than ``n_z`` itself, so symmetric denormalize modes
        (``"scaled"``, ``"normal_clamp"``) behave correctly.  This layer
        adds ``1`` back before applying the dynamics equation.

        Args:
            x: Mapping containing ``fdm_a_spec_ms2``, ``fdm_n_z_residual``,
                ``era_tas_ms``, and ``fdm_gamma_rad``.

        Returns:
            Dictionary with ``fdm_d_tas_ms2`` and ``fdm_d_gamma_rads``.
        """
        a_spec = x["fdm_a_spec_ms2"]
        n_z = x["fdm_n_z_residual"] + 1.0
        tas = x["era_tas_ms"]
        gamma = x["fdm_gamma_rad"]

        tas_safe = torch.clamp(tas, min=V_MIN_CLAMP)
        d_tas = a_spec - G * torch.sin(gamma)
        d_gamma = (G / tas_safe) * (n_z - torch.cos(gamma))

        return {
            "fdm_d_tas_ms2": d_tas,
            "fdm_d_gamma_rads": d_gamma,
            "fdm_n_z": n_z,
        }
