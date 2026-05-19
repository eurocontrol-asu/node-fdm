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

import numpy as np
import torch
import torch.nn as nn

from node_fdm_data.schemas.adsb_hybrid import A320_MTOW_KG, A320_OEW_KG

__all__ = [
    "CL_MAX",
    "CL_REF",
    "S_REF_A320_M2",
    "V_MIN_CLAMP",
    "G",
    "PhysicsLayer",
    "cl_ref_steady",
    "cl_ref_steady_np",
]

# Standard gravity used by both longitudinal and lateral dynamics equations.
G: float = 9.80665
"""Standard gravity (m/s²)."""

# Reference mass used as the ``m_ref`` constant in the Newton-mode lift
# reconstruction ``L = L_res_norm · m_ref·g + m·g``. Picked as the midpoint
# of the A320 TCDS plausible range so the residual stays close to zero on
# average — the MassEncoder absorbs the per-flight deviation.
_M_REF_KG: float = 0.5 * (A320_OEW_KG + A320_MTOW_KG)

# A320 wing reference area (m², public TCDS).
S_REF_A320_M2: float = 122.6
# Cruise-typical lift coefficient. Centers the CL-mode residual on the
# [0.3, 0.7] band in cruise — used as the asymptotic reference value at
# high dynamic pressure, and as a diagnostic anchor in reports.
#
# **Note (post-AXM-1739 refactor)**: at runtime, the CL-mode PhysicsLayer
# does NOT use this constant directly. It calls ``cl_ref_steady(q)`` which
# returns ``m_ref · g / (q · S_REF)`` — the CL value that exactly balances
# weight at the current dynamic pressure. With CL_REF constant, the NN had
# to learn a phase-dependent shift (≈ ±0.2 at low altitude, ≈ 0 at cruise)
# on top of the physics fine structure, starving the gradient on the real
# signal. With the q-dependent steady-state baseline, the NN learns a
# small correction around 0 in every phase — and the CL formulation
# becomes algebraically equivalent to Newton-mode (``L = m_ref·g + q·S·cl_residual``).
# See ``data/models/full_hybrid_v3/cl_distribution.md`` §6 for the
# empirical evidence.
CL_REF: float = 0.5
# A320 stall coefficient at 1 g, clean configuration, sea level. Used by
# downstream sanity gates and ``nn_output_caps``.
CL_MAX: float = 1.5


def cl_ref_steady(q_pa: torch.Tensor) -> torch.Tensor:
    """Steady-state CL that balances aircraft weight at the given ``q``.

    ``CL_steady = m_ref · g / (q · S_REF)`` — the lift coefficient the
    NN-emitted residual is centered around in the CL-mode PhysicsLayer.
    The clip protects against pathological near-zero ``q`` from corrupted
    ODE substeps at very low TAS (matches the protection used in
    ``_compute_cl_residual`` on the numpy side).
    """
    q_safe = torch.clamp(q_pa, min=100.0)
    return torch.full_like(q_safe, _M_REF_KG * G) / (q_safe * S_REF_A320_M2)


def cl_ref_steady_np(q_pa: np.ndarray) -> np.ndarray:
    """NumPy twin of :func:`cl_ref_steady` for dataset preprocessing.

    Used by ``_compute_cl_residual`` (the analytical inverse of the
    PhysicsLayer CL-mode reconstruction) so target and prediction stay
    algebraically symmetric.
    """
    q_safe = np.maximum(q_pa, 100.0)
    return np.asarray(_M_REF_KG * G / (q_safe * S_REF_A320_M2), dtype=np.float64)


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

        if "fdm_cl_residual" in x and "fdm_lift_residual_norm" in x:
            msg = (
                "PhysicsLayer received both `fdm_cl_residual` and "
                "`fdm_lift_residual_norm`; the two lift-reconstruction modes "
                "are mutually exclusive."
            )
            raise ValueError(msg)

        if "fdm_t_minus_d_norm" in x:
            # The NN learns *normalised* (adimensional) quantities so the
            # output scale matches the legacy ``(a_spec, n_z_residual)``
            # baseline (p999 ~ 1 m/s² for a_spec, ~ 0.07 for n_z_residual).
            # Forces in Newtons are reconstructed analytically here:
            #
            #   T - D = t_minus_d_norm · m_ref          (~ m_ref · a_spec_baseline)
            #   L     = L_residual_norm · m_ref · g + m · g
            #         = m·g + m_ref·g · L_residual_norm
            #
            # Then the equations of motion expose ``m`` in the denominator
            # so the MassEncoder receives a non-zero gradient via both
            # ``d_TAS`` and ``d_gamma``:
            #
            #   d_TAS   = (t_minus_d_norm * m_ref) / m  -  g*sin(gamma)
            #   d_gamma = ((L_res_norm * m_ref*g) / m  +  g*(1 - cos gamma)) / V
            #
            # Per PHASE_1_MASS_ENCODER.md §4.3 the underlying forces don't
            # depend on m (engine + aerodynamics), so the NN can learn the
            # normalised quantities universally while m absorbs the per-
            # flight variation.
            t_minus_d_norm = x["fdm_t_minus_d_norm"]
            mass = x["fdm_mass_kg"]
            d_tas = t_minus_d_norm * (_M_REF_KG / mass) - G * torch.sin(gamma)

            if "fdm_cl_residual" in x:
                # CL-mode reconstruction: ``L = q · S · (CL_steady(q) + cl_residual)``
                # where ``CL_steady(q) = m_ref · g / (q · S)`` is the lift
                # coefficient that exactly balances weight at the current
                # dynamic pressure. This makes the formulation algebraically
                # equivalent to Newton (``L = m_ref·g + q·S·cl_residual``) so
                # the NN learns a small correction around 0 in every phase
                # instead of a phase-dependent shift on top of CL_REF=0.5.
                # See AXM-1739 cl_distribution.md §6 for the empirical
                # evidence behind this baseline choice. The lift no longer
                # depends on ``m``; mass identifiability is preserved through
                # the ``1/m`` factor in ``d_gamma``.
                cl_residual = x["fdm_cl_residual"]
                q_pa = x["fdm_q_pa"]
                lift = q_pa * S_REF_A320_M2 * (cl_ref_steady(q_pa) + cl_residual)
                d_gamma = (lift / mass - G * torch.cos(gamma)) / tas_safe
            else:
                lift_residual_norm = x["fdm_lift_residual_norm"]
                d_gamma = (
                    lift_residual_norm * (_M_REF_KG / mass) * G + G * (1.0 - torch.cos(gamma))
                ) / tas_safe
                lift = lift_residual_norm * _M_REF_KG * G + mass * G

            out: dict[str, torch.Tensor] = {
                "fdm_d_tas_ms2": d_tas,
                "fdm_d_gamma_rads": d_gamma,
                "fdm_d_mass_kgs": torch.zeros_like(mass),
                # Reconstructed full lift (Newtons), exposed for downstream
                # diagnostics (CL_shadow once Phase 1.5 lands).
                "fdm_lift_N": lift,
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
