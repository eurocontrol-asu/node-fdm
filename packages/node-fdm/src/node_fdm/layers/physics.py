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
    "cl_baseline",
    "cl_baseline_np",
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
# [0.3, 0.7] band in cruise — used as a diagnostic anchor in reports.
#
# **Note (post-AXM-1739)**: at runtime, the CL-mode PhysicsLayer does
# NOT use this constant. It calls ``cl_baseline(q, mass)`` which returns
# ``mass · g / (q · S_REF)`` — the CL that exactly balances weight at
# the current dynamic pressure and mass. With ``CL_REF=0.5`` constant
# the NN had to absorb a phase-dependent shift; with the q-only
# ``m_ref·g/(q·S)`` baseline (intermediate AXM-1739 attempt), the NN
# still had to absorb ``(m - m_ref)·g/(q·S)`` because ``m`` was not in
# the baseline. The current mass-aware baseline makes CL-mode
# algebraically equivalent to Newton (``L = m·g + q·S·cl_residual``),
# with ``mass`` sourced from the state vector (not the NN), preserving
# the AC4 identifiability invariant.
CL_REF: float = 0.5
# A320 stall coefficient at 1 g, clean configuration, sea level. Used by
# downstream sanity gates and ``nn_output_caps``.
CL_MAX: float = 1.5


def cl_baseline(q_pa: torch.Tensor, mass_kg: torch.Tensor) -> torch.Tensor:
    """Mass-aware steady-state CL that balances weight at the current ``q``.

    ``CL_baseline = mass · g / (q · S_REF)`` -- the lift coefficient that
    exactly balances weight at the current dynamic pressure and current
    aircraft mass (P&S Part 1 §5: ``C_L = 2·m·g/(gamma_air·p_inf·M^2·S_ref)``,
    equivalent to ``m·g/(q·S)``).

    The NN-emitted ``cl_residual`` is the small correction around this
    analytical baseline. This makes CL-mode algebraically equivalent to
    Newton-mode: ``L = q·S·(CL_baseline + cl_residual) = m·g + q·S·cl_residual``.
    The ``mass`` input comes from the state vector (``x["fdm_mass_kg"]``)
    -- never from the NN -- preserving the AC4 identifiability invariant
    that the longitudinal NN backbone is mass-agnostic.

    Args:
        q_pa: Dynamic pressure (Pa). Clamped at 100 Pa to protect against
            pathological near-zero ``q`` from corrupted ODE substeps at
            very low TAS.
        mass_kg: Aircraft mass at the current step (kg). Comes from the
            propagated state, initialised by the MassEncoder.
    """
    q_safe = torch.clamp(q_pa, min=100.0)
    return mass_kg * G / (q_safe * S_REF_A320_M2)


def cl_baseline_np(q_pa: np.ndarray, mass_kg: np.ndarray) -> np.ndarray:
    """NumPy twin of :func:`cl_baseline` for dataset preprocessing.

    Used by ``_compute_cl_residual`` (the analytical inverse of the
    PhysicsLayer CL-mode reconstruction) so target and prediction stay
    algebraically symmetric.
    """
    q_safe = np.maximum(q_pa, 100.0)
    return np.asarray(mass_kg * G / (q_safe * S_REF_A320_M2), dtype=np.float64)


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
                # CL-mode reconstruction with mass-aware baseline:
                #
                #   L = q · S · (CL_baseline(q, m) + cl_residual)
                #     = q · S · (m·g / (q·S))  +  q·S·cl_residual
                #     = m·g  +  q·S·cl_residual
                #
                # This is algebraically equivalent to Newton-mode at m=m_ref:
                # the NN learns a small correction around the analytical
                # ``m·g`` baseline, exactly like Newton's
                # ``lift_residual_norm·m_ref·g``. The ``mass`` term comes
                # from the state vector (propagated by the ODE, initialised
                # by MassEncoder), not from the NN -- so the AC4
                # identifiability invariant (NN backbone mass-agnostic)
                # is preserved.
                #
                # AXM-1739 post-mortem: the previous baselines (``CL_REF=0.5``
                # constant, and the m_ref-only ``CL_steady(q) = m_ref·g/(q·S)``)
                # failed because they forced the NN to absorb the phase shift
                # ``(m·g - baseline)``, which is huge at low altitude. See
                # ``data/models/full_hybrid_v3/cl_distribution.md`` §6 for
                # the empirical evidence.
                cl_residual = x["fdm_cl_residual"]
                q_pa = x["fdm_q_pa"]
                lift = q_pa * S_REF_A320_M2 * (cl_baseline(q_pa, mass) + cl_residual)
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
