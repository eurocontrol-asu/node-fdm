"""Poll-Schumann analytical thrust layer (Phase 3).

Implements a torch-native A320 + CFM56-5B4_P thrust prior using P&S
Part 3 Eq 28 + Eq 17 :

    h_2(M)       = (1 + 0.55 M) / (1 + 0.55 M_DO) * (M_DO/M)^2     [Eq 28]
    (C_T)_etaB   = h_2(M) * (C_T)_DO                               [normalisation]
    C_T_inst     = throttle · (C_T)_etaB                           [linear throttle]
    T_total      = C_T_inst · q_inf · S_ref                         [Eq 17 inverse]

The downstream architecture (`adsb_hybrid_v13_psthrust`, Phase 3)
composes this with a bounded NN correction :

    T = T_PS * (1 + 0.05 * tanh(t_correction_NN))

so the NN may flex the analytical prior by +/-5 %. The throttle chi
itself is learned by the NN as ``fdm_throttle_norm`` (state-latent),
mapped to [0.1, 1.0] via ``0.1 + 0.9 * sigmoid(...)`` upstream.

Pure analytical layer (no trainable params, no NN gradient through
the engine constants). The constants come from the ``ps_engine``
database Table 1 + Table 2 (A320 entry, CFM56-5B4_P), kept hard-coded
here so the layer stays a leaf node in the import graph.
"""

from __future__ import annotations

import torch
from torch import nn

from node_fdm.layers.physics import S_REF_A320_M2

__all__ = [
    "PSThrustLayer",
    "compute_a320_thrust_constants",
]


# A320 + CFM56-5B4_P engine characteristics (Table 1 + Table 2, ps_engine).
_A320_M_DO = 0.753  # Table 2, design-optimum Mach
_A320_C_T_DO = 0.0347  # Table 1, design-optimum total thrust coefficient
_A320_BPR = 5.6  # Table 1, bypass ratio (not used here but archived)
_A320_F00_ISA_KN = 225.0  # Table 1, SLS takeoff thrust (both engines, kN)
_A320_N_ENGINES = 2  # twin-engine

# Eq 28 thrust-coefficient ratio constant (P&S Part 3).
_C_T_RATIO_K = 0.55

# Throttle physical envelope (R8 of PHASE_3_THRUST_PS_TICKET).
THROTTLE_MIN = 0.1
THROTTLE_MAX = 1.0


def compute_a320_thrust_constants() -> dict[str, float]:
    """Return the constants this layer uses, computed *from* ps_engine database.

    Useful for unit-tests and AC2 (compare T_PS against ``ps_engine``-based
    reference composition). Imported lazily because the lib lives outside
    the repo.
    """
    import sys as _sys

    _src = "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages/ps-engine/src"
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from ps_engine.database import get_engine_params  # type: ignore[import-not-found]

    engine = get_engine_params("A320")
    return {
        "M_DO": engine.M_DO,
        "C_T_DO": engine.C_T_DO,
        "BPR": engine.BPR,
        "F00_ISA_kN": _A320_F00_ISA_KN,
        "n_engines": _A320_N_ENGINES,
        "S_ref_m2": S_REF_A320_M2,
        "C_T_ratio_K": _C_T_RATIO_K,
    }


def t_ps_reference(
    throttle: float,
    mach: float,
    q_pa: float,
) -> float:
    """NumPy/scalar reference for the same forward chain used by the torch layer.

    Implemented as a plain Python function with the *same constants and same
    equations* so that any algebraic discrepancy between the layer and this
    function is a numerical / torch-implementation issue, not a model
    difference. Used by AC2 grid parity tests.
    """
    m_safe = max(mach, 0.1)
    h_2 = (
        (1.0 + _C_T_RATIO_K * m_safe)
        / (1.0 + _C_T_RATIO_K * _A320_M_DO)
        * (_A320_M_DO / m_safe) ** 2
    )
    c_t_eta_b = h_2 * _A320_C_T_DO
    chi = min(max(throttle, 0.0), 1.0)
    c_t_inst = chi * c_t_eta_b
    return c_t_inst * q_pa * S_REF_A320_M2


class PSThrustLayer(nn.Module):
    """Analytical T(chi, M, h | A320 + CFM56-5B4_P) from Poll-Schumann Part 3.

    Inputs (all batched tensors, shape ``(B,)`` or ``(B, T)``) :
        throttle   — fractional throttle chi in [0, 1] (already sigmoid-mapped
                     by the architecture wrapper, so this layer just clamps
                     to [0, 1] for safety).
        mach       — Mach number (from trajectory layer ``era_mach``).
        alt_m      — altitude in metres (accepted for API symmetry with the
                     ticket spec ; unused because q_pa already encodes
                     ambient density).
        temp_k     — static air temperature in K (accepted for API symmetry ;
                     unused by the present forward model).
        q_pa       — dynamic pressure in Pa (from ``fdm_q_pa``).

    Output : ``T_total`` tensor matching the input shape (Newtons) — total
    aircraft thrust summed across both engines, no NN correction yet.
    The downstream PhysicsLayer applies ``(1 + 0.05 * tanh(t_correction))``
    to compose the final thrust with the bounded NN tweak.
    """

    m_do: torch.Tensor
    c_t_do: torch.Tensor
    c_t_ratio_k: torch.Tensor
    s_ref: torch.Tensor
    denom_const: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("m_do", torch.tensor(_A320_M_DO, dtype=torch.float32))
        self.register_buffer("c_t_do", torch.tensor(_A320_C_T_DO, dtype=torch.float32))
        self.register_buffer(
            "c_t_ratio_k",
            torch.tensor(_C_T_RATIO_K, dtype=torch.float32),
        )
        self.register_buffer("s_ref", torch.tensor(S_REF_A320_M2, dtype=torch.float32))
        # Pre-compute the Mach-independent denominator (1 + K M_DO) of Eq 28.
        denom = 1.0 + _C_T_RATIO_K * _A320_M_DO
        self.register_buffer("denom_const", torch.tensor(denom, dtype=torch.float32))

    def forward(
        self,
        throttle: torch.Tensor,
        mach: torch.Tensor,
        alt_m: torch.Tensor,
        temp_k: torch.Tensor,
        q_pa: torch.Tensor,
    ) -> torch.Tensor:
        """Return analytical T_total (Newtons) from Eq 28 + Eq 17."""
        del alt_m, temp_k  # accepted for API symmetry, unused by this model
        # Mach floor to keep (M_DO/M)^2 finite even at near-zero Mach
        # transients. The training data is filtered to TAS >= ~80 m/s so M
        # rarely drops below 0.15 ; this is a numerical safety bound only.
        mach_safe = mach.clamp(min=0.15)
        # Eq 28 : h_2(M) = (1 + K M) / (1 + K M_DO) * (M_DO / M)^2.
        h_2 = (
            (1.0 + self.c_t_ratio_k * mach_safe)
            / self.denom_const
            * (self.m_do / mach_safe).pow(2)
        )
        c_t_eta_b = h_2 * self.c_t_do
        # Clamp chi to the physical envelope (R8 of the Phase 3 ticket).
        chi = throttle.clamp(min=0.0, max=1.0)
        c_t_inst = chi * c_t_eta_b
        # Eq 17 rearranged : T_total = C_T * q_inf * S_ref.
        q_safe = q_pa.clamp(min=100.0)
        return c_t_inst * q_safe * self.s_ref
