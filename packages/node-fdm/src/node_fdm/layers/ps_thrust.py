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

# Engine characteristic constants for TET-nonlinear forward (Eqs E3-E5).
_A320_M_EC = 0.701  # Table 1, engine characteristic Mach
_A320_T_REC = 5.59  # Table 1, engine characteristic throttle constant
_A320_TET_MCC_K = 1529.0  # Table 1, max continuous-climb turbine entry temp (K)
# TET at idle — not in EEDB Table 1 ; estimated from ICAO LTO data
# (idle mode fuel flow + std combustion modelling). Linear-throttle
# interpolation `TET(chi) = TET_idle + chi · (TET_MCC - TET_idle)` is
# itself an approximation absorbed by the bounded NN correction.
_A320_TET_IDLE_K = 800.0

# Eq 28 thrust-coefficient ratio constant (P&S Part 3).
_C_T_RATIO_K = 0.55

# Eq E3 thrust-coefficient slope (h_4 = 1 + slope·(T_R - 1)).
_E3_SLOPE = 2.5

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
        alt_m      — altitude in metres (used by ``mode='tet_e3e5'`` only,
                     for atmosphere-independent total-temp ratio ; the linear
                     mode discards it).
        temp_k     — static air temperature in K (used by ``mode='tet_e3e5'``).
        q_pa       — dynamic pressure in Pa (from ``fdm_q_pa``).

    Output : ``T_total`` tensor matching the input shape (Newtons) — total
    aircraft thrust summed across both engines, no NN correction yet.
    The downstream PhysicsLayer applies ``(1 + 0.05 * tanh(t_correction))``
    to compose the final thrust with the bounded NN tweak.

    Two forward modes :
        - ``'linear'`` : ``C_T_inst = chi · (C_T)_etaB(M)`` (v13 baseline).
          Max thrust at full throttle is `(C_T)_etaB(M)` itself.
        - ``'tet_e3e5'`` : `C_T_inst = chi · h_4(chi, M, T_inf) · (C_T)_etaB(M)`
          (v13c Exp 17). The TET-nonlinear h_4 expands the max thrust
          envelope to cover regimes where the linear model bottoms out
          (cf. Exp 15 : 17 % of cruise needs T > T_PS_linear_max).
    """

    m_do: torch.Tensor
    c_t_do: torch.Tensor
    c_t_ratio_k: torch.Tensor
    s_ref: torch.Tensor
    denom_const: torch.Tensor
    m_ec: torch.Tensor
    t_rec: torch.Tensor
    tet_mcc: torch.Tensor
    tet_idle: torch.Tensor
    e3_slope: torch.Tensor

    def __init__(self, mode: str = "linear") -> None:
        super().__init__()
        if mode not in {"linear", "tet_e3e5"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self.register_buffer("m_do", torch.tensor(_A320_M_DO, dtype=torch.float32))
        self.register_buffer("c_t_do", torch.tensor(_A320_C_T_DO, dtype=torch.float32))
        self.register_buffer(
            "c_t_ratio_k",
            torch.tensor(_C_T_RATIO_K, dtype=torch.float32),
        )
        self.register_buffer("s_ref", torch.tensor(S_REF_A320_M2, dtype=torch.float32))
        denom = 1.0 + _C_T_RATIO_K * _A320_M_DO
        self.register_buffer("denom_const", torch.tensor(denom, dtype=torch.float32))
        # TET-nonlinear buffers (used only in mode='tet_e3e5').
        self.register_buffer("m_ec", torch.tensor(_A320_M_EC, dtype=torch.float32))
        self.register_buffer("t_rec", torch.tensor(_A320_T_REC, dtype=torch.float32))
        self.register_buffer("tet_mcc", torch.tensor(_A320_TET_MCC_K, dtype=torch.float32))
        self.register_buffer("tet_idle", torch.tensor(_A320_TET_IDLE_K, dtype=torch.float32))
        self.register_buffer("e3_slope", torch.tensor(_E3_SLOPE, dtype=torch.float32))

    def forward(
        self,
        throttle: torch.Tensor,
        mach: torch.Tensor,
        alt_m: torch.Tensor,
        temp_k: torch.Tensor,
        q_pa: torch.Tensor,
    ) -> torch.Tensor:
        """Return analytical T_total (Newtons) from Eq 28 + Eq 17.

        In ``mode='tet_e3e5'`` additionally composes the Eqs E3-E5 TET
        nonlinearity, extending C_T beyond ``(C_T)_etaB`` at full throttle.
        """
        del alt_m  # accepted for API symmetry, unused by both modes
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

        if self.mode == "tet_e3e5":
            # Eqs E3-E5 : TET-driven thrust amplification beyond (C_T)_etaB.
            # TET(chi) linear interpolation between idle and max-continuous-climb.
            tet = self.tet_idle + chi * (self.tet_mcc - self.tet_idle)
            # Eq E4 : throttle parameter T_R(M, TET, T_inf).
            # T_R = (1/T_REC) · (TET/T_inf) · [1 - 0.53·(M - M_EC)²] / (1 + 0.2·M²)
            temp_safe = temp_k.clamp(min=180.0)  # stratosphere floor
            tet_ratio = tet / temp_safe
            mach_dev = mach_safe - self.m_ec
            num = 1.0 - 0.53 * mach_dev.pow(2)
            den = 1.0 + 0.2 * mach_safe.pow(2)
            t_r = (1.0 / self.t_rec) * tet_ratio * num / den
            # Eq E3 : h_4 = 1 + 2.5·(T_R - 1). Clamp to avoid numerical blow-up.
            h_4 = (1.0 + self.e3_slope * (t_r - 1.0)).clamp(min=0.2, max=3.0)
            c_t_inst = chi * h_4 * c_t_eta_b
        else:
            c_t_inst = chi * c_t_eta_b

        # Eq 17 rearranged : T_total = C_T * q_inf * S_ref.
        q_safe = q_pa.clamp(min=100.0)
        return c_t_inst * q_safe * self.s_ref
