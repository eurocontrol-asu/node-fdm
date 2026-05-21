"""Poll-Schumann analytical drag-polar layer (Phase 2).

Implements a torch-native version of the Part 3 §4 full polar
(`L_over_D_part3` in `poll_schumann_lib`) hard-coded for the A320.
Returns the analytical C_D from observed `(C_L, M, alt_or_temp, q, V)`.

The downstream architecture composes this with a bounded NN
correction :

    C_D = C_D_PS * (1 + 0.05 * tanh(cd_correction_NN))
    D   = 0.5 * rho * V^2 * S_REF * C_D

so the NN can flex the drag by +/-5 % around the physics prior. This
forces the rest of the NN (the `fdm_t_minus_d_norm` head) to absorb
the thrust component, which is the Phase 2 ticket's central goal.

Pure analytical layer (no trainable params, no NN gradient through
the polar coefficients). The constants come from the
`poll_schumann_lib` A320 entry (Table 1 geometry + Table 2 psi) and
have been verified against `polar_part3.full_polar` to < 1 % error
in the A320 cruise envelope (M = 0.65-0.82, FL 300-410).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from node_fdm.layers.physics import S_REF_A320_M2

__all__ = [
    "PSDragLayer",
    "compute_a320_polar_constants",
]


# A320 geometry + polar fit constants (Table 1 + Table 2, ps_aircraft).
_A320_PSI_0 = 7.846
_A320_SPAN_M = 34.10
_A320_S_REF_M2 = 122.4  # ps_aircraft Table 1 ; node-fdm uses 122.6
_A320_SWEEP_DEG = 25.0
_A320_AR = _A320_SPAN_M * _A320_SPAN_M / _A320_S_REF_M2  # ~ 9.50
_A320_MTF_AC = 0.87  # Part 3 §4 typical fit
_A320_J_1 = 11.0
_A320_J_2 = 1.0

# Skin friction power-law (Eq 28 / Part 3 §4 Eq 33).
_SKIN_FRICTION_A = 0.0269
_SKIN_FRICTION_B = 0.14

# Sutherland's law constants (Eq 25, ps_core.atmosphere).
_MU_REF_KG_MS = 1.716e-5
_T_REF_K = 273.15
_S_SUTHERLAND_K = 110.4

# Reference length for Reynolds (Eq 3 ps_core.reynolds). l_ref = sqrt(S_ref).
_L_REF_M = math.sqrt(_A320_S_REF_M2)

# Oswald factor pre-computed for A320 at typical cruise (Shevell k_1 +
# wing-fuselage interference + winglets). Calibrated against
# ``poll_schumann_lib`` Part 3 §4 oswald_with_components at AR=9.5,
# sweep=25, c_d0=0.0237. Computed once via :func:`compute_a320_polar_constants`.
_A320_E_LS = 0.778
_A320_K = 1.0 / (math.pi * _A320_AR * _A320_E_LS)  # ~ 0.0431


def compute_a320_polar_constants() -> dict[str, float]:
    """Return the constants this layer uses, computed *from* poll_schumann_lib.

    Useful for unit-tests and AC2 (compare C_D against Table 2 / full_polar).
    Imported lazily because the lib lives outside the repo.
    """
    import sys as _sys

    _src = "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages/ps-core/src"
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    _src2 = "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages/ps-aircraft/src"
    if _src2 not in _sys.path:
        _sys.path.insert(0, _src2)
    from ps_aircraft import oswald_with_components  # type: ignore[import-not-found]

    e_ls = oswald_with_components(
        aspect_ratio=_A320_AR,
        sweep_deg=_A320_SWEEP_DEG,
        c_d0=0.0237,  # cruise-typical c_d0
        bf_over_span=4.0 / _A320_SPAN_M,  # body fineness / span ratio
    )
    return {
        "psi_0": _A320_PSI_0,
        "AR": _A320_AR,
        "S_ref_m2": _A320_S_REF_M2,
        "sweep_deg": _A320_SWEEP_DEG,
        "MTF_ac": _A320_MTF_AC,
        "j_1": _A320_J_1,
        "j_2": _A320_J_2,
        "e_LS": float(e_ls),
        "k_induced": 1.0 / (math.pi * _A320_AR * float(e_ls)),
        "skin_friction_a": _SKIN_FRICTION_A,
        "skin_friction_b": _SKIN_FRICTION_B,
        "l_ref_m": _L_REF_M,
    }


class PSDragLayer(nn.Module):
    """Analytical C_D(C_L, M, Re | psi_A320) from Poll-Schumann Part 3 §4.

    Inputs (all batched tensors, shape ``(B,)`` or ``(B, T)``) :
        c_l        — lift coefficient (computed by PhysicsLayer from CL-mode).
        mach       — Mach number (from trajectory layer ``era_mach``).
        temp_K     — static air temperature (from ``era_temp_K``).
        q_pa       — dynamic pressure (from trajectory layer ``fdm_q_pa``).
        tas_ms     — true airspeed in m/s (from state ``era_tas_ms``).

    Output : ``c_d_ps`` tensor matching the input shape — the analytical
    drag coefficient, no NN correction yet. The downstream PhysicsLayer
    applies ``(1 + 0.05 * tanh(cd_correction))`` to compose the final
    drag with the bounded NN tweak.

    Uses ``S_REF_A320_M2`` from the PhysicsLayer (122.6) for the
    Reynolds reference length, matching the rest of the project.
    """

    cos_sweep: torch.Tensor
    cos_sweep_cubed: torch.Tensor
    skin_a: torch.Tensor
    skin_b: torch.Tensor
    psi_0: torch.Tensor
    k_induced: torch.Tensor
    mtf_ac: torch.Tensor
    j_1: torch.Tensor
    j_2: torch.Tensor
    mu_ref: torch.Tensor
    t_ref: torch.Tensor
    s_suth: torch.Tensor
    l_ref: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        cs = math.cos(math.radians(_A320_SWEEP_DEG))
        # Use S_REF_A320_M2 (122.6) for Reynolds reference length to match
        # the rest of node-fdm-v2 — the geometry-side l_ref is taken from
        # the canonical project-wide constant, not from ps_aircraft's
        # 122.4 m^2 entry. The 0.08 % offset is operationally invisible.
        l_ref = math.sqrt(S_REF_A320_M2)
        self.register_buffer("cos_sweep", torch.tensor(cs, dtype=torch.float32))
        self.register_buffer(
            "cos_sweep_cubed",
            torch.tensor(cs * cs * cs, dtype=torch.float32),
        )
        self.register_buffer(
            "skin_a",
            torch.tensor(_SKIN_FRICTION_A, dtype=torch.float32),
        )
        self.register_buffer(
            "skin_b",
            torch.tensor(_SKIN_FRICTION_B, dtype=torch.float32),
        )
        self.register_buffer("psi_0", torch.tensor(_A320_PSI_0, dtype=torch.float32))
        self.register_buffer("k_induced", torch.tensor(_A320_K, dtype=torch.float32))
        self.register_buffer("mtf_ac", torch.tensor(_A320_MTF_AC, dtype=torch.float32))
        self.register_buffer("j_1", torch.tensor(_A320_J_1, dtype=torch.float32))
        self.register_buffer("j_2", torch.tensor(_A320_J_2, dtype=torch.float32))
        self.register_buffer("mu_ref", torch.tensor(_MU_REF_KG_MS, dtype=torch.float32))
        self.register_buffer("t_ref", torch.tensor(_T_REF_K, dtype=torch.float32))
        self.register_buffer(
            "s_suth",
            torch.tensor(_S_SUTHERLAND_K, dtype=torch.float32),
        )
        self.register_buffer("l_ref", torch.tensor(l_ref, dtype=torch.float32))

    def forward(
        self,
        c_l: torch.Tensor,
        mach: torch.Tensor,
        temp_k: torch.Tensor,
        q_pa: torch.Tensor,
        tas_ms: torch.Tensor,
    ) -> torch.Tensor:
        """Return analytical C_D from Part 3 §4 full polar."""
        # 1. Dynamic viscosity via Sutherland's law (Eq 25).
        temp_safe = temp_k.clamp(min=180.0)  # stratosphere floor
        mu = (
            self.mu_ref
            * (temp_safe / self.t_ref).clamp(min=0.1) ** 1.5
            * (self.t_ref + self.s_suth)
            / (temp_safe + self.s_suth)
        )
        # 2. Air density from dynamic pressure and TAS (q = 0.5 rho V^2).
        tas_safe = tas_ms.clamp(min=50.0)
        rho = (2.0 * q_pa.clamp(min=100.0)) / (tas_safe * tas_safe)
        # 3. Aircraft Reynolds number (Eq 3).
        re_ac = (rho * tas_safe * self.l_ref / mu.clamp(min=1e-7)).clamp(min=1e3)
        # 4. Skin friction power-law (Eq 33 / 28).
        c_f = self.skin_a / re_ac**self.skin_b
        # 5. Profile drag C_d0 = psi_0 · c_f (Eq 32).
        c_d0 = self.psi_0 * c_f
        # 6. Wave-drag variables (Eqs 40-41).
        cl_safe = c_l.clamp(min=0.0, max=2.0)
        m_cc = self.mtf_ac - 0.10 * cl_safe / (self.cos_sweep * self.cos_sweep)
        m_cc_safe = m_cc.clamp(min=0.30)
        x = mach * self.cos_sweep / m_cc_safe
        # 7. Wave drag — drag creep region only (Eq 42) ; the project
        #    stays well below the drag-rise X_DO ~ 1.4 threshold in cruise.
        c_dw = self.cos_sweep_cubed * self.j_1 * torch.clamp(x - self.j_2, min=0.0).pow(2)
        # 8. Induced drag C_l^2 / (pi AR e_LS) (Eq 38).
        c_di = self.k_induced * cl_safe * cl_safe
        # 9. Total drag (Eq 39).
        return c_d0 + c_di + c_dw
