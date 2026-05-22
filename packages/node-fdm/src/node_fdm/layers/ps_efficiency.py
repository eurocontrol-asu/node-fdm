"""Poll-Schumann analytical overall propulsive efficiency layer (Phase 4).

Implements a torch-native A320 + CFM56-5B4_P efficiency prior using
P&S Part 3 Eqs 24-29 (high-thrust branch) plus Appendix C cubic for
the low-thrust regime (`C_T / (C_T)_etaB < 0.3`) :

    eta_2(BPR)   = 0.65 * (1 - 0.035 * BPR)                                [Eq 29]
    h_1(M)       = (M / M_DO) ** eta_2                                     [Eq 27]
    (eta_o)_B    = h_1(M) * (eta_o)_DO
    h_2(M)       = (1 + 0.55*M) / (1 + 0.55*M_DO) * (M_DO/M) ** 2          [Eq 28]
    (C_T)_etaB   = h_2(M) * (C_T)_DO
    arg          = C_T / (C_T)_etaB - 1
    Omega(M)     = 1.30 * (0.4 - M) for M in [0.2, 0.4], else clipped      [Eqs 25-26]
    h_0          = (1 - 0.43 * arg**2) * (1 + Omega * arg**2)              [Eq 24]
    eta_o        = h_0 * (eta_o)_B

Low-thrust fallback (Appendix C1-C5) for C_T / (C_T)_etaB < 0.3 :

    ratio        = C_T / (C_T)_etaB
    H_1, H_2, H_3 = 6.560*(1+0.824*Om), -19.43*(1+1.053*Om), 21.11*(1+1.063*Om)
    h_0_low      = H_1*ratio + H_2*ratio**2 + H_3*ratio**3
    eta_o        = h_0_low * (eta_o)_B

The downstream architecture (`adsb_hybrid_v14_psefficiency`, Phase 4)
composes this with a bounded NN correction :

    eta = eta_o * (1 + 0.03 * tanh(eta_correction_NN))

so the NN may flex the analytical prior by +/-3 % (tighter than the
+/-5 % used in Phase 2 / 3 because eta is more tightly physically
constrained).

Pure analytical layer (no trainable params, no NN gradient through
the engine constants). The constants come from the ``ps_engine``
database Table 1 + Table 2 (A320 entry, CFM56-5B4_P), kept hard-coded
here so the layer stays a leaf node in the import graph.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "PSEfficiencyLayer",
    "compute_a320_efficiency_constants",
    "eta_o_reference",
]


# A320 + CFM56-5B4_P engine characteristics (Table 1 + Table 2, ps_engine).
_A320_M_DO = 0.753  # Table 2, design-optimum Mach
_A320_C_T_DO = 0.0347  # Table 1, design-optimum total thrust coefficient
_A320_BPR = 5.6  # Table 1, bypass ratio
_A320_ETA_O_DO = 0.309  # Table 1, design-optimum overall efficiency

# Eq 29 : eta_2 = 0.65 * (1 - 0.035 * BPR). Computed once from BPR=5.6.
# 0.65 * (1 - 0.035 * 5.6) = 0.65 * (1 - 0.196) = 0.65 * 0.804 = 0.5226
_ETA2_LEAD = 0.65
_ETA2_BPR_SLOPE = 0.035

# Eq 28 thrust-coefficient ratio constant.
_C_T_RATIO_K = 0.55

# Eq 24 high-thrust curvature.
_H0_CURVATURE = 0.43

# Eqs 25-26 Omega.
_OMEGA_MACH_LOW = 0.2
_OMEGA_MACH_HIGH = 0.4
_OMEGA_SLOPE = 1.30

# Appendix C1-C5 low-thrust cubic coefficients.
_H1_BASE = 6.560
_H2_BASE = -19.43
_H3_BASE = 21.11
_H1_OMEGA = 0.824
_H2_OMEGA = 1.053
_H3_OMEGA = 1.063

# Threshold below which the cubic fallback applies (ratio = C_T / (C_T)_etaB).
LOW_THRUST_C_T_RATIO_CUTOFF = 0.3

# Physical clamps to keep eta out of pathological regimes.
ETA_MIN = 0.05  # below this the fuel flow Eq 19 blows up
ETA_MAX = 0.45  # R8 of PHASE_4_FUEL_FLOW_TICKET (max realistic cruise)


def compute_a320_efficiency_constants() -> dict[str, float]:
    """Return the constants this layer uses, computed *from* ps_engine database.

    Useful for unit-tests and AC2 (compare eta_torch against ``ps_engine``-based
    reference composition). Imported lazily because the lib lives outside
    the repo.
    """
    import sys as _sys

    _src = "/Users/gabriel/axm/04-papers/PS_MODEL/poll_schumann_lib/packages/ps-engine/src"
    if _src not in _sys.path:
        _sys.path.insert(0, _src)
    from ps_engine.database import get_engine_params  # type: ignore[import-not-found]

    engine = get_engine_params("A320")
    eta_2 = _ETA2_LEAD * (1.0 - _ETA2_BPR_SLOPE * engine.BPR)
    return {
        "M_DO": engine.M_DO,
        "C_T_DO": engine.C_T_DO,
        "BPR": engine.BPR,
        "eta_o_DO": engine.eta_o_DO,
        "eta_2": eta_2,
        "C_T_ratio_K": _C_T_RATIO_K,
        "H0_curvature": _H0_CURVATURE,
        "low_thrust_cutoff": LOW_THRUST_C_T_RATIO_CUTOFF,
    }


def eta_o_reference(
    c_t: float,
    mach: float,
    *,
    use_low_thrust_fallback: bool = True,
) -> float:
    """NumPy/scalar reference for the same forward chain used by the torch layer.

    Implemented as a plain Python function with the *same constants and same
    equations* so that any algebraic discrepancy between the layer and this
    function is a numerical / torch-implementation issue, not a model
    difference. Used by AC2 grid parity tests.
    """
    m_safe = max(mach, 0.1)
    # Eq 29 : eta_2(BPR). Scalar.
    eta_2 = _ETA2_LEAD * (1.0 - _ETA2_BPR_SLOPE * _A320_BPR)
    # Eq 27 : h_1(M) = (M / M_DO) ** eta_2.
    h_1 = (m_safe / _A320_M_DO) ** eta_2
    eta_o_b = h_1 * _A320_ETA_O_DO
    # Eq 28 : h_2(M).
    h_2 = (
        (1.0 + _C_T_RATIO_K * m_safe)
        / (1.0 + _C_T_RATIO_K * _A320_M_DO)
        * (_A320_M_DO / m_safe) ** 2
    )
    c_t_eta_b = h_2 * _A320_C_T_DO
    ratio = c_t / c_t_eta_b
    # Eqs 25-26 : Omega(M).
    if m_safe >= _OMEGA_MACH_HIGH:
        omega = 0.0
    elif m_safe >= _OMEGA_MACH_LOW:
        omega = _OMEGA_SLOPE * (_OMEGA_MACH_HIGH - m_safe)
    else:
        omega = _OMEGA_SLOPE * (_OMEGA_MACH_HIGH - _OMEGA_MACH_LOW)

    if ratio < LOW_THRUST_C_T_RATIO_CUTOFF and use_low_thrust_fallback:
        # Appendix C1-C5 cubic.
        h1c = _H1_BASE * (1.0 + _H1_OMEGA * omega)
        h2c = _H2_BASE * (1.0 + _H2_OMEGA * omega)
        h3c = _H3_BASE * (1.0 + _H3_OMEGA * omega)
        h_0 = h1c * ratio + h2c * ratio**2 + h3c * ratio**3
    else:
        # Eq 24 (high-thrust).
        arg = ratio - 1.0
        h_0 = (1.0 - _H0_CURVATURE * arg * arg) * (1.0 + omega * arg * arg)

    return h_0 * eta_o_b


class PSEfficiencyLayer(nn.Module):
    """Analytical eta_o(C_T, M | A320 + CFM56-5B4_P) from Poll-Schumann Part 3.

    Inputs (all batched tensors, shape ``(B,)`` or ``(B, T)``) :
        c_t      — instantaneous total thrust coefficient C_T = T / (q · S_ref).
        mach     — Mach number (from trajectory layer ``era_mach``).

    Output : ``eta_o`` tensor matching the input shape (dimensionless,
    typically in [0.20, 0.45] for cruise stable). No NN correction yet —
    the downstream PhysicsLayer applies ``(1 + 0.03 * tanh(eta_correction))``
    to compose the final efficiency with the bounded NN tweak.

    Two forward modes :
        - ``'cruise'`` : Eq 24 only (high-thrust branch). Sufficient for
          A320 ADS-B data where idle / LTO regime never appears.
        - ``'with_idle_fallback'`` : switches to Appendix C cubic when
          ``C_T / (C_T)_etaB < 0.3``. Use this in scripts that may
          encounter low-thrust samples (climb-out, descent, taxi).
    """

    m_do: torch.Tensor
    c_t_do: torch.Tensor
    eta_o_do: torch.Tensor
    eta_2: torch.Tensor
    c_t_ratio_k: torch.Tensor
    denom_const: torch.Tensor
    h0_curvature: torch.Tensor
    omega_slope: torch.Tensor
    omega_mach_low: torch.Tensor
    omega_mach_high: torch.Tensor
    omega_floor: torch.Tensor
    low_thrust_cutoff: torch.Tensor
    h1_base: torch.Tensor
    h2_base: torch.Tensor
    h3_base: torch.Tensor
    h1_omega: torch.Tensor
    h2_omega: torch.Tensor
    h3_omega: torch.Tensor

    def __init__(self, mode: str = "cruise") -> None:
        super().__init__()
        if mode not in {"cruise", "with_idle_fallback"}:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        eta_2 = _ETA2_LEAD * (1.0 - _ETA2_BPR_SLOPE * _A320_BPR)
        self.register_buffer("m_do", torch.tensor(_A320_M_DO, dtype=torch.float32))
        self.register_buffer("c_t_do", torch.tensor(_A320_C_T_DO, dtype=torch.float32))
        self.register_buffer("eta_o_do", torch.tensor(_A320_ETA_O_DO, dtype=torch.float32))
        self.register_buffer("eta_2", torch.tensor(eta_2, dtype=torch.float32))
        self.register_buffer(
            "c_t_ratio_k",
            torch.tensor(_C_T_RATIO_K, dtype=torch.float32),
        )
        denom = 1.0 + _C_T_RATIO_K * _A320_M_DO
        self.register_buffer("denom_const", torch.tensor(denom, dtype=torch.float32))
        self.register_buffer(
            "h0_curvature",
            torch.tensor(_H0_CURVATURE, dtype=torch.float32),
        )
        self.register_buffer("omega_slope", torch.tensor(_OMEGA_SLOPE, dtype=torch.float32))
        self.register_buffer(
            "omega_mach_low",
            torch.tensor(_OMEGA_MACH_LOW, dtype=torch.float32),
        )
        self.register_buffer(
            "omega_mach_high",
            torch.tensor(_OMEGA_MACH_HIGH, dtype=torch.float32),
        )
        omega_floor = _OMEGA_SLOPE * (_OMEGA_MACH_HIGH - _OMEGA_MACH_LOW)
        self.register_buffer("omega_floor", torch.tensor(omega_floor, dtype=torch.float32))
        self.register_buffer(
            "low_thrust_cutoff",
            torch.tensor(LOW_THRUST_C_T_RATIO_CUTOFF, dtype=torch.float32),
        )
        self.register_buffer("h1_base", torch.tensor(_H1_BASE, dtype=torch.float32))
        self.register_buffer("h2_base", torch.tensor(_H2_BASE, dtype=torch.float32))
        self.register_buffer("h3_base", torch.tensor(_H3_BASE, dtype=torch.float32))
        self.register_buffer("h1_omega", torch.tensor(_H1_OMEGA, dtype=torch.float32))
        self.register_buffer("h2_omega", torch.tensor(_H2_OMEGA, dtype=torch.float32))
        self.register_buffer("h3_omega", torch.tensor(_H3_OMEGA, dtype=torch.float32))

    def _omega(self, mach_safe: torch.Tensor) -> torch.Tensor:
        """Eqs 25-26 : Omega(M) low-Mach correction, vectorised."""
        # Default to the >0.4 case (omega = 0), then patch the two lower bands.
        omega = torch.zeros_like(mach_safe)
        # M in [0.2, 0.4] : linear slope.
        in_mid = (mach_safe >= self.omega_mach_low) & (mach_safe < self.omega_mach_high)
        omega = torch.where(
            in_mid,
            self.omega_slope * (self.omega_mach_high - mach_safe),
            omega,
        )
        # M < 0.2 : clipped at omega_floor.
        below = mach_safe < self.omega_mach_low
        omega = torch.where(below, self.omega_floor.expand_as(omega), omega)
        return omega

    def forward(
        self,
        c_t: torch.Tensor,
        mach: torch.Tensor,
    ) -> torch.Tensor:
        """Return analytical eta_o (dimensionless) from Eqs 24-29.

        In ``mode='with_idle_fallback'`` additionally composes Appendix C
        cubic for samples where ``C_T / (C_T)_etaB < 0.3``.
        """
        mach_safe = mach.clamp(min=0.15)
        # Eq 27 : h_1 = (M/M_DO) ** eta_2.
        h_1 = (mach_safe / self.m_do).pow(self.eta_2)
        eta_o_b = h_1 * self.eta_o_do
        # Eq 28 : h_2(M).
        h_2 = (
            (1.0 + self.c_t_ratio_k * mach_safe)
            / self.denom_const
            * (self.m_do / mach_safe).pow(2)
        )
        c_t_eta_b = h_2 * self.c_t_do
        # Protect ratio from c_t_eta_b near zero (would need M -> 0).
        ratio = c_t / c_t_eta_b.clamp(min=1e-6)
        omega = self._omega(mach_safe)

        arg = ratio - 1.0
        arg_sq = arg * arg
        h0_high = (1.0 - self.h0_curvature * arg_sq) * (1.0 + omega * arg_sq)

        if self.mode == "with_idle_fallback":
            h1c = self.h1_base * (1.0 + self.h1_omega * omega)
            h2c = self.h2_base * (1.0 + self.h2_omega * omega)
            h3c = self.h3_base * (1.0 + self.h3_omega * omega)
            h0_low = h1c * ratio + h2c * ratio.pow(2) + h3c * ratio.pow(3)
            h_0 = torch.where(ratio < self.low_thrust_cutoff, h0_low, h0_high)
        else:
            h_0 = h0_high

        return h_0 * eta_o_b
