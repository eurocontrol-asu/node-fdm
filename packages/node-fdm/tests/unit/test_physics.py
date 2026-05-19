from __future__ import annotations

import math

import pytest
import torch

from node_fdm.layers.physics import (
    _M_REF_KG,
    S_REF_A320_M2,
    V_MIN_CLAMP,
    G,
    PhysicsLayer,
    cl_baseline,
)


def test_legacy_mode_unchanged() -> None:
    """AC5: Legacy mode keeps the pre-ticket d_tas, d_gamma and n_z outputs."""
    layer = PhysicsLayer()

    out = layer(
        {
            "fdm_a_spec_ms2": torch.tensor(1.5),
            "fdm_n_z_residual": torch.tensor(0.2),
            "era_tas_ms": torch.tensor(240.0),
            "fdm_gamma_rad": torch.tensor(0.05),
        }
    )

    assert torch.allclose(out["fdm_d_tas_ms2"], torch.tensor(1.0098718), atol=1e-6)
    assert torch.allclose(out["fdm_d_gamma_rads"], torch.tensor(0.0082233), atol=1e-6)
    assert torch.allclose(out["fdm_n_z"], torch.tensor(1.2), atol=1e-6)


def test_newton_mode_d_tas_matches_formula() -> None:
    """Newton mode: d_tas = t_minus_d_norm * m_ref / m  -  g*sin(gamma)."""
    layer = PhysicsLayer()
    t_minus_d_norm = torch.tensor(1.5)
    mass = torch.tensor(65000.0)
    gamma = torch.tensor(0.05)

    out = layer(
        {
            "fdm_t_minus_d_norm": t_minus_d_norm,
            "fdm_lift_residual_norm": torch.tensor(0.0),
            "fdm_mass_kg": mass,
            "era_tas_ms": torch.tensor(240.0),
            "fdm_gamma_rad": gamma,
        }
    )

    expected = t_minus_d_norm * _M_REF_KG / mass - G * torch.sin(gamma)
    assert torch.allclose(out["fdm_d_tas_ms2"], expected, atol=1e-5)


def test_newton_mode_d_gamma_matches_formula() -> None:
    """Newton mode: d_gamma = (L_res_norm * m_ref*g / m  +  g*(1 - cos gamma)) / V."""
    layer = PhysicsLayer()
    lift_residual_norm = torch.tensor(0.05)
    mass = torch.tensor(65000.0)
    tas = torch.tensor(240.0)
    gamma = torch.tensor(0.05)

    out = layer(
        {
            "fdm_t_minus_d_norm": torch.tensor(0.0),
            "fdm_lift_residual_norm": lift_residual_norm,
            "fdm_mass_kg": mass,
            "era_tas_ms": tas,
            "fdm_gamma_rad": gamma,
        }
    )

    tas_safe = torch.clamp(tas, min=50.0)
    expected = (
        lift_residual_norm * _M_REF_KG * G / mass + G * (1.0 - torch.cos(gamma))
    ) / tas_safe
    assert torch.allclose(out["fdm_d_gamma_rads"], expected, atol=1e-5)


def test_newton_mode_d_mass_is_zero_tensor() -> None:
    """Newton mode emits a fresh zero tensor matching mass shape and device."""
    layer = PhysicsLayer()
    mass = torch.tensor([65000.0, 66000.0])

    out = layer(
        {
            "fdm_t_minus_d_norm": torch.tensor([0.5, 0.6]),
            "fdm_lift_residual_norm": torch.tensor([0.01, 0.02]),
            "fdm_mass_kg": mass,
            "era_tas_ms": torch.tensor([240.0, 250.0]),
            "fdm_gamma_rad": torch.tensor([0.05, 0.04]),
        }
    )

    d_mass = out["fdm_d_mass_kgs"]
    assert isinstance(d_mass, torch.Tensor)
    assert d_mass.shape == mass.shape
    assert d_mass.device == mass.device
    assert torch.equal(d_mass, torch.zeros_like(mass))
    assert d_mass is not mass


def test_newton_mode_exposes_full_lift() -> None:
    """Newton mode exposes the reconstructed full lift ``L = L_res_norm · m_ref·g + m·g``."""
    layer = PhysicsLayer()
    mass = torch.tensor(65000.0)

    out = layer(
        {
            "fdm_t_minus_d_norm": torch.tensor(0.0),
            "fdm_lift_residual_norm": torch.tensor(0.0),
            "fdm_mass_kg": mass,
            "era_tas_ms": torch.tensor(240.0),
            "fdm_gamma_rad": torch.tensor(0.0),
        }
    )

    expected = mass * G
    assert torch.allclose(out["fdm_lift_N"], expected, atol=1e-2)


def test_newton_mode_with_phi_bank() -> None:
    """Newton mode keeps the lateral coordinated-turn branch working."""
    layer = PhysicsLayer()
    tas = torch.tensor(240.0)
    phi_bank = torch.tensor(0.3)

    out = layer(
        {
            "fdm_t_minus_d_norm": torch.tensor(0.5),
            "fdm_lift_residual_norm": torch.tensor(0.0),
            "fdm_mass_kg": torch.tensor(65000.0),
            "era_tas_ms": tas,
            "fdm_gamma_rad": torch.tensor(0.05),
            "fdm_phi_bank_rad": phi_bank,
        }
    )

    expected = G / tas * torch.tan(phi_bank)
    assert torch.allclose(out["fdm_d_heading_rads"], expected, atol=1e-5)


def test_legacy_mode_with_phi_bank_still_works() -> None:
    """AC4: Legacy mode keeps the lateral coordinated-turn branch working."""
    layer = PhysicsLayer()
    tas = torch.tensor(240.0)
    phi_bank = torch.tensor(0.3)

    out = layer(
        {
            "fdm_a_spec_ms2": torch.tensor(1.5),
            "fdm_n_z_residual": torch.tensor(0.2),
            "era_tas_ms": tas,
            "fdm_gamma_rad": torch.tensor(0.05),
            "fdm_phi_bank_rad": phi_bank,
        }
    )

    expected = G / tas * torch.tan(phi_bank)
    assert torch.allclose(out["fdm_d_heading_rads"], expected, atol=1e-5)


def test_newton_mode_low_tas_clamped() -> None:
    """Newton mode reuses the 50 m/s TAS clamp for d_gamma."""
    layer = PhysicsLayer()
    mass = torch.tensor(65000.0)
    gamma = torch.tensor(0.05)
    lift_residual_norm = torch.tensor(0.05)

    out = layer(
        {
            "fdm_t_minus_d_norm": torch.tensor(0.0),
            "fdm_lift_residual_norm": lift_residual_norm,
            "fdm_mass_kg": mass,
            "era_tas_ms": torch.tensor(10.0),
            "fdm_gamma_rad": gamma,
        }
    )

    tas_safe = torch.tensor(50.0)
    expected = (
        lift_residual_norm * _M_REF_KG * G / mass + G * (1.0 - torch.cos(gamma))
    ) / tas_safe
    assert torch.isfinite(out["fdm_d_gamma_rads"])
    assert torch.allclose(out["fdm_d_gamma_rads"], expected, atol=1e-5)


def test_newton_mode_batch_shape() -> None:
    """Newton mode preserves batched tensor shapes for all emitted outputs."""
    layer = PhysicsLayer()
    batch_size = 8

    out = layer(
        {
            "fdm_t_minus_d_norm": torch.full((batch_size,), 0.5),
            "fdm_lift_residual_norm": torch.full((batch_size,), 0.05),
            "fdm_mass_kg": torch.full((batch_size,), 65000.0),
            "era_tas_ms": torch.full((batch_size,), 240.0),
            "fdm_gamma_rad": torch.full((batch_size,), 0.05),
        }
    )

    assert out["fdm_d_tas_ms2"].shape == (batch_size,)
    assert out["fdm_d_gamma_rads"].shape == (batch_size,)
    assert out["fdm_d_mass_kgs"].shape == (batch_size,)
    assert out["fdm_lift_N"].shape == (batch_size,)


# ---------------------------------------------------------------------------
# CL-mode branch (AXM-1737) — third branch keyed on ``fdm_cl_residual``.
# Purely additive: legacy and Newton tests above must keep passing unchanged.
# ---------------------------------------------------------------------------


def _cl_inputs(
    *,
    cl_residual: float = 0.0,
    t_minus_d_norm: float = 0.5,
    mass_kg: float = 60_000.0,
    tas_ms: float = 200.0,
    gamma_rad: float = 0.0,
    q_pa: float = 15_000.0,
    batch: int = 1,
    phi_bank_rad: float | None = None,
) -> dict[str, torch.Tensor]:
    def _t(v: float) -> torch.Tensor:
        return torch.full((batch,), v, dtype=torch.float64)

    x: dict[str, torch.Tensor] = {
        "fdm_t_minus_d_norm": _t(t_minus_d_norm),
        "fdm_cl_residual": _t(cl_residual),
        "fdm_mass_kg": _t(mass_kg),
        "era_tas_ms": _t(tas_ms),
        "fdm_gamma_rad": _t(gamma_rad),
        "fdm_q_pa": _t(q_pa),
    }
    if phi_bank_rad is not None:
        x["fdm_phi_bank_rad"] = _t(phi_bank_rad)
    return x


def test_cl_mode_branch_selected_on_cl_residual_key() -> None:
    """AC1: branch selection on ``fdm_cl_residual`` emits the documented keys."""
    layer = PhysicsLayer()
    out = layer(_cl_inputs())
    for key in (
        "fdm_d_tas_ms2",
        "fdm_d_gamma_rads",
        "fdm_d_mass_kgs",
        "fdm_lift_N",
    ):
        assert key in out


def test_cl_mode_raises_if_both_residual_keys_present() -> None:
    """AC2: strict-exclusive branch guard rejects ambiguous input."""
    layer = PhysicsLayer()
    x = _cl_inputs()
    x["fdm_lift_residual_norm"] = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"cl_residual.*lift_residual_norm"):
        layer(x)


def test_cl_mode_lift_formula() -> None:
    """AC3, AC6: ``L = q * S * (CL_baseline(q, m) + cl_residual)`` (full lift).

    Post-AXM-1739: the baseline is now mass-aware
    ``CL_baseline(q, m) = m·g/(q·S_REF)`` (using the state mass, not the
    NN), so the residual is the small correction around ``m·g``. With
    ``cl_residual=0.1`` and the real mass of the sample, the lift
    collapses to ``m·g + q·S·0.1``.
    """
    layer = PhysicsLayer()
    out = layer(_cl_inputs(cl_residual=0.1, q_pa=15_000.0, mass_kg=60_000.0))
    cl_base = float(cl_baseline(torch.tensor(15_000.0), torch.tensor(60_000.0)).item())
    expected = 15_000.0 * S_REF_A320_M2 * (cl_base + 0.1)
    # rel_tol=1e-5 to absorb the float32 ↔ float64 cast in the helper
    # (PhysicsLayer keeps full float64 internally).
    assert math.isclose(float(out["fdm_lift_N"].item()), expected, rel_tol=1e-5)


def test_cl_mode_d_tas_matches_newton_formula() -> None:
    """AC4: ``d_TAS = (t_minus_d_norm * m_ref) / m - g*sin(gamma)``."""
    layer = PhysicsLayer()
    out = layer(_cl_inputs(t_minus_d_norm=0.5, mass_kg=60_000.0, gamma_rad=0.0, cl_residual=0.0))
    expected = (0.5 * _M_REF_KG) / 60_000.0 - G * math.sin(0.0)
    assert math.isclose(float(out["fdm_d_tas_ms2"].item()), expected, rel_tol=1e-9)


def test_cl_mode_d_gamma_matches_newton_formula() -> None:
    """AC4: ``d_gamma = (L/m - g*cos(gamma)) / V_safe``.

    With ``cl_residual=0`` and the mass-aware baseline, the lift
    collapses to ``q·S·CL_baseline(q, m) = m·g`` -- Newton-equivalent
    (post-AXM-1739 fix).
    """
    layer = PhysicsLayer()
    out = layer(
        _cl_inputs(
            tas_ms=200.0,
            gamma_rad=0.0,
            mass_kg=60_000.0,
            cl_residual=0.0,
            q_pa=15_000.0,
        )
    )
    cl_base = float(cl_baseline(torch.tensor(15_000.0), torch.tensor(60_000.0)).item())
    lift = 15_000.0 * S_REF_A320_M2 * cl_base
    expected = (lift / 60_000.0 - G * math.cos(0.0)) / 200.0
    # rel_tol=1e-5 + abs_tol=1e-8 because the mass-aware baseline at
    # cl_residual=0 collapses d_gamma to exactly 0 in steady-level flight
    # (gamma=0, cl_residual=0, m=m → lift=m·g → d_gamma=0). float32 rounding
    # produces a tiny non-zero result on the "expected" side, against which
    # rel_tol alone fails (no relative scale when the target is 0).
    assert math.isclose(
        float(out["fdm_d_gamma_rads"].item()), expected, rel_tol=1e-5, abs_tol=1e-8
    )


def test_cl_mode_d_mass_is_zero_tensor() -> None:
    """AC1: ``d_mass`` is the zero tensor for the CL branch."""
    layer = PhysicsLayer()
    x = _cl_inputs(batch=4)
    out = layer(x)
    assert torch.equal(out["fdm_d_mass_kgs"], torch.zeros_like(x["fdm_mass_kg"]))


def test_cl_mode_low_tas_clamped_to_v_min() -> None:
    """AC4: TAS below ``V_MIN_CLAMP`` is clamped before the d_gamma divide."""
    layer = PhysicsLayer()
    low = layer(
        _cl_inputs(
            tas_ms=10.0,
            gamma_rad=0.0,
            mass_kg=60_000.0,
            cl_residual=0.0,
            q_pa=15_000.0,
        )
    )
    pinned = layer(
        _cl_inputs(
            tas_ms=V_MIN_CLAMP,
            gamma_rad=0.0,
            mass_kg=60_000.0,
            cl_residual=0.0,
            q_pa=15_000.0,
        )
    )
    assert math.isclose(
        float(low["fdm_d_gamma_rads"].item()),
        float(pinned["fdm_d_gamma_rads"].item()),
        rel_tol=1e-12,
    )


def test_cl_mode_with_phi_bank_emits_d_heading() -> None:
    """AC7: lateral channel ``d_heading = (G/V_safe)*tan(phi_bank)`` preserved."""
    layer = PhysicsLayer()
    out = layer(_cl_inputs(tas_ms=200.0, phi_bank_rad=0.5))
    expected = (G / 200.0) * math.tan(0.5)
    assert math.isclose(float(out["fdm_d_heading_rads"].item()), expected, rel_tol=1e-9)


def test_cl_mode_batch_shape_preserved() -> None:
    """AC1: all outputs preserve the batch shape ``(B,)``."""
    layer = PhysicsLayer()
    out = layer(_cl_inputs(batch=4))
    for key in (
        "fdm_d_tas_ms2",
        "fdm_d_gamma_rads",
        "fdm_d_mass_kgs",
        "fdm_lift_N",
    ):
        assert out[key].shape == (4,)
