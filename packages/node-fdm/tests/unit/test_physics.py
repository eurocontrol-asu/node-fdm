from __future__ import annotations

import torch

from node_fdm.layers.physics import PhysicsLayer


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
    """AC1, AC2: Newton mode computes d_tas from T-D, mass and gamma."""
    layer = PhysicsLayer()
    t_minus_d = torch.tensor(10000.0)
    mass = torch.tensor(65000.0)
    gamma = torch.tensor(0.05)

    out = layer(
        {
            "fdm_t_minus_d_N": t_minus_d,
            "fdm_lift_N": torch.tensor(600000.0),
            "fdm_mass_kg": mass,
            "era_tas_ms": torch.tensor(240.0),
            "fdm_gamma_rad": gamma,
        }
    )

    expected = t_minus_d / mass - 9.80665 * torch.sin(gamma)
    assert torch.allclose(out["fdm_d_tas_ms2"], expected, atol=1e-5)


def test_newton_mode_d_gamma_matches_formula() -> None:
    """AC1, AC2: Newton mode computes d_gamma from lift, mass, tas and gamma."""
    layer = PhysicsLayer()
    lift = torch.tensor(600000.0)
    mass = torch.tensor(65000.0)
    tas = torch.tensor(240.0)
    gamma = torch.tensor(0.05)

    out = layer(
        {
            "fdm_t_minus_d_N": torch.tensor(10000.0),
            "fdm_lift_N": lift,
            "fdm_mass_kg": mass,
            "era_tas_ms": tas,
            "fdm_gamma_rad": gamma,
        }
    )

    expected = (lift / mass - 9.80665 * torch.cos(gamma)) / torch.clamp(tas, min=50.0)
    assert torch.allclose(out["fdm_d_gamma_rads"], expected, atol=1e-5)


def test_newton_mode_d_mass_is_zero_tensor() -> None:
    """AC3: Newton mode emits a fresh zero tensor matching mass shape and device."""
    layer = PhysicsLayer()
    mass = torch.tensor([65000.0, 66000.0])

    out = layer(
        {
            "fdm_t_minus_d_N": torch.tensor([10000.0, 11000.0]),
            "fdm_lift_N": torch.tensor([600000.0, 610000.0]),
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


def test_newton_mode_with_phi_bank() -> None:
    """AC4: Newton mode keeps the lateral coordinated-turn branch working."""
    layer = PhysicsLayer()
    tas = torch.tensor(240.0)
    phi_bank = torch.tensor(0.3)

    out = layer(
        {
            "fdm_t_minus_d_N": torch.tensor(10000.0),
            "fdm_lift_N": torch.tensor(600000.0),
            "fdm_mass_kg": torch.tensor(65000.0),
            "era_tas_ms": tas,
            "fdm_gamma_rad": torch.tensor(0.05),
            "fdm_phi_bank_rad": phi_bank,
        }
    )

    expected = 9.80665 / tas * torch.tan(phi_bank)
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

    expected = 9.80665 / tas * torch.tan(phi_bank)
    assert torch.allclose(out["fdm_d_heading_rads"], expected, atol=1e-5)


def test_newton_mode_low_tas_clamped() -> None:
    """AC6: Newton mode reuses the 50 m/s TAS clamp for d_gamma."""
    layer = PhysicsLayer()
    lift = torch.tensor(600000.0)
    mass = torch.tensor(65000.0)
    gamma = torch.tensor(0.05)

    out = layer(
        {
            "fdm_t_minus_d_N": torch.tensor(10000.0),
            "fdm_lift_N": lift,
            "fdm_mass_kg": mass,
            "era_tas_ms": torch.tensor(10.0),
            "fdm_gamma_rad": gamma,
        }
    )

    expected = (lift / mass - 9.80665 * torch.cos(gamma)) / torch.tensor(50.0)
    assert torch.isfinite(out["fdm_d_gamma_rads"])
    assert torch.allclose(out["fdm_d_gamma_rads"], expected, atol=1e-5)


def test_newton_mode_batch_shape() -> None:
    """AC2: Newton mode preserves batched tensor shapes for all emitted outputs."""
    layer = PhysicsLayer()
    batch_size = 8

    out = layer(
        {
            "fdm_t_minus_d_N": torch.full((batch_size,), 10000.0),
            "fdm_lift_N": torch.full((batch_size,), 600000.0),
            "fdm_mass_kg": torch.full((batch_size,), 65000.0),
            "era_tas_ms": torch.full((batch_size,), 240.0),
            "fdm_gamma_rad": torch.full((batch_size,), 0.05),
        }
    )

    assert out["fdm_d_tas_ms2"].shape == (batch_size,)
    assert out["fdm_d_gamma_rads"].shape == (batch_size,)
    assert out["fdm_d_mass_kgs"].shape == (batch_size,)
