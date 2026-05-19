"""Integration tests for CL-mode ≡ Newton-mode equivalence (AXM-1737).

Cross-mode algebraic identity: a CL-mode batch built from the inverse
residual produces the same ``d_tas`` / ``d_gamma`` as the equivalent
Newton-mode batch. Also pins the additive-only contract on existing
physics tests (AC9).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from node_fdm.layers.physics import (
    _M_REF_KG,
    CL_REF,
    S_REF_A320_M2,
    G,
    PhysicsLayer,
)

pytestmark = pytest.mark.integration


def _t(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_cl_mode_equivalent_to_newton_under_algebraic_identity() -> None:
    """AC5: CL-mode equivalent to Newton-mode under cl_residual identity.

    Build two batches with the same per-sample lift target ``L_target``
    and identical longitudinal inputs. The Newton branch reaches
    ``L_target`` via ``lift_residual_norm``; the CL branch reaches it via
    ``cl_residual``. ``d_tas`` and ``d_gamma`` must agree within ``1e-5``
    relative tolerance on a non-degenerate batch.
    """
    t_minus_d_norm = _t([0.3, -0.2, 0.5, 0.1])
    mass = _t([55_000.0, 65_000.0, 60_000.0, 72_000.0])
    tas = _t([180.0, 220.0, 200.0, 240.0])
    gamma = _t([0.0, 0.05, -0.02, 0.03])
    q_pa = _t([12_000.0, 16_000.0, 14_500.0, 18_000.0])

    lift_residual_norm = _t([0.05, -0.03, 0.02, 0.0])
    l_target = lift_residual_norm * _M_REF_KG * G + mass * G
    cl_residual = (l_target - q_pa * S_REF_A320_M2 * CL_REF) / (q_pa * S_REF_A320_M2)

    layer = PhysicsLayer()

    out_newton = layer(
        {
            "fdm_t_minus_d_norm": t_minus_d_norm,
            "fdm_lift_residual_norm": lift_residual_norm,
            "fdm_mass_kg": mass,
            "era_tas_ms": tas,
            "fdm_gamma_rad": gamma,
        }
    )
    out_cl = layer(
        {
            "fdm_t_minus_d_norm": t_minus_d_norm,
            "fdm_cl_residual": cl_residual,
            "fdm_mass_kg": mass,
            "era_tas_ms": tas,
            "fdm_gamma_rad": gamma,
            "fdm_q_pa": q_pa,
        }
    )

    torch.testing.assert_close(
        out_cl["fdm_d_tas_ms2"], out_newton["fdm_d_tas_ms2"], rtol=1e-5, atol=1e-9
    )
    torch.testing.assert_close(
        out_cl["fdm_d_gamma_rads"],
        out_newton["fdm_d_gamma_rads"],
        rtol=1e-5,
        atol=1e-9,
    )


def test_cl_mode_does_not_regress_existing_physics_tests() -> None:
    """AC9: pre-existing legacy/Newton test functions in ``test_physics.py`` survive.

    Snapshot guard against accidental deletion of existing
    ``test_legacy_mode_*`` and ``test_newton_mode_*`` test cases during
    the CL-branch addition. The ticket explicitly states the new branch
    is purely additive.
    """
    test_file = Path(__file__).resolve().parents[1] / "unit" / "test_physics.py"
    text = test_file.read_text(encoding="utf-8")
    assert "def test_legacy_mode_unchanged" in text
    assert "def test_newton_mode_d_tas_matches_formula" in text
    assert "def test_newton_mode_d_gamma_matches_formula" in text
    assert "def test_newton_mode_d_mass_is_zero_tensor" in text
    assert "def test_newton_mode_exposes_full_lift" in text
    assert "def test_newton_mode_with_phi_bank" in text
    assert "def test_legacy_mode_with_phi_bank_still_works" in text
    assert "def test_newton_mode_low_tas_clamped" in text
    assert "def test_newton_mode_batch_shape" in text
