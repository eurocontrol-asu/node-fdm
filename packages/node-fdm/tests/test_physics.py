"""Tests for the analytical PhysicsLayer.

The layer converts NN outputs ``(fdm_a_spec_ms2, fdm_n_z)`` into the ODE
derivatives ``(fdm_d_tas_ms2, fdm_d_gamma_rads)`` by applying gravity
analytically:

    d_tas   = a_spec - g * sin(gamma)
    d_gamma = (g / max(tas, V_MIN)) * (n_z - cos(gamma))
"""

from __future__ import annotations

import math

import torch

from node_fdm.layers.physics import V_MIN_CLAMP, G, PhysicsLayer


def _make_inputs(
    a_spec: float,
    n_z: float,
    tas: float,
    gamma: float,
    batch: int = 4,
) -> dict[str, torch.Tensor]:
    """Build a vect_dict with constant scalars broadcast to ``(batch,)``."""
    return {
        "fdm_a_spec_ms2": torch.full((batch,), a_spec, dtype=torch.float32),
        "fdm_n_z": torch.full((batch,), n_z, dtype=torch.float32),
        "era_tas_ms": torch.full((batch,), tas, dtype=torch.float32),
        "fdm_gamma_rad": torch.full((batch,), gamma, dtype=torch.float32),
    }


class TestPhysicsLayerStructure:
    """Structural and signature contracts."""

    def test_no_trainable_parameters(self) -> None:
        """Layer has no learnable parameters."""
        layer = PhysicsLayer()
        assert sum(p.numel() for p in layer.parameters()) == 0

    def test_input_stats_accepted(self) -> None:
        """Constructor accepts ``input_stats`` as a no-op kwarg (pipeline contract)."""
        stats = {"era_tas_ms": {"mean": 200.0, "std": 30.0}}
        layer = PhysicsLayer(input_stats=stats)
        assert layer is not None

    def test_output_keys(self) -> None:
        """Forward returns exactly the two ODE derivative columns."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z=1.0, tas=200.0, gamma=0.0))
        assert set(out.keys()) == {"fdm_d_tas_ms2", "fdm_d_gamma_rads"}

    def test_batch_shape_preserved(self) -> None:
        """Output tensors keep the batch dimension of inputs."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z=1.0, tas=200.0, gamma=0.0, batch=32))
        assert out["fdm_d_tas_ms2"].shape == (32,)
        assert out["fdm_d_gamma_rads"].shape == (32,)


class TestPhysicsLayerEquations:
    """Numerical correctness of the analytical equations."""

    def test_level_cruise_equilibrium(self) -> None:
        """gamma=0, n_z=1, a_spec=0 → both derivatives are zero (level cruise)."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z=1.0, tas=200.0, gamma=0.0))
        assert torch.allclose(out["fdm_d_tas_ms2"], torch.zeros(4), atol=1e-6)
        assert torch.allclose(out["fdm_d_gamma_rads"], torch.zeros(4), atol=1e-6)

    def test_pure_gravity_descent(self) -> None:
        """In a balanced descent (n_z=cos(gamma), a_spec=0), only gravity accelerates V."""
        gamma = -0.1
        n_z_balanced = math.cos(gamma)
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z=n_z_balanced, tas=200.0, gamma=gamma))
        # d_tas = -g sin(gamma) > 0 in descent
        expected_d_tas = -G * math.sin(gamma)
        assert torch.allclose(
            out["fdm_d_tas_ms2"],
            torch.full((4,), expected_d_tas),
            atol=1e-5,
        )
        assert torch.allclose(out["fdm_d_gamma_rads"], torch.zeros(4), atol=1e-6)

    def test_pure_load_factor_pull_up(self) -> None:
        """gamma=0, n_z=2 → d_gamma = (g/V) * (n_z - 1) > 0 (pull-up)."""
        layer = PhysicsLayer()
        tas = 200.0
        out = layer(_make_inputs(a_spec=0.0, n_z=2.0, tas=tas, gamma=0.0))
        expected_d_gamma = (G / tas) * (2.0 - 1.0)
        assert torch.allclose(
            out["fdm_d_gamma_rads"],
            torch.full((4,), expected_d_gamma),
            atol=1e-6,
        )

    def test_a_spec_passes_through_at_level(self) -> None:
        """At gamma=0, d_tas equals a_spec exactly (no gravity projection)."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=1.5, n_z=1.0, tas=200.0, gamma=0.0))
        assert torch.allclose(out["fdm_d_tas_ms2"], torch.full((4,), 1.5), atol=1e-6)


class TestPhysicsLayerEdgeCases:
    """Numerical stability under degenerate inputs."""

    def test_low_tas_clamped(self) -> None:
        """tas=0 must not produce NaN/Inf in d_gamma (clamp protects 1/V)."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z=2.0, tas=0.0, gamma=0.0))
        assert torch.isfinite(out["fdm_d_gamma_rads"]).all()
        assert torch.isfinite(out["fdm_d_tas_ms2"]).all()
        # With tas clamped to V_MIN_CLAMP, d_gamma = g / V_MIN_CLAMP * (n_z - 1)
        expected = (G / V_MIN_CLAMP) * (2.0 - 1.0)
        assert torch.allclose(
            out["fdm_d_gamma_rads"],
            torch.full((4,), expected),
            atol=1e-4,
        )

    def test_negative_n_z(self) -> None:
        """Inverted flight (n_z < 0) produces a finite negative d_gamma."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z=-0.5, tas=200.0, gamma=0.0))
        assert torch.isfinite(out["fdm_d_gamma_rads"]).all()
        assert (out["fdm_d_gamma_rads"] < 0).all()

    def test_extreme_gamma(self) -> None:
        """Steep climb (gamma=0.5 rad ≈ 28°) keeps outputs finite and signed correctly."""
        gamma = 0.5
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z=1.0, tas=200.0, gamma=gamma))
        assert torch.isfinite(out["fdm_d_tas_ms2"]).all()
        assert torch.isfinite(out["fdm_d_gamma_rads"]).all()
        # d_tas = -g sin(gamma) < 0 in steep climb
        assert (out["fdm_d_tas_ms2"] < 0).all()
