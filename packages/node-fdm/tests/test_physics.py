"""Tests for the analytical PhysicsLayer.

The layer converts NN outputs ``(fdm_a_spec_ms2, fdm_n_z_residual)`` into
the ODE derivatives ``(fdm_d_tas_ms2, fdm_d_gamma_rads)`` by applying
gravity analytically:

    d_tas   = a_spec - g * sin(gamma)
    d_gamma = (g / max(tas, V_MIN)) * ((n_z_residual + 1) - cos(gamma))

The NN emits ``n_z_residual = n_z - 1`` (centered on zero) so that
symmetric denormalize modes (``"scaled"``, ``"normal_clamp"``) work
correctly; PhysicsLayer adds 1 back inside.
"""

from __future__ import annotations

import math

import torch

from node_fdm.layers.physics import V_MIN_CLAMP, G, PhysicsLayer


def _make_inputs(
    a_spec: float,
    n_z_residual: float,
    tas: float,
    gamma: float,
    batch: int = 4,
) -> dict[str, torch.Tensor]:
    """Build a vect_dict with constant scalars broadcast to ``(batch,)``.

    ``n_z_residual`` is what the NN emits (n_z - 1).
    """
    return {
        "fdm_a_spec_ms2": torch.full((batch,), a_spec, dtype=torch.float32),
        "fdm_n_z_residual": torch.full((batch,), n_z_residual, dtype=torch.float32),
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
        """Forward returns the two ODE derivative columns plus reconstructed n_z."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=0.0, tas=200.0, gamma=0.0))
        assert set(out.keys()) == {"fdm_d_tas_ms2", "fdm_d_gamma_rads", "fdm_n_z"}

    def test_n_z_reconstructed(self) -> None:
        """fdm_n_z = fdm_n_z_residual + 1 in the output dict."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=0.3, tas=200.0, gamma=0.0))
        assert torch.allclose(out["fdm_n_z"], torch.full((4,), 1.3), atol=1e-6)

    def test_batch_shape_preserved(self) -> None:
        """Output tensors keep the batch dimension of inputs."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=0.0, tas=200.0, gamma=0.0, batch=32))
        assert out["fdm_d_tas_ms2"].shape == (32,)
        assert out["fdm_d_gamma_rads"].shape == (32,)


class TestPhysicsLayerEquations:
    """Numerical correctness of the analytical equations."""

    def test_level_cruise_equilibrium(self) -> None:
        """gamma=0, n_z_residual=0 (so n_z=1), a_spec=0 → both derivatives are zero."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=0.0, tas=200.0, gamma=0.0))
        assert torch.allclose(out["fdm_d_tas_ms2"], torch.zeros(4), atol=1e-6)
        assert torch.allclose(out["fdm_d_gamma_rads"], torch.zeros(4), atol=1e-6)

    def test_pure_gravity_descent(self) -> None:
        """Balanced descent (n_z=cos gamma so n_z_residual=cos gamma - 1, a_spec=0).

        Only gravity accelerates V; gamma is steady.
        """
        gamma = -0.1
        n_z_residual = math.cos(gamma) - 1.0
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=n_z_residual, tas=200.0, gamma=gamma))
        expected_d_tas = -G * math.sin(gamma)
        assert torch.allclose(
            out["fdm_d_tas_ms2"],
            torch.full((4,), expected_d_tas),
            atol=1e-5,
        )
        assert torch.allclose(out["fdm_d_gamma_rads"], torch.zeros(4), atol=1e-6)

    def test_pure_load_factor_pull_up(self) -> None:
        """gamma=0, n_z=2 (residual=1) → d_gamma = (g/V) * 1 > 0."""
        layer = PhysicsLayer()
        tas = 200.0
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=1.0, tas=tas, gamma=0.0))
        expected_d_gamma = (G / tas) * 1.0
        assert torch.allclose(
            out["fdm_d_gamma_rads"],
            torch.full((4,), expected_d_gamma),
            atol=1e-6,
        )

    def test_a_spec_passes_through_at_level(self) -> None:
        """At gamma=0, d_tas equals a_spec exactly (no gravity projection)."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=1.5, n_z_residual=0.0, tas=200.0, gamma=0.0))
        assert torch.allclose(out["fdm_d_tas_ms2"], torch.full((4,), 1.5), atol=1e-6)


class TestPhysicsLayerEdgeCases:
    """Numerical stability under degenerate inputs."""

    def test_low_tas_clamped(self) -> None:
        """tas below stall must not produce NaN/Inf in d_gamma (clamp protects 1/V).

        Important: V_MIN_CLAMP must be set to a realistic stall speed (~50 m/s)
        rather than a tiny value, otherwise an intermediate ODE substep that
        produces a corrupted TAS (negative or near-zero) sees ``g/V`` blow up
        far above the dx_bounds, triggering a NaN cascade.
        """
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=1.0, tas=0.0, gamma=0.0))
        assert torch.isfinite(out["fdm_d_gamma_rads"]).all()
        assert torch.isfinite(out["fdm_d_tas_ms2"]).all()
        expected = G / V_MIN_CLAMP
        assert torch.allclose(
            out["fdm_d_gamma_rads"],
            torch.full((4,), expected),
            atol=1e-4,
        )

    def test_negative_tas_clamped(self) -> None:
        """Corrupted negative TAS (from RK4 intermediate state) is clamped to V_MIN."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=1.0, tas=-13.5, gamma=0.0))
        assert torch.isfinite(out["fdm_d_gamma_rads"]).all()
        # With tas clamped to V_MIN_CLAMP=50, d_gamma stays in safe range
        assert (out["fdm_d_gamma_rads"].abs() <= G / V_MIN_CLAMP + 1e-4).all()

    def test_negative_n_z(self) -> None:
        """Inverted flight (n_z = -0.5, so residual = -1.5) → finite negative d_gamma."""
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=-1.5, tas=200.0, gamma=0.0))
        assert torch.isfinite(out["fdm_d_gamma_rads"]).all()
        assert (out["fdm_d_gamma_rads"] < 0).all()

    def test_extreme_gamma(self) -> None:
        """Steep climb (gamma=0.5 rad ≈ 28°) keeps outputs finite and signed correctly."""
        gamma = 0.5
        layer = PhysicsLayer()
        out = layer(_make_inputs(a_spec=0.0, n_z_residual=0.0, tas=200.0, gamma=gamma))
        assert torch.isfinite(out["fdm_d_tas_ms2"]).all()
        assert torch.isfinite(out["fdm_d_gamma_rads"]).all()
        # d_tas = -g sin(gamma) < 0 in steep climb
        assert (out["fdm_d_tas_ms2"] < 0).all()


class TestPhysicsLayerInitBias:
    """Init-time equilibrium: NN heads zero-init must produce d_tas=d_gamma=0
    at cruise, because n_z_residual=0 → n_z=1 (lift balances gravity).
    """

    def test_fresh_adsb_model_is_at_cruise_equilibrium(self) -> None:
        """A freshly-built NODE_ADSB_V1 produces d_tas≈0 and d_gamma≈0 at gamma=0."""
        from node_fdm.architectures.adsb import NODE_ADSB_V1
        from node_fdm.models.fdm import FlightDynamicsModel

        all_cols = (
            NODE_ADSB_V1.x_cols
            + NODE_ADSB_V1.u_cols
            + NODE_ADSB_V1.e0_cols
            + NODE_ADSB_V1.e1_cols
            + [c for _, c in NODE_ADSB_V1.dx_cols]
            + ["fdm_a_spec_ms2", "fdm_n_z_residual", "fdm_phi_bank_rad"]
        )
        stats = {c: {"mean": 0.0, "std": 1.0, "max": 1.0, "p999": 1.0} for c in all_cols}
        model = FlightDynamicsModel(spec=NODE_ADSB_V1, stats_dict=stats, model_params=(2, 1, 32))

        batch = 8
        n_x = len(NODE_ADSB_V1.x_cols)
        n_u = len(NODE_ADSB_V1.u_cols)
        n_e = len(NODE_ADSB_V1.e0_cols)
        x = torch.zeros(batch, n_x)
        x[:, NODE_ADSB_V1.x_cols.index("raw_alt_m")] = 5000.0
        x[:, NODE_ADSB_V1.x_cols.index("era_tas_ms")] = 200.0
        u = torch.zeros(batch, n_u)
        e = torch.zeros(batch, n_e)

        with torch.no_grad():
            dx = model(x, u, e)

        dx_names = [c for _, c in NODE_ADSB_V1.dx_cols]
        i_d_gamma = dx_names.index("fdm_d_gamma_rads")
        i_d_tas = dx_names.index("fdm_d_tas_ms2")
        assert torch.allclose(dx[:, i_d_gamma], torch.zeros(batch), atol=1e-6)
        assert torch.allclose(dx[:, i_d_tas], torch.zeros(batch), atol=1e-6)
