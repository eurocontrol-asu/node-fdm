"""Tests for GammaDefaultNet and TrajectoryLayer gamma integration."""

from __future__ import annotations

import torch

from node_fdm.layers.blocks import GammaDefaultNet
from node_fdm.layers.trajectory import TrajectoryLayer

# ---------------------------------------------------------------------------
# Unit tests — GammaDefaultNet
# ---------------------------------------------------------------------------


class TestGammaDefaultNetOutput:
    """GammaDefaultNet output shape and value tests."""

    def test_output_shape(self) -> None:
        """GammaDefaultNet()(alt, gamma, tas) with batch=4 → shape (4,)."""
        net = GammaDefaultNet()
        alt = torch.randn(4)
        gamma = torch.randn(4)
        tas = torch.randn(4)
        out = net(alt, gamma, tas)
        assert out.shape == (4,)

    def test_zero_init_output(self) -> None:
        """Fresh GammaDefaultNet with random inputs → all outputs ≈ 0.0."""
        net = GammaDefaultNet()
        alt = torch.randn(8)
        gamma = torch.randn(8)
        tas = torch.randn(8)
        out = net(alt, gamma, tas)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)

    def test_gradient_flow(self) -> None:
        """Forward + .sum().backward() → alt.grad is not None."""
        net = GammaDefaultNet()
        alt = torch.randn(4, requires_grad=True)
        gamma = torch.randn(4)
        tas = torch.randn(4)
        out = net(alt, gamma, tas)
        out.sum().backward()
        assert alt.grad is not None

    def test_buffer_in_state_dict(self) -> None:
        """state_dict() contains '_scale' key."""
        net = GammaDefaultNet()
        sd = net.state_dict()
        assert "_scale" in sd

    def test_device_transfer(self) -> None:
        """.to('cpu') after init → _scale.device == cpu."""
        net = GammaDefaultNet()
        net = net.to("cpu")
        assert net._scale.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Functional tests — TrajectoryLayer integration
# ---------------------------------------------------------------------------


def _make_trajectory_layer() -> TrajectoryLayer:
    """Create a TrajectoryLayer with gamma_sel and gamma_known in col_map."""
    return TrajectoryLayer(
        col_map={
            "gamma_sel": "gamma_sel_rad",
            "gamma_known": "gamma_known",
            "gamma_diff": "gamma_diff_rad",
        },
    )


def _base_inputs(batch: int = 4) -> dict[str, torch.Tensor]:
    """Minimal input dict for TrajectoryLayer forward."""
    return {
        "tas_ms": torch.full((batch,), 200.0),
        "gamma_rad": torch.full((batch,), 0.05),
        "altitude_m": torch.full((batch,), 5000.0),
        "long_wind_ms": torch.zeros(batch),
    }


class TestTrajectoryLayerGammaIntegration:
    """Functional tests for gamma_diff with GammaDefaultNet."""

    def test_gamma_diff_unknown_uses_net(self) -> None:
        """known=0, fresh layer → gamma_diff ≈ 0.0 - gamma (zero-init)."""
        layer = _make_trajectory_layer()
        inputs = _base_inputs(4)
        inputs["gamma_sel_rad"] = torch.full((4,), 0.1)
        inputs["gamma_known"] = torch.zeros(4)
        out = layer(inputs)
        gamma = inputs["gamma_rad"]
        # Zero-init net outputs ~0, so gamma_diff ≈ 0.0 - gamma
        expected = torch.zeros(4) - gamma
        assert torch.allclose(out["gamma_diff_rad"], expected, atol=1e-5)

    def test_gamma_diff_known_bypasses_net(self) -> None:
        """known=1, modified net weights → gamma_diff = target - gamma (net ignored)."""
        layer = _make_trajectory_layer()
        # Perturb net weights so its output is non-zero
        with torch.no_grad():
            for p in layer.gamma_default_net.parameters():
                p.fill_(1.0)
        inputs = _base_inputs(4)
        target = torch.full((4,), 0.1)
        inputs["gamma_sel_rad"] = target
        inputs["gamma_known"] = torch.ones(4)
        out = layer(inputs)
        gamma = inputs["gamma_rad"]
        expected = target - gamma
        assert torch.allclose(out["gamma_diff_rad"], expected, atol=1e-5)

    def test_gamma_diff_mixed(self) -> None:
        """known=[0,1] → idx0 uses net, idx1 uses target."""
        layer = _make_trajectory_layer()
        inputs = _base_inputs(2)
        target = torch.tensor([0.1, 0.2])
        inputs["gamma_sel_rad"] = target
        inputs["gamma_known"] = torch.tensor([0.0, 1.0])
        out = layer(inputs)
        gamma = inputs["gamma_rad"]
        diff = out["gamma_diff_rad"]
        # idx0: unknown → net(≈0) - gamma
        assert torch.allclose(diff[0:1], (torch.zeros(1) - gamma[0:1]), atol=1e-5)
        # idx1: known → target - gamma
        assert torch.allclose(diff[1:2], (target[1:2] - gamma[1:2]), atol=1e-5)

    def test_net_is_trainable(self) -> None:
        """layer.gamma_default_net.parameters() → count > 0, all requires_grad."""
        layer = _make_trajectory_layer()
        params = list(layer.gamma_default_net.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGammaDefaultNetEdgeCases:
    """Edge-case tests for GammaDefaultNet / TrajectoryLayer gamma path."""

    def test_all_known(self) -> None:
        """gamma_known all 1.0 → net output multiplied by 0, zero gradient to net."""
        layer = _make_trajectory_layer()
        inputs = _base_inputs(4)
        inputs["gamma_sel_rad"] = torch.full((4,), 0.1)
        inputs["gamma_known"] = torch.ones(4)
        # Enable grad tracking on net params
        for p in layer.gamma_default_net.parameters():
            p.requires_grad_(True)
        out = layer(inputs)
        out["gamma_diff_rad"].sum().backward()
        # Net params should have zero gradient (multiplied by 0)
        for p in layer.gamma_default_net.parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad, torch.zeros_like(p.grad), atol=1e-7)

    def test_all_unknown(self) -> None:
        """gamma_known all 0.0 → entire gamma_diff from net."""
        layer = _make_trajectory_layer()
        inputs = _base_inputs(4)
        inputs["gamma_sel_rad"] = torch.full((4,), 0.1)
        inputs["gamma_known"] = torch.zeros(4)
        out = layer(inputs)
        gamma = inputs["gamma_rad"]
        # Zero-init net → gamma_diff ≈ 0.0 - gamma
        expected = torch.zeros(4) - gamma
        assert torch.allclose(out["gamma_diff_rad"], expected, atol=1e-5)

    def test_extreme_altitude(self) -> None:
        """alt=15000 m → no NaN/Inf (clamped by normalization)."""
        net = GammaDefaultNet()
        alt = torch.full((4,), 15000.0)
        gamma = torch.randn(4)
        tas = torch.randn(4)
        out = net(alt, gamma, tas)
        assert torch.isfinite(out).all()

    def test_zero_tas(self) -> None:
        """tas=0 → no division by zero (multiplicative scaling only)."""
        net = GammaDefaultNet()
        alt = torch.randn(4)
        gamma = torch.randn(4)
        tas = torch.zeros(4)
        out = net(alt, gamma, tas)
        assert torch.isfinite(out).all()

    def test_no_gamma_sel_column(self) -> None:
        """gamma_sel not in input dict → gamma_diff not computed, net never called."""
        layer = _make_trajectory_layer()
        inputs = _base_inputs(4)
        # No gamma_sel_rad in inputs
        out = layer(inputs)
        assert "gamma_diff_rad" not in out
