"""Tests for TrajectoryLayer gamma_diff computation.

After removing GammaDefaultNet, gamma_diff is simply:
    known * (target - gamma)
When known=0, gamma_diff=0 (no correction signal).
"""

from __future__ import annotations

import torch

from node_fdm.layers.trajectory import TrajectoryLayer

# ---------------------------------------------------------------------------
# Helpers
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
        "alt_sel_m": torch.full((batch,), 6000.0),
    }


# ---------------------------------------------------------------------------
# Tests — gamma_diff without GammaDefaultNet
# ---------------------------------------------------------------------------


class TestGammaDiffKnown:
    """When known=1, gamma_diff = target - gamma."""

    def test_gamma_diff_all_known(self) -> None:
        layer = _make_trajectory_layer()
        inputs = _base_inputs(4)
        inputs["gamma_sel_rad"] = torch.full((4,), 0.1)
        inputs["gamma_known"] = torch.ones(4)
        out = layer(inputs)
        expected = torch.full((4,), 0.1) - inputs["gamma_rad"]
        assert torch.allclose(out["gamma_diff_rad"], expected, atol=1e-6)


class TestGammaDiffUnknown:
    """When known=0, gamma_diff = 0."""

    def test_gamma_diff_all_unknown(self) -> None:
        layer = _make_trajectory_layer()
        inputs = _base_inputs(4)
        inputs["gamma_sel_rad"] = torch.full((4,), 0.1)
        inputs["gamma_known"] = torch.zeros(4)
        out = layer(inputs)
        assert torch.allclose(out["gamma_diff_rad"], torch.zeros(4), atol=1e-6)


class TestGammaDiffMixed:
    """Mixed known/unknown produces correct values per element."""

    def test_gamma_diff_mixed(self) -> None:
        layer = _make_trajectory_layer()
        inputs = _base_inputs(2)
        target = torch.tensor([0.1, 0.2])
        inputs["gamma_sel_rad"] = target
        inputs["gamma_known"] = torch.tensor([0.0, 1.0])
        out = layer(inputs)
        gamma = inputs["gamma_rad"]
        diff = out["gamma_diff_rad"]
        # idx0: unknown → 0
        assert torch.allclose(diff[0:1], torch.zeros(1), atol=1e-6)
        # idx1: known → target - gamma
        assert torch.allclose(diff[1:2], target[1:2] - gamma[1:2], atol=1e-6)


class TestGammaKnownPassthrough:
    """gamma_known flag is passed through to output for StructuredLayer."""

    def test_gamma_known_in_output(self) -> None:
        layer = _make_trajectory_layer()
        inputs = _base_inputs(4)
        inputs["gamma_sel_rad"] = torch.full((4,), 0.1)
        inputs["gamma_known"] = torch.tensor([0.0, 1.0, 0.0, 1.0])
        out = layer(inputs)
        assert "gamma_known" in out
        assert torch.equal(out["gamma_known"], inputs["gamma_known"])


class TestGammaDiffEdgeCases:
    """Edge cases for the simplified gamma_diff path."""

    def test_no_gamma_sel_column(self) -> None:
        """gamma_sel not in input dict → gamma_diff not computed."""
        layer = _make_trajectory_layer()
        inputs = _base_inputs(4)
        out = layer(inputs)
        assert "gamma_diff_rad" not in out

    def test_no_gamma_known_column(self) -> None:
        """gamma_sel present but no gamma_known → fallback: diff = target - gamma."""
        layer = TrajectoryLayer(
            col_map={"gamma_sel": "gamma_sel_rad", "gamma_diff": "gamma_diff_rad"},
        )
        inputs = _base_inputs(4)
        inputs["gamma_sel_rad"] = torch.full((4,), 0.1)
        out = layer(inputs)
        expected = torch.full((4,), 0.1) - inputs["gamma_rad"]
        assert torch.allclose(out["gamma_diff_rad"], expected, atol=1e-6)

    def test_no_trainable_parameters(self) -> None:
        """TrajectoryLayer has no trainable parameters (GammaNet removed)."""
        layer = _make_trajectory_layer()
        params = list(layer.parameters())
        assert len(params) == 0
