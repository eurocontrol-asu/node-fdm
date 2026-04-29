"""Tests for TrajectoryLayer g_sin_gamma and cos_gamma kinematic features.

These features expose the gravity-compensation terms analytically so the
NN does not need to re-learn them from data.
"""

from __future__ import annotations

import math

import torch

from node_fdm.layers.trajectory import TrajectoryLayer


def _base_inputs(
    gamma_val: float,
    tas_val: float = 250.0,
    alt_val: float = 5000.0,
) -> dict[str, torch.Tensor]:
    """Minimal input dict for TrajectoryLayer with default col_map."""
    return {
        "tas_ms": torch.tensor([tas_val]),
        "gamma_rad": torch.tensor([gamma_val]),
        "altitude_m": torch.tensor([alt_val]),
    }


class TestTrajectoryKinematicFeaturesPresent:
    """g_sin_gamma and cos_gamma always appear in the output dict."""

    def test_g_sin_gamma_in_output(self) -> None:
        """g_sin_gamma key exists in TrajectoryLayer output."""
        layer = TrajectoryLayer()
        out = layer(_base_inputs(0.0))
        assert "fdm_g_sin_gamma_ms2" in out

    def test_cos_gamma_in_output(self) -> None:
        """cos_gamma key exists in TrajectoryLayer output."""
        layer = TrajectoryLayer()
        out = layer(_base_inputs(0.0))
        assert "fdm_cos_gamma" in out


class TestTrajectoryKinematicFeaturesValues:
    """Numeric correctness of g_sin_gamma and cos_gamma."""

    G = 9.80665

    def test_level_flight_g_sin_gamma_is_zero(self) -> None:
        """At gamma=0 (level flight), g*sin(0)=0."""
        layer = TrajectoryLayer()
        out = layer(_base_inputs(0.0))
        assert torch.isclose(out["fdm_g_sin_gamma_ms2"], torch.tensor(0.0), atol=1e-6)

    def test_level_flight_cos_gamma_is_one(self) -> None:
        """At gamma=0 (level flight), cos(0)=1."""
        layer = TrajectoryLayer()
        out = layer(_base_inputs(0.0))
        assert torch.isclose(out["fdm_cos_gamma"], torch.tensor(1.0), atol=1e-6)

    def test_g_sin_gamma_at_01_rad(self) -> None:
        """At gamma=0.1 rad, g_sin_gamma ≈ G * sin(0.1)."""
        layer = TrajectoryLayer()
        out = layer(_base_inputs(0.1))
        expected = torch.tensor(self.G * math.sin(0.1))
        assert torch.isclose(out["fdm_g_sin_gamma_ms2"], expected, atol=1e-4)

    def test_cos_gamma_at_01_rad(self) -> None:
        """At gamma=0.1 rad, cos_gamma ≈ cos(0.1)."""
        layer = TrajectoryLayer()
        out = layer(_base_inputs(0.1))
        expected = torch.tensor(math.cos(0.1))
        assert torch.isclose(out["fdm_cos_gamma"], expected, atol=1e-6)

    def test_g_sin_gamma_negative_gamma(self) -> None:
        """Descent (gamma < 0) yields negative g_sin_gamma."""
        layer = TrajectoryLayer()
        out = layer(_base_inputs(-0.1))
        expected = torch.tensor(self.G * math.sin(-0.1))
        assert torch.isclose(out["fdm_g_sin_gamma_ms2"], expected, atol=1e-4)

    def test_cos_gamma_negative_gamma(self) -> None:
        """cos_gamma is symmetric: cos(-gamma) == cos(gamma)."""
        layer = TrajectoryLayer()
        out_pos = layer(_base_inputs(0.15))
        out_neg = layer(_base_inputs(-0.15))
        assert torch.isclose(out_pos["fdm_cos_gamma"], out_neg["fdm_cos_gamma"], atol=1e-6)

    def test_batch_shape_preserved(self) -> None:
        """Batch dimension is preserved for kinematic features."""
        layer = TrajectoryLayer()
        x = {
            "tas_ms": torch.tensor([250.0, 300.0, 150.0]),
            "gamma_rad": torch.tensor([0.0, 0.1, -0.05]),
            "altitude_m": torch.tensor([5000.0, 10000.0, 2000.0]),
        }
        out = layer(x)
        assert out["fdm_g_sin_gamma_ms2"].shape == (3,)
        assert out["fdm_cos_gamma"].shape == (3,)


class TestTrajectoryKinematicFeaturesCustomColMap:
    """Custom col_map keys are respected for kinematic feature output names."""

    def test_custom_col_map_renames_g_sin_gamma(self) -> None:
        """Custom g_sin_gamma key is used in output dict."""
        col_map = {
            "tas": "era_tas_ms",
            "gamma": "fdm_gamma_rad",
            "alt": "raw_alt_m",
            "g_sin_gamma": "fdm_g_sin_gamma_ms2",
            "cos_gamma": "fdm_cos_gamma",
        }
        layer = TrajectoryLayer(col_map=col_map)
        x = {
            "era_tas_ms": torch.tensor([250.0]),
            "fdm_gamma_rad": torch.tensor([0.0]),
            "raw_alt_m": torch.tensor([5000.0]),
        }
        out = layer(x)
        assert "fdm_g_sin_gamma_ms2" in out
        assert "fdm_cos_gamma" in out
