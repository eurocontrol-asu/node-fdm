"""Tests for TrajectoryLayer gamma_diff computation.

AXM-806: initial gamma_diff support.
AXM-810: NaN-aware gamma_diff — NaN target yields zero diff, not -gamma.

Validates that:
- TrajectoryLayer computes gamma difference (target - current) when gamma_sel is in col_map.
- Output includes fdm_gamma_diff_rad when properly configured.
- Backward compatibility: no gamma_diff output when gamma_sel not in col_map.
- NODE_ADSB_V1 spec includes fdm_gamma_diff_rad in StructuredLayer input_cols.
- NaN-aware: NaN gamma_target → gamma_diff = 0 (not -gamma).
"""

from __future__ import annotations

from typing import ClassVar

import torch

from node_fdm.architectures.registry import get
from node_fdm.layers.trajectory import TrajectoryLayer
from node_fdm.models.fdm import FlightDynamicsModel

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestTrajectoryGammaDiffOutput:
    """TrajectoryLayer computes gamma difference when gamma_sel is in col_map."""

    def test_trajectory_gamma_diff_output(self) -> None:
        """gamma_diff output equals gamma_target - gamma."""
        col_map = {
            "tas": "era_tas_ms",
            "gamma": "fdm_gamma_rad",
            "alt": "raw_alt_m",
            "wind": "fdm_long_wind_ms",
            "gamma_sel": "fdm_gamma_sel_rad",
            "gamma_diff": "fdm_gamma_diff_rad",
        }

        layer = TrajectoryLayer(col_map=col_map)

        x = {
            "era_tas_ms": torch.tensor([250.0]),
            "fdm_gamma_rad": torch.tensor([0.05]),
            "raw_alt_m": torch.tensor([5000.0]),
            "fdm_long_wind_ms": torch.tensor([10.0]),
            "fdm_gamma_sel_rad": torch.tensor([0.0]),
        }

        output = layer(x)

        assert "fdm_gamma_diff_rad" in output
        expected = torch.tensor([-0.05])  # target(0.0) - gamma(0.05)
        assert torch.allclose(output["fdm_gamma_diff_rad"], expected, atol=1e-5)


class TestTrajectoryGammaDiffMissing:
    """TrajectoryLayer does not output gamma diff when gamma_sel not in col_map."""

    def test_trajectory_gamma_diff_missing(self) -> None:
        """No gamma_diff in output when gamma_sel not in col_map."""
        layer = TrajectoryLayer()

        x = {
            "tas_ms": torch.tensor([250.0]),
            "gamma_rad": torch.tensor([0.05]),
            "altitude_m": torch.tensor([5000.0]),
            "long_wind_ms": torch.tensor([10.0]),
        }

        output = layer(x)

        assert "gamma_diff_rad" not in output


# ---------------------------------------------------------------------------
# Architecture spec tests
# ---------------------------------------------------------------------------


class TestAdsbGammaDiffSpec:
    """NODE_ADSB_V1 spec includes fdm_gamma_diff_rad in layer wiring."""

    def test_adsb_structured_input_has_gamma_diff(self) -> None:
        """fdm_gamma_diff_rad must appear in the StructuredLayer (layers[1]) input_cols."""
        spec = get("node_adsb_v1")
        structured_input_cols = spec.layers[1].input_cols
        assert "fdm_gamma_diff_rad" in structured_input_cols

    def test_adsb_trajectory_col_map_has_gamma_sel(self) -> None:
        """TrajectoryLayer col_map includes gamma_sel for fdm_gamma_target_rad."""
        spec = get("node_adsb_v1")
        trajectory_config: dict[str, object] = spec.layers[0].config
        trajectory_col_map = trajectory_config.get("col_map", {})
        assert isinstance(trajectory_col_map, dict)
        assert "gamma_sel" in trajectory_col_map
        assert trajectory_col_map["gamma_sel"] == "fdm_gamma_target_rad"


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def _make_stats(cols: list[str]) -> dict[str, dict[str, float]]:
    """Build a dummy stats_dict covering all columns."""
    return {col: {"mean": 0.0, "std": 1.0, "max": 1.0} for col in cols}


class TestAdsbForwardPassWithGammaDiff:
    """Full forward pass with fdm_gamma_diff_rad in the spec."""

    def test_model_forward_new_dims(self) -> None:
        """Build FDM with gamma diff spec, run forward — output shape correct, no error."""
        spec = get("node_adsb_v1")
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols
        stats = _make_stats(all_cols)

        model = FlightDynamicsModel(spec, stats)
        batch = 4
        x = torch.randn(batch, len(spec.x_cols))
        u = torch.randn(batch, len(spec.u_cols))
        e = torch.randn(batch, len(spec.e0_cols))

        out = model(x, u, e)
        assert out.shape == (batch, len(spec.dx_cols))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGammaDiffLearnableDefault:
    """Gamma diff uses known mask + learnable default when target is unknown."""

    _col_map: ClassVar[dict[str, str]] = {
        "tas": "era_tas_ms",
        "gamma": "fdm_gamma_rad",
        "alt": "raw_alt_m",
        "wind": "fdm_long_wind_ms",
        "gamma_sel": "fdm_gamma_target_rad",
        "gamma_known": "fdm_gamma_target_known",
        "gamma_diff": "fdm_gamma_diff_rad",
    }

    def _base_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "era_tas_ms": torch.tensor([250.0, 300.0]),
            "fdm_gamma_rad": torch.tensor([0.05, -0.03]),
            "raw_alt_m": torch.tensor([5000.0, 10000.0]),
            "fdm_long_wind_ms": torch.tensor([10.0, 5.0]),
        }

    def test_gamma_diff_unknown_uses_default(self) -> None:
        """known=0 → gamma_diff = gamma_default - gamma (learnable)."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["fdm_gamma_target_rad"] = torch.tensor([0.0, 0.0])  # filled value (was NaN)
        x["fdm_gamma_target_known"] = torch.tensor([0.0, 0.0])  # unknown

        output = layer(x)

        assert "fdm_gamma_diff_rad" in output
        # gamma_default starts at 0.0, gamma = [0.05, -0.03]
        # diff = 0.0 - gamma = [-0.05, 0.03]
        expected = torch.tensor([0.0 - 0.05, 0.0 - (-0.03)])
        assert torch.allclose(output["fdm_gamma_diff_rad"], expected, atol=1e-6)

    def test_gamma_diff_known_uses_target(self) -> None:
        """known=1 → gamma_diff = target - gamma."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["fdm_gamma_rad"] = torch.tensor([0.03, 0.03])
        x["fdm_gamma_target_rad"] = torch.tensor([0.05, 0.05])
        x["fdm_gamma_target_known"] = torch.tensor([1.0, 1.0])

        output = layer(x)

        expected = torch.tensor([0.02, 0.02])
        assert torch.allclose(output["fdm_gamma_diff_rad"], expected, atol=1e-6)

    def test_gamma_diff_mixed_known_unknown(self) -> None:
        """Mixed known/unknown: target where known=1, default where known=0."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["fdm_gamma_target_rad"] = torch.tensor([0.0, 0.02])  # [filled, real]
        x["fdm_gamma_target_known"] = torch.tensor([0.0, 1.0])  # [unknown, known]

        output = layer(x)

        # idx 0: unknown → gamma_default(0.0) - 0.05 = -0.05
        assert torch.isclose(output["fdm_gamma_diff_rad"][0], torch.tensor(-0.05), atol=1e-6)
        # idx 1: known → 0.02 - (-0.03) = 0.05
        assert torch.isclose(output["fdm_gamma_diff_rad"][1], torch.tensor(0.05), atol=1e-6)

    def test_gamma_default_is_learnable(self) -> None:
        """gamma_default_net is an nn.Module with trainable parameters."""
        layer = TrajectoryLayer(col_map=self._col_map)
        assert hasattr(layer, "gamma_default_net")
        params = list(layer.gamma_default_net.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)


class TestTrajectoryGammaDiffEdgeCases:
    """Edge-case handling for gamma diff computation."""

    def test_trajectory_gamma_diff_zero_gamma(self) -> None:
        """Zero gamma (level flight): gamma_diff = target - 0 = target."""
        col_map = {
            "tas": "era_tas_ms",
            "gamma": "fdm_gamma_rad",
            "alt": "raw_alt_m",
            "wind": "fdm_long_wind_ms",
            "gamma_sel": "fdm_gamma_sel_rad",
            "gamma_diff": "fdm_gamma_diff_rad",
        }

        layer = TrajectoryLayer(col_map=col_map)

        x = {
            "era_tas_ms": torch.tensor([250.0]),
            "fdm_gamma_rad": torch.tensor([0.0]),
            "raw_alt_m": torch.tensor([5000.0]),
            "fdm_long_wind_ms": torch.tensor([10.0]),
            "fdm_gamma_sel_rad": torch.tensor([0.03]),
        }

        output = layer(x)

        assert "fdm_gamma_diff_rad" in output
        assert torch.isclose(output["fdm_gamma_diff_rad"][0], torch.tensor(0.03), atol=1e-5)
