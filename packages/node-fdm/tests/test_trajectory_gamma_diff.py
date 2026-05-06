"""Tests for TrajectoryLayer gamma_diff computation.

AXM-806: initial gamma_diff support.
AXM-810: NaN-aware gamma_diff — NaN target yields zero diff, not -gamma.

After removing GammaDefaultNet:
- known=1: gamma_diff = target - gamma
- known=0: gamma_diff = 0
- gamma_known flag is passed through to output
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

    def test_adsb_structured_input_has_gamma_known(self) -> None:
        """fdm_gamma_target_known flag in StructuredLayer input_cols."""
        spec = get("node_adsb_v1")
        assert "fdm_gamma_target_known" in spec.layers[1].input_cols


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def _make_stats(cols: list[str]) -> dict[str, dict[str, float]]:
    """Build a dummy stats_dict covering all columns."""
    return {col: {"mean": 0.0, "std": 1.0, "max": 1.0, "p999": 0.8} for col in cols}


class TestAdsbForwardPassWithGammaDiff:
    """Full forward pass with fdm_gamma_diff_rad in the spec."""

    def test_model_forward_new_dims(self) -> None:
        """Build FDM with gamma diff spec, run forward — output shape correct, no error."""
        spec = get("node_adsb_v1")
        dx_col_names = [c for _, c in spec.dx_cols]
        all_cols = spec.x_cols + spec.u_cols + spec.e0_cols + spec.e1_cols + dx_col_names
        all_cols += spec.derived_output_cols
        stats = _make_stats(all_cols)

        model = FlightDynamicsModel(spec, stats)
        batch = 4
        x = torch.randn(batch, len(spec.x_cols))
        u = torch.randn(batch, len(spec.u_cols))
        e = torch.randn(batch, len(spec.e0_cols))

        out = model(x, u, e)
        assert out.shape == (batch, len(spec.dx_cols))


# ---------------------------------------------------------------------------
# Known / Unknown gamma_diff tests
# ---------------------------------------------------------------------------


class TestGammaDiffKnownUnknown:
    """gamma_diff = known * (target - gamma); 0 when unknown."""

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

    def test_gamma_diff_unknown_is_zero(self) -> None:
        """known=0 → gamma_diff = 0."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["fdm_gamma_target_rad"] = torch.tensor([0.0, 0.0])
        x["fdm_gamma_target_known"] = torch.tensor([0.0, 0.0])

        output = layer(x)

        assert "fdm_gamma_diff_rad" in output
        assert torch.allclose(output["fdm_gamma_diff_rad"], torch.zeros(2), atol=1e-6)

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
        """Mixed: target where known=1, zero where known=0."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["fdm_gamma_target_rad"] = torch.tensor([0.0, 0.02])
        x["fdm_gamma_target_known"] = torch.tensor([0.0, 1.0])

        output = layer(x)

        # idx 0: unknown → 0
        assert torch.isclose(output["fdm_gamma_diff_rad"][0], torch.tensor(0.0), atol=1e-6)
        # idx 1: known → 0.02 - (-0.03) = 0.05
        assert torch.isclose(output["fdm_gamma_diff_rad"][1], torch.tensor(0.05), atol=1e-6)

    def test_no_trainable_parameters(self) -> None:
        """TrajectoryLayer has no trainable parameters (GammaNet removed)."""
        layer = TrajectoryLayer(col_map=self._col_map)
        params = list(layer.parameters())
        assert len(params) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


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
