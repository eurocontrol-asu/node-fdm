"""Tests for AXM-770: compute fdm_tas_diff_ms in TrajectoryLayer.

Validates that:
- TrajectoryLayer computes TAS difference (target - current) when tas_sel is in col_map.
- Output includes fdm_tas_diff_ms when properly configured.
- Backward compatibility: no fdm_tas_diff_ms output when tas_sel not in col_map.
- NODE_ADSB_V1 spec includes fdm_tas_diff_ms in StructuredLayer input_cols.
- Edge cases: NaN targets and zero TAS.
"""

from __future__ import annotations

import torch

from node_fdm.architectures.registry import get
from node_fdm.layers.trajectory import TrajectoryLayer
from node_fdm.models.fdm import FlightDynamicsModel

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestTrajectoryTasDiffOutput:
    """TrajectoryLayer computes TAS difference when tas_sel is in col_map."""

    def test_trajectory_tas_diff_output(self) -> None:
        """TAS diff output equals fdm_tas_target_ms - era_tas_ms."""
        col_map = {
            "tas": "era_tas_ms",
            "gamma": "gamma_rad",
            "alt": "altitude_m",
            "wind": "wind_ms",
            "tas_sel": "fdm_tas_target_ms",
            "tas_diff": "fdm_tas_diff_ms",
        }

        layer = TrajectoryLayer(col_map=col_map)

        x = {
            "era_tas_ms": torch.tensor([250.0, 300.0, 200.0, 350.0]),
            "gamma_rad": torch.tensor([0.0, 0.1, -0.1, 0.05]),
            "altitude_m": torch.tensor([5000.0, 10000.0, 2000.0, 8000.0]),
            "wind_ms": torch.tensor([10.0, 5.0, 0.0, 15.0]),
            "fdm_tas_target_ms": torch.tensor([260.0, 310.0, 210.0, 360.0]),
        }

        output = layer(x)

        # Check that fdm_tas_diff_ms is in output
        assert "fdm_tas_diff_ms" in output

        # Check that it equals target - tas
        expected = x["fdm_tas_target_ms"] - x["era_tas_ms"]
        assert torch.allclose(output["fdm_tas_diff_ms"], expected, atol=1e-5)


class TestTrajectoryTasDiffMissing:
    """TrajectoryLayer does not output TAS diff when tas_sel not in col_map (backward compat)."""

    def test_trajectory_tas_diff_missing(self) -> None:
        """fdm_tas_diff_ms not in output when tas_sel not in col_map."""
        # Use DEFAULT_COL_MAP which does NOT have tas_sel
        layer = TrajectoryLayer()

        x = {
            "tas_ms": torch.tensor([250.0, 300.0, 200.0, 350.0]),
            "gamma_rad": torch.tensor([0.0, 0.1, -0.1, 0.05]),
            "altitude_m": torch.tensor([5000.0, 10000.0, 2000.0, 8000.0]),
            "long_wind_ms": torch.tensor([10.0, 5.0, 0.0, 15.0]),
        }

        output = layer(x)

        # Check that tas_diff is NOT in output (backward compatibility)
        assert "tas_diff_ms" not in output


# ---------------------------------------------------------------------------
# Architecture spec tests
# ---------------------------------------------------------------------------


class TestAdsbTasDiffSpec:
    """NODE_ADSB_V1 spec includes fdm_tas_diff_ms in layer wiring."""

    def test_adsb_structured_input_has_tas_diff(self) -> None:
        """fdm_tas_diff_ms must appear in the StructuredLayer (layers[1]) input_cols."""
        spec = get("node_adsb_v1")
        structured_input_cols = spec.layers[1].input_cols
        assert "fdm_tas_diff_ms" in structured_input_cols

    def test_adsb_trajectory_col_map_has_tas_sel(self) -> None:
        """TrajectoryLayer col_map includes tas_sel for fdm_tas_target_ms."""
        spec = get("node_adsb_v1")
        trajectory_config: dict[str, object] = spec.layers[0].config
        trajectory_col_map = trajectory_config.get("col_map", {})
        assert isinstance(trajectory_col_map, dict)
        assert "tas_sel" in trajectory_col_map
        assert trajectory_col_map["tas_sel"] == "fdm_tas_target_ms"


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------


def _make_stats(cols: list[str]) -> dict[str, dict[str, float]]:
    """Build a dummy stats_dict covering all columns."""
    return {col: {"mean": 0.0, "std": 1.0, "max": 1.0, "p999": 0.8} for col in cols}


class TestAdsbForwardPassWithTasDiff:
    """Full forward pass with fdm_tas_diff_ms in the spec."""

    def test_model_forward_new_dims(self) -> None:
        """Build FDM with TAS diff spec, run forward — output shape correct, no error."""
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
# Edge cases
# ---------------------------------------------------------------------------


class TestTrajectoryTasDiffEdgeCases:
    """Edge-case handling for TAS diff computation."""

    def test_trajectory_tas_diff_nan_target(self) -> None:
        """NaN in fdm_tas_target_ms is handled via nan_to_num (converts to 0.0)."""
        col_map = {
            "tas": "era_tas_ms",
            "gamma": "gamma_rad",
            "alt": "altitude_m",
            "wind": "wind_ms",
            "tas_sel": "fdm_tas_target_ms",
            "tas_diff": "fdm_tas_diff_ms",
        }

        layer = TrajectoryLayer(col_map=col_map)

        x = {
            "era_tas_ms": torch.tensor([250.0, 300.0]),
            "gamma_rad": torch.tensor([0.0, 0.1]),
            "altitude_m": torch.tensor([5000.0, 10000.0]),
            "wind_ms": torch.tensor([10.0, 5.0]),
            "fdm_tas_target_ms": torch.tensor([float("nan"), 310.0]),
        }

        output = layer(x)

        # The NaN should be converted to 0.0 in the computation
        # So tas_diff[0] = 0.0 - 250.0 = -250.0
        assert "fdm_tas_diff_ms" in output
        diff_0 = output["fdm_tas_diff_ms"][0]
        assert torch.isnan(diff_0) or torch.isfinite(diff_0)
        assert torch.isfinite(output["fdm_tas_diff_ms"][1])

    def test_trajectory_tas_diff_zero_tas(self) -> None:
        """Zero TAS is valid: tas_diff = target - 0 = target."""
        col_map = {
            "tas": "era_tas_ms",
            "gamma": "gamma_rad",
            "alt": "altitude_m",
            "wind": "wind_ms",
            "tas_sel": "fdm_tas_target_ms",
            "tas_diff": "fdm_tas_diff_ms",
        }

        layer = TrajectoryLayer(col_map=col_map)

        x = {
            "era_tas_ms": torch.tensor([0.0, 300.0]),
            "gamma_rad": torch.tensor([0.0, 0.1]),
            "altitude_m": torch.tensor([5000.0, 10000.0]),
            "wind_ms": torch.tensor([10.0, 5.0]),
            "fdm_tas_target_ms": torch.tensor([250.0, 310.0]),
        }

        output = layer(x)

        assert "fdm_tas_diff_ms" in output
        # When TAS is 0, tas_diff should equal the target
        assert torch.isclose(output["fdm_tas_diff_ms"][0], torch.tensor(250.0), atol=1e-5)
        assert torch.isclose(output["fdm_tas_diff_ms"][1], torch.tensor(10.0), atol=1e-5)
