"""Tests for TrajectoryLayer tas_diff masking via fdm_tas_target_known.

Mirrors the gamma_diff/known pattern:
- known=1: tas_diff = target - tas
- known=0: tas_diff = 0  (avoids the false -tas signal that nan_to_num
  on the target alone would produce when target is NaN)
- tas_known flag is passed through to output for the StructuredLayer.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from node_fdm.architectures.registry import get
from node_fdm.layers.trajectory import TrajectoryLayer


class TestTrajectoryTasDiffKnownUnknown:
    """tas_diff = known * (target - tas); 0 when unknown."""

    _col_map: ClassVar[dict[str, str]] = {
        "tas": "era_tas_ms",
        "gamma": "fdm_gamma_rad",
        "alt": "raw_alt_m",
        "wind": "fdm_long_wind_ms",
        "tas_sel": "fdm_tas_target_ms",
        "tas_known": "fdm_tas_target_known",
        "tas_diff": "fdm_tas_diff_ms",
    }

    def _base_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "era_tas_ms": torch.tensor([250.0, 300.0]),
            "fdm_gamma_rad": torch.tensor([0.05, -0.03]),
            "raw_alt_m": torch.tensor([5000.0, 10000.0]),
            "fdm_long_wind_ms": torch.tensor([10.0, 5.0]),
        }

    def test_tas_diff_unknown_is_zero(self) -> None:
        """known=0 → tas_diff = 0 (not -tas)."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["fdm_tas_target_ms"] = torch.tensor([0.0, 0.0])
        x["fdm_tas_target_known"] = torch.tensor([0.0, 0.0])

        output = layer(x)

        assert "fdm_tas_diff_ms" in output
        assert torch.allclose(output["fdm_tas_diff_ms"], torch.zeros(2), atol=1e-6)

    def test_tas_diff_known_uses_target(self) -> None:
        """known=1 → tas_diff = target - tas."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["era_tas_ms"] = torch.tensor([250.0, 250.0])
        x["fdm_tas_target_ms"] = torch.tensor([260.0, 260.0])
        x["fdm_tas_target_known"] = torch.tensor([1.0, 1.0])

        output = layer(x)

        expected = torch.tensor([10.0, 10.0])
        assert torch.allclose(output["fdm_tas_diff_ms"], expected, atol=1e-5)

    def test_tas_diff_mixed_known_unknown(self) -> None:
        """Mixed: target where known=1, zero where known=0."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["era_tas_ms"] = torch.tensor([250.0, 250.0])
        x["fdm_tas_target_ms"] = torch.tensor([0.0, 260.0])
        x["fdm_tas_target_known"] = torch.tensor([0.0, 1.0])

        output = layer(x)

        # idx 0: unknown → 0  (NOT -250 like the buggy pre-fix behavior)
        assert torch.isclose(output["fdm_tas_diff_ms"][0], torch.tensor(0.0), atol=1e-5)
        # idx 1: known → 260 - 250 = 10
        assert torch.isclose(output["fdm_tas_diff_ms"][1], torch.tensor(10.0), atol=1e-5)

    def test_tas_known_passthrough(self) -> None:
        """fdm_tas_target_known is propagated to output for StructuredLayer."""
        layer = TrajectoryLayer(col_map=self._col_map)

        x = self._base_inputs()
        x["fdm_tas_target_ms"] = torch.tensor([260.0, 260.0])
        x["fdm_tas_target_known"] = torch.tensor([1.0, 0.0])

        output = layer(x)

        assert "fdm_tas_target_known" in output
        assert torch.allclose(output["fdm_tas_target_known"], torch.tensor([1.0, 0.0]))

    def test_tas_diff_no_known_key_backcompat(self) -> None:
        """Without tas_known in col_map: fall back to nan_to_num (legacy path)."""
        col_map_no_known = {
            "tas": "era_tas_ms",
            "gamma": "fdm_gamma_rad",
            "alt": "raw_alt_m",
            "wind": "fdm_long_wind_ms",
            "tas_sel": "fdm_tas_target_ms",
            "tas_diff": "fdm_tas_diff_ms",
        }
        layer = TrajectoryLayer(col_map=col_map_no_known)

        x = {
            "era_tas_ms": torch.tensor([250.0]),
            "fdm_gamma_rad": torch.tensor([0.05]),
            "raw_alt_m": torch.tensor([5000.0]),
            "fdm_long_wind_ms": torch.tensor([10.0]),
            "fdm_tas_target_ms": torch.tensor([260.0]),
        }

        output = layer(x)

        # Legacy behavior: target - tas
        assert torch.isclose(output["fdm_tas_diff_ms"][0], torch.tensor(10.0), atol=1e-5)


class TestAdsbTasKnownSpec:
    """NODE_ADSB_V1 spec wires fdm_tas_target_known into trajectory + structured layers."""

    def test_adsb_trajectory_col_map_has_tas_known(self) -> None:
        """TrajectoryLayer col_map maps tas_known → fdm_tas_target_known."""
        spec = get("node_adsb_v1")
        trajectory_config: dict[str, object] = spec.layers[0].config
        trajectory_col_map = trajectory_config.get("col_map", {})
        assert isinstance(trajectory_col_map, dict)
        assert "tas_known" in trajectory_col_map
        assert trajectory_col_map["tas_known"] == "fdm_tas_target_known"

    def test_adsb_structured_input_has_tas_known(self) -> None:
        """fdm_tas_target_known flag in StructuredLayer input_cols."""
        spec = get("node_adsb_v1")
        assert "fdm_tas_target_known" in spec.layers[1].input_cols
