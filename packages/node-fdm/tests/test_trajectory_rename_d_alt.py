"""Tests for AXM-844: rename fdm_d_vz_ms → fdm_d_alt_ms.

Model-layer assertions: TrajectoryLayer output keys, architecture DX_COLS,
E1_COLS, and FDM forward output.
"""

from __future__ import annotations

from typing import cast

import torch

from node_fdm.architectures.registry import get
from node_fdm.layers.trajectory import TrajectoryLayer


class TestTrajectoryLayerOutputKeys:
    """Unit: TrajectoryLayer forward output uses fdm_d_alt_ms."""

    def test_trajectory_layer_output_keys(self) -> None:
        """TrajectoryLayer output contains fdm_d_alt_ms, not fdm_d_vz_ms."""
        col_map = {
            "tas": "era_tas_ms",
            "gamma": "fdm_gamma_rad",
            "alt": "raw_alt_m",
            "wind": "fdm_long_wind_ms",
            "alt_sel": "fdm_alt_target_m",
            "vz": "fdm_d_alt_ms",
            "gs": "raw_gs_ms",
            "mach": "era_mach",
            "cas": "fdm_cas_ms",
            "alt_diff": "fdm_alt_diff_m",
        }
        layer = TrajectoryLayer(col_map=col_map)
        x = {
            "era_tas_ms": torch.tensor([250.0]),
            "fdm_gamma_rad": torch.tensor([0.05]),
            "raw_alt_m": torch.tensor([10000.0]),
            "fdm_long_wind_ms": torch.tensor([5.0]),
            "fdm_alt_target_m": torch.tensor([11000.0]),
        }
        output = layer.forward(x)
        assert "fdm_d_alt_ms" in output, "Expected fdm_d_alt_ms in TrajectoryLayer output"
        assert "fdm_d_vz_ms" not in output, "fdm_d_vz_ms should no longer appear"


class TestAdsbArchitectureDxCols:
    """Functional: node_adsb_v1 spec uses fdm_d_alt_ms in DX_COLS."""

    def test_adsb_architecture_dx_cols(self) -> None:
        """DX_COLS[0] == (1, 'fdm_d_alt_ms') after rename."""
        import node_fdm_models.architectures.adsb  # noqa: F401

        spec = get("node_adsb_v1")
        assert spec.dx_cols[0] == (
            1,
            "fdm_d_alt_ms",
        ), f"Expected (1, 'fdm_d_alt_ms'), got {spec.dx_cols[0]}"

    def test_adsb_e1_cols_renamed(self) -> None:
        """E1_COLS uses fdm_d_alt_ms, not fdm_d_vz_ms."""
        from node_fdm_models.schemas.adsb import E1_COLS

        assert "fdm_d_alt_ms" in E1_COLS
        assert "fdm_d_vz_ms" not in E1_COLS

    def test_adsb_col_map_vz_renamed(self) -> None:
        """Architecture col_map maps vz → fdm_d_alt_ms."""
        spec = get("node_adsb_v1")
        col_map = cast("dict[str, str]", spec.layers[0].config["col_map"])
        assert col_map["vz"] == "fdm_d_alt_ms"

    def test_adsb_dx_bounds_renamed(self) -> None:
        """Architecture dx_bounds uses fdm_d_alt_ms key."""
        spec = get("node_adsb_v1")
        assert "fdm_d_alt_ms" in spec.dx_bounds
        assert "fdm_d_vz_ms" not in spec.dx_bounds


class TestFdmForwardOutput:
    """Functional: FDM forward produces fdm_d_alt_ms."""

    def test_fdm_forward_output(self) -> None:
        """TrajectoryLayer with adsb col_map outputs fdm_d_alt_ms."""
        spec = get("node_adsb_v1")
        col_map = cast("dict[str, str]", spec.layers[0].config["col_map"])
        layer = TrajectoryLayer(col_map=col_map)

        x = {
            "era_tas_ms": torch.tensor([250.0]),
            "fdm_gamma_rad": torch.tensor([0.05]),
            "raw_alt_m": torch.tensor([10000.0]),
            "fdm_long_wind_ms": torch.tensor([5.0]),
            "fdm_alt_target_m": torch.tensor([11000.0]),
        }
        output = layer.forward(x)
        assert "fdm_d_alt_ms" in output, "First output column should be fdm_d_alt_ms"
        assert "fdm_d_vz_ms" not in output
