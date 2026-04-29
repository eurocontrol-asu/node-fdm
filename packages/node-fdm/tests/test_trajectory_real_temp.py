"""Tests for TrajectoryLayer real-temperature speed-of-sound.

Mirrors the data-pipeline `tas_to_cas_real`: when the col_map maps `temp` to a
column present in the input, the layer uses that real temperature to compute
the speed of sound (and therefore Mach and CAS) instead of ISA. ISA pressure
remains the static-pressure source. When `temp` is absent from col_map (or
not provided in inputs), the layer falls back to the ISA temperature path.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch

from node_fdm.architectures.registry import get
from node_fdm.layers.trajectory import TrajectoryLayer
from node_fdm_data.physics.speed import tas_to_cas, tas_to_cas_real


class TestTrajectoryRealTemp:
    """Real-temp Mach/CAS path matches NumPy reference; ISA fallback preserved."""

    _col_map_real: ClassVar[dict[str, str]] = {
        "tas": "era_tas_ms",
        "gamma": "fdm_gamma_rad",
        "alt": "raw_alt_m",
        "wind": "fdm_long_wind_ms",
        "temp": "era_temp_K",
        "vz": "fdm_d_alt_ms",
        "gs": "raw_gs_ms",
        "mach": "era_mach",
        "cas": "fdm_cas_ms",
    }

    _col_map_isa: ClassVar[dict[str, str]] = {
        "tas": "era_tas_ms",
        "gamma": "fdm_gamma_rad",
        "alt": "raw_alt_m",
        "wind": "fdm_long_wind_ms",
        "vz": "fdm_d_alt_ms",
        "gs": "raw_gs_ms",
        "mach": "era_mach",
        "cas": "fdm_cas_ms",
    }

    def _base_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "era_tas_ms": torch.tensor([240.0, 240.0]),
            "fdm_gamma_rad": torch.tensor([0.0, 0.0]),
            "raw_alt_m": torch.tensor([10000.0, 10000.0]),
            "fdm_long_wind_ms": torch.tensor([0.0, 0.0]),
        }

    def test_trajectory_cas_uses_real_temp(self) -> None:
        """When `temp` is in col_map and inputs, CAS matches tas_to_cas_real."""
        layer = TrajectoryLayer(col_map=self._col_map_real)

        # Real temp at FL328 is typically ~228 K (warmer than ISA's 218.7 K).
        x = self._base_inputs()
        x["era_temp_K"] = torch.tensor([228.71, 228.71])

        out = layer(x)

        expected = tas_to_cas_real(np.array([240.0]), np.array([10000.0]), np.array([228.71]))
        assert torch.allclose(
            out["fdm_cas_ms"][0:1], torch.tensor(expected, dtype=torch.float32), atol=1e-3
        )

    def test_trajectory_cas_real_temp_differs_from_isa(self) -> None:
        """Real-temp CAS deviates from ISA CAS when era_temp_K != T_ISA."""
        layer_real = TrajectoryLayer(col_map=self._col_map_real)
        layer_isa = TrajectoryLayer(col_map=self._col_map_isa)

        x = self._base_inputs()
        x["era_temp_K"] = torch.tensor([228.71, 228.71])  # warmer than ISA

        out_real = layer_real(x)
        out_isa = layer_isa({k: v for k, v in x.items() if k != "era_temp_K"})

        # The two paths must differ because temperature affects Mach via a = sqrt(gamma*R*T)
        assert not torch.allclose(out_real["fdm_cas_ms"], out_isa["fdm_cas_ms"], atol=0.5)

    def test_trajectory_cas_isa_fallback_when_temp_absent(self) -> None:
        """Without `temp` in col_map, CAS uses ISA path (back-compat)."""
        layer = TrajectoryLayer(col_map=self._col_map_isa)

        x = self._base_inputs()
        out = layer(x)

        expected = tas_to_cas(np.array([240.0]), np.array([10000.0]))
        assert torch.allclose(
            out["fdm_cas_ms"][0:1], torch.tensor(expected, dtype=torch.float32), atol=1e-3
        )

    def test_trajectory_mach_uses_real_temp(self) -> None:
        """Mach = tas / sqrt(gamma*R*T) uses real temperature when mapped."""
        from node_fdm_data.physics.constants import GAMMA_AIR, R

        layer = TrajectoryLayer(col_map=self._col_map_real)

        x = self._base_inputs()
        x["era_temp_K"] = torch.tensor([228.71, 228.71])

        out = layer(x)

        a_real = float(np.sqrt(GAMMA_AIR * R * 228.71))
        expected_mach = 240.0 / a_real
        assert (
            out["era_mach"][0].item() == np.float32(expected_mach).item()
            or abs(out["era_mach"][0].item() - expected_mach) < 1e-4
        )


class TestAdsbRealTempSpec:
    """NODE_ADSB_V1 wires `temp` → `era_temp_K` in the trajectory layer col_map."""

    def test_adsb_trajectory_col_map_has_temp(self) -> None:
        """col_map["temp"] == "era_temp_K" — real ERA5 temperature."""
        spec = get("node_adsb_v1")
        trajectory_config: dict[str, object] = spec.layers[0].config
        col_map = trajectory_config.get("col_map", {})
        assert isinstance(col_map, dict)
        assert col_map.get("temp") == "era_temp_K"
