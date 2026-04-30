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


class TestDynamicPressurePresent:
    """fdm_q_pa and fdm_g_over_v appear in TrajectoryLayer output with adsb col_map."""

    def _adsb_layer(self) -> TrajectoryLayer:
        return TrajectoryLayer(
            col_map={
                "tas": "era_tas_ms",
                "gamma": "fdm_gamma_rad",
                "alt": "raw_alt_m",
                "q": "fdm_q_pa",
                "g_over_v": "fdm_g_over_v",
            }
        )

    def _inputs(self, tas: float, alt: float = 0.0) -> dict[str, torch.Tensor]:
        return {
            "era_tas_ms": torch.tensor([tas]),
            "fdm_gamma_rad": torch.tensor([0.0]),
            "raw_alt_m": torch.tensor([alt]),
        }

    def test_q_pa_in_output(self) -> None:
        """fdm_q_pa key exists in TrajectoryLayer output."""
        layer = self._adsb_layer()
        out = layer(self._inputs(200.0))
        assert "fdm_q_pa" in out

    def test_g_over_v_in_output(self) -> None:
        """fdm_g_over_v key exists in TrajectoryLayer output."""
        layer = self._adsb_layer()
        out = layer(self._inputs(200.0))
        assert "fdm_g_over_v" in out


class TestDynamicPressureValues:
    """Numeric correctness of fdm_q_pa and fdm_g_over_v."""

    G = 9.80665
    R = 287.05  # J/(kg·K)
    P0 = 101_325.0  # Pa
    T0 = 288.15  # K

    def _adsb_layer(self) -> TrajectoryLayer:
        return TrajectoryLayer(
            col_map={
                "tas": "era_tas_ms",
                "gamma": "fdm_gamma_rad",
                "alt": "raw_alt_m",
                "q": "fdm_q_pa",
                "g_over_v": "fdm_g_over_v",
            }
        )

    def _inputs(self, tas: float, alt: float = 0.0) -> dict[str, torch.Tensor]:
        return {
            "era_tas_ms": torch.tensor([tas]),
            "fdm_gamma_rad": torch.tensor([0.0]),
            "raw_alt_m": torch.tensor([alt]),
        }

    def test_q_sea_level_v200(self) -> None:
        """At sea level ISA (alt=0), V=200 m/s: rho≈1.225, q≈24500 Pa."""
        layer = self._adsb_layer()
        out = layer(self._inputs(200.0, alt=0.0))
        # rho = P0 / (R * T0) = 101325 / (287.05 * 288.15) ≈ 1.2252
        rho_expected = self.P0 / (self.R * self.T0)
        q_expected = 0.5 * rho_expected * 200.0**2  # ≈ 24504 Pa
        assert torch.isclose(
            out["fdm_q_pa"], torch.tensor(q_expected, dtype=torch.float32), rtol=1e-3
        )

    def test_q_cruise_alt10000_v230(self) -> None:
        """At cruise (alt=10000 m, ISA T≈223.15 K), V=230 m/s: q≈8800 Pa."""
        layer = self._adsb_layer()
        out = layer(self._inputs(230.0, alt=10000.0))
        # ISA at 10 km: T=223.15 K, P=26500 Pa (approx); rho≈P/(R*T)≈0.414
        # q ≈ 0.5 * 0.414 * 230^2 ≈ 10949 Pa
        # Loose tolerance to accept both ISA formula variations
        q_val = out["fdm_q_pa"].item()
        assert 7000.0 < q_val < 13000.0, f"q={q_val:.1f} Pa out of expected range"

    def test_g_over_v_at_200(self) -> None:
        """At V=200 m/s, g/V = 9.80665/200 ≈ 0.04903."""
        layer = self._adsb_layer()
        out = layer(self._inputs(200.0))
        expected = self.G / 200.0
        assert torch.isclose(
            out["fdm_g_over_v"], torch.tensor(expected, dtype=torch.float32), rtol=1e-4
        )

    def test_g_over_v_at_10(self) -> None:
        """At V=10 m/s (low speed), g/V = 9.80665/10 ≈ 0.9807, finite."""
        layer = self._adsb_layer()
        out = layer(self._inputs(10.0))
        expected = self.G / 10.0
        assert torch.isclose(
            out["fdm_g_over_v"], torch.tensor(expected, dtype=torch.float32), rtol=1e-4
        )

    def test_g_over_v_zero_tas_clamped(self) -> None:
        """At V→0 (clamped to 1.0 m/s), g/V must be finite (no NaN/Inf)."""
        layer = self._adsb_layer()
        out = layer(self._inputs(0.0))
        val = out["fdm_g_over_v"]
        assert torch.isfinite(val).all(), f"g_over_v is not finite at V=0: {val}"
        # Clamped to 1.0 m/s: g/V = G/1.0
        assert torch.isclose(val, torch.tensor(self.G, dtype=torch.float32), rtol=1e-4)

    def test_g_over_v_negative_gamma_unaffected(self) -> None:
        """g/V depends only on TAS magnitude, not gamma sign."""
        layer = self._adsb_layer()
        x_pos = {
            "era_tas_ms": torch.tensor([200.0]),
            "fdm_gamma_rad": torch.tensor([0.1]),
            "raw_alt_m": torch.tensor([5000.0]),
        }
        x_neg = {
            "era_tas_ms": torch.tensor([200.0]),
            "fdm_gamma_rad": torch.tensor([-0.1]),
            "raw_alt_m": torch.tensor([5000.0]),
        }
        out_pos = layer(x_pos)
        out_neg = layer(x_neg)
        assert torch.isclose(out_pos["fdm_g_over_v"], out_neg["fdm_g_over_v"], atol=1e-6)

    def test_batch_shape_q_g_over_v(self) -> None:
        """Batch shape is preserved for dynamic pressure features."""
        layer = self._adsb_layer()
        x = {
            "era_tas_ms": torch.tensor([200.0, 230.0, 100.0]),
            "fdm_gamma_rad": torch.tensor([0.0, 0.05, -0.02]),
            "raw_alt_m": torch.tensor([0.0, 10000.0, 2000.0]),
        }
        out = layer(x)
        assert out["fdm_q_pa"].shape == (3,)
        assert out["fdm_g_over_v"].shape == (3,)
