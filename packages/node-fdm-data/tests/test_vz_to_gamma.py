"""Tests for node_fdm_data.physics.speed — vz_to_gamma conversion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from node_fdm_data.physics.speed import vz_to_gamma

# ---------------------------------------------------------------------------
# Unit tests — vz_to_gamma
# ---------------------------------------------------------------------------


class TestVzToGammaLevel:
    """Level flight: vz=0 → gamma=0."""

    def test_vz_to_gamma_level(self) -> None:
        gamma = vz_to_gamma(0.0, 250.0)
        assert float(gamma) == pytest.approx(0.0, abs=1e-12)


class TestVzToGammaSigned:
    """Climb (vz>0) and descent (vz<0): gamma ≈ arcsin(vz/tas)."""

    @pytest.mark.parametrize(
        ("vz", "tas"),
        [
            pytest.param(10.0, 250.0, id="climb"),
            pytest.param(-7.6, 200.0, id="descent"),
        ],
    )
    def test_vz_to_gamma_signed(self, vz: float, tas: float) -> None:
        gamma = vz_to_gamma(vz, tas)
        expected = math.asin(vz / tas)
        assert float(gamma) == pytest.approx(expected, rel=1e-6)


class TestVzToGammaArray:
    """Element-wise computation on numpy arrays."""

    def test_vz_to_gamma_array(self) -> None:
        vz = np.array([0.0, 10.0, -7.6])
        tas = np.array([250.0, 250.0, 200.0])
        gamma = vz_to_gamma(vz, tas)

        expected = np.arcsin(vz / tas)
        np.testing.assert_allclose(gamma, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCasesVzToGamma:
    """Boundary / degenerate inputs."""

    def test_zero_tas(self) -> None:
        """tas=0, vz=5 → NaN or clamped ±π/2, no crash."""
        result = vz_to_gamma(5.0, 0.0)
        val = float(result)
        assert math.isnan(val) or abs(val) == pytest.approx(math.pi / 2, abs=1e-6)

    def test_ratio_greater_than_one(self) -> None:
        """vz=300, tas=100 → ratio=3.0, clamped to π/2."""
        result = vz_to_gamma(300.0, 100.0)
        val = float(result)
        assert val == pytest.approx(math.pi / 2, abs=1e-6)

    def test_nan_input(self) -> None:
        """vz=NaN → returns NaN."""
        result = vz_to_gamma(float("nan"), 250.0)
        assert math.isnan(float(result))
