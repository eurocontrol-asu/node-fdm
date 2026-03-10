"""Tests for node_fdm_data.physics.isa — ISA atmosphere model."""

from __future__ import annotations

import numpy as np
import pytest

from node_fdm_data.physics import P0, RHO0, T0
from node_fdm_data.physics.isa import isa_density, isa_pressure, isa_temperature


class TestIsaTemperature:
    """ISA temperature model."""

    def test_sea_level(self) -> None:
        assert isa_temperature(0.0) == pytest.approx(T0)

    def test_tropopause(self) -> None:
        assert isa_temperature(11_000.0) == pytest.approx(216.65)

    def test_stratosphere(self) -> None:
        """Above 11 km -> isothermal at ~216.65 K."""
        assert isa_temperature(15_000.0) == pytest.approx(216.65)

    def test_vectorised(self) -> None:
        h = np.array([0.0, 5_000.0, 11_000.0, 15_000.0])
        result = np.asarray(isa_temperature(h))
        assert result[0] == pytest.approx(T0)
        assert result[2] == pytest.approx(216.65)
        assert result[3] == pytest.approx(216.65)


class TestIsaPressure:
    """ISA pressure model."""

    def test_sea_level(self) -> None:
        assert isa_pressure(0.0) == pytest.approx(101_325.0)

    def test_cruise_altitude(self) -> None:
        """FL350 ~ 35000 ft ~ 10668 m -> ~23842 Pa."""
        h = 35_000 * 0.3048
        assert isa_pressure(h) == pytest.approx(23_842, abs=50)

    def test_negative_altitude(self) -> None:
        """Below sea level -> pressure > P0."""
        p = isa_pressure(-500.0)
        assert float(p) > P0

    def test_tropopause_continuity(self) -> None:
        """Pressure is continuous across the tropopause boundary."""
        p_below = isa_pressure(10_999.0)
        p_above = isa_pressure(11_001.0)
        assert float(p_below) == pytest.approx(float(p_above), rel=1e-3)


class TestIsaDensity:
    """ISA density model."""

    def test_sea_level(self) -> None:
        assert isa_density(0.0) == pytest.approx(RHO0, rel=1e-6)

    def test_decreases_with_altitude(self) -> None:
        rho_0 = float(isa_density(0.0))
        rho_5k = float(isa_density(5_000.0))
        rho_10k = float(isa_density(10_000.0))
        assert rho_0 > rho_5k > rho_10k
