"""Tests for node_fdm_data.physics — constants and ISA model."""

from __future__ import annotations

import numpy as np
import pytest

from node_fdm_data.physics import (
    A0,
    FT,
    GAMMA_AIR,
    KT,
    NM,
    P0,
    RHO0,
    T0,
    TROPOPAUSE_ALT_M,
    G,
    L,
    R,
)
from node_fdm_data.physics.constants import (
    flap_dict,
    gear_set_dict,
    on_ground_dict,
)
from node_fdm_data.physics.isa import isa_density, isa_pressure, isa_temperature

# ── Constants ──────────────────────────────────────────────────────────────


class TestConstants:
    """Verify physical constants are reasonable."""

    def test_sea_level_temperature(self) -> None:
        assert T0 == pytest.approx(288.15)

    def test_sea_level_pressure(self) -> None:
        assert P0 == pytest.approx(101_325.0)

    def test_gravity(self) -> None:
        assert G == pytest.approx(9.80665)

    def test_gas_constant(self) -> None:
        assert R == pytest.approx(287.05)

    def test_gamma(self) -> None:
        assert GAMMA_AIR == pytest.approx(1.4)

    def test_lapse_rate(self) -> None:
        assert L == pytest.approx(0.0065)

    def test_tropopause(self) -> None:
        assert TROPOPAUSE_ALT_M == pytest.approx(11_000.0)

    def test_speed_of_sound(self) -> None:
        expected = (GAMMA_AIR * R * T0) ** 0.5
        assert A0 == pytest.approx(expected)

    def test_density(self) -> None:
        expected = P0 / (R * T0)
        assert RHO0 == pytest.approx(expected)

    def test_nm(self) -> None:
        assert NM == pytest.approx(1852.0)

    def test_ft(self) -> None:
        assert FT == pytest.approx(0.3048)

    def test_kt(self) -> None:
        assert KT == pytest.approx(NM / 3600.0)


class TestLookupDicts:
    """Verify discrete-signal lookup dictionaries."""

    def test_gear_set_dict(self) -> None:
        assert gear_set_dict["UP"] == 1
        assert gear_set_dict["DOWN"] == 0

    def test_flap_dict_keys(self) -> None:
        assert set(flap_dict) == {"0", "1", "2", "3", "FULL", "NRD"}

    def test_on_ground_dict(self) -> None:
        assert on_ground_dict["AIR"] == 0
        assert on_ground_dict["GROUND"] == 1


# ── ISA model ──────────────────────────────────────────────────────────────


class TestIsaTemperature:
    """ISA temperature model."""

    def test_sea_level(self) -> None:
        assert isa_temperature(0.0) == pytest.approx(T0)

    def test_tropopause(self) -> None:
        assert isa_temperature(11_000.0) == pytest.approx(216.65)

    def test_stratosphere(self) -> None:
        """Above 11 km → isothermal at ~216.65 K."""
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
        """FL350 ≈ 35000 ft ≈ 10668 m → ~23842 Pa."""
        h = 35_000 * 0.3048
        assert isa_pressure(h) == pytest.approx(23_842, abs=50)

    def test_negative_altitude(self) -> None:
        """Below sea level → pressure > P0."""
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
