"""Tests for node_fdm_data.physics.constants — physical constants and lookup dicts."""

from __future__ import annotations

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
