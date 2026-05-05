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

    @pytest.mark.parametrize(
        ("actual", "expected"),
        [
            pytest.param(T0, 288.15, id="sea_level_temperature"),
            pytest.param(P0, 101_325.0, id="sea_level_pressure"),
            pytest.param(G, 9.80665, id="gravity"),
            pytest.param(R, 287.05, id="gas_constant"),
            pytest.param(GAMMA_AIR, 1.4, id="gamma"),
            pytest.param(L, 0.0065, id="lapse_rate"),
            pytest.param(TROPOPAUSE_ALT_M, 11_000.0, id="tropopause"),
            pytest.param(NM, 1852.0, id="nm"),
            pytest.param(FT, 0.3048, id="ft"),
        ],
    )
    def test_literal_constant(self, actual: float, expected: float) -> None:
        assert actual == pytest.approx(expected)

    def test_speed_of_sound(self) -> None:
        expected = (GAMMA_AIR * R * T0) ** 0.5
        assert A0 == pytest.approx(expected)

    def test_density(self) -> None:
        expected = P0 / (R * T0)
        assert RHO0 == pytest.approx(expected)

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
