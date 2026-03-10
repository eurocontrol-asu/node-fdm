"""Tests for BADA 4.2 aircraft mapping."""

from __future__ import annotations

import pytest

from node_fdm_bada.aircraft_mapping import BADA_4_2_MAPPING, get_bada_identifier


class TestAircraftMapping:
    """Tests for ICAO → BADA 4.2 identifier mapping."""

    def test_a320(self) -> None:
        assert BADA_4_2_MAPPING["A320"] == "A320-214"

    def test_b738(self) -> None:
        assert BADA_4_2_MAPPING["B738"] == "B738W26"

    def test_a388(self) -> None:
        assert BADA_4_2_MAPPING["A388"] == "A380-841"

    def test_unknown_aircraft_dict(self) -> None:
        with pytest.raises(KeyError):
            _ = BADA_4_2_MAPPING["UNKNOWN"]

    def test_get_bada_identifier_valid(self) -> None:
        assert get_bada_identifier("A320") == "A320-214"

    def test_get_bada_identifier_unknown(self) -> None:
        with pytest.raises(KeyError):
            get_bada_identifier("UNKNOWN")

    def test_mapping_not_empty(self) -> None:
        assert len(BADA_4_2_MAPPING) > 60
