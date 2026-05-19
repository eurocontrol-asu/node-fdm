"""Tests for the NODE_ADSB_HYBRID_V2 architecture.

Sibling of NODE_ADSB_HYBRID_V1 — same Newton-force outputs and physics wiring,
but the MassEncoder consumes the 6-feature set (5 baseline features plus
``mach_cruise_planned``).
"""

from __future__ import annotations

import node_fdm.architectures.adsb
import node_fdm.architectures.adsb_hybrid
import node_fdm.architectures.adsb_hybrid_v2  # noqa: F401 — registers v2
from node_fdm.architectures.registry import get
from node_fdm_data.schemas.adsb_hybrid import (
    FLIGHT_FEATURE_COLS_6,
    FLIGHT_FEATURE_SIGNS_6,
)


class TestHybridV2Registration:
    def test_hybrid_v2_is_registered(self) -> None:
        spec = get("node_adsb_hybrid_v2")
        assert spec.name == "node_adsb_hybrid_v2"


class TestHybridV2ShapesMatchV1:
    """v2 shares x_cols / dx_cols / trajectory / lateral / physics with v1."""

    def test_state_cols_match_v1(self) -> None:
        assert get("node_adsb_hybrid_v2").x_cols == get("node_adsb_hybrid_v1").x_cols

    def test_dx_cols_match_v1(self) -> None:
        assert get("node_adsb_hybrid_v2").dx_cols == get("node_adsb_hybrid_v1").dx_cols

    def test_data_ode_long_outputs_match_v1(self) -> None:
        def _long(name: str) -> list[str]:
            return next(
                layer.output_cols for layer in get(name).layers if layer.name == "data_ode_long"
            )

        assert _long("node_adsb_hybrid_v2") == _long("node_adsb_hybrid_v1")

    def test_data_ode_long_inputs_match_baseline(self) -> None:
        """Identifiability invariant: v2 NN backbone reads the same set as v1/baseline."""

        def _long_inputs(name: str) -> list[str]:
            return next(
                layer.input_cols for layer in get(name).layers if layer.name == "data_ode_long"
            )

        assert set(_long_inputs("node_adsb_hybrid_v2")) == set(_long_inputs("node_adsb_v1"))


class TestHybridV2FlightFeatures:
    def test_flight_feature_cols_is_6(self) -> None:
        spec = get("node_adsb_hybrid_v2")
        assert spec.flight_feature_cols == FLIGHT_FEATURE_COLS_6
        assert len(spec.flight_feature_cols) == 6
        assert "mach_cruise_planned" in spec.flight_feature_cols

    def test_flight_feature_signs_is_6(self) -> None:
        spec = get("node_adsb_hybrid_v2")
        assert spec.flight_feature_signs == FLIGHT_FEATURE_SIGNS_6


class TestHybridV1Unaffected:
    """Registering v2 must not mutate the v1 spec."""

    def test_v1_still_has_5_features(self) -> None:
        spec = get("node_adsb_hybrid_v1")
        assert len(spec.flight_feature_cols) == 5
        assert "mach_cruise_planned" not in spec.flight_feature_cols
