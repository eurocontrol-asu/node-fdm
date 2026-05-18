"""Tests for the NODE_ADSB_HYBRID_V1 architecture (AXM-1735).

Verifies the hybrid architecture is registered with Newton-force NN outputs
(fdm_t_minus_d_N, fdm_lift_N), mass state coupling, and that the legacy
NODE_ADSB_V1 baseline is unaffected by the sibling registration.
"""

from __future__ import annotations

import node_fdm.architectures.adsb
import node_fdm.architectures.adsb_hybrid  # noqa: F401 — triggers hybrid auto-register
from node_fdm.architectures.registry import get
from node_fdm_data.schemas import adsb_hybrid as h_schema
from node_fdm_data.schemas.adsb_hybrid import FLIGHT_FEATURE_COLS


class TestHybridV1Registration:
    """AC2, AC11: hybrid architecture is registered and retrievable."""

    def test_hybrid_v1_is_registered(self) -> None:
        """AC2, AC11: ``get('node_adsb_hybrid_v1')`` returns a spec."""
        spec = get("node_adsb_hybrid_v1")
        assert spec is not None
        assert spec.name == "node_adsb_hybrid_v1"


class TestHybridV1State:
    """AC3: x_cols / dx_cols come from the hybrid schema and include mass."""

    def test_hybrid_v1_x_cols_includes_mass(self) -> None:
        """AC3: ``fdm_mass_kg`` is appended at the end of x_cols."""
        spec = get("node_adsb_hybrid_v1")
        assert "fdm_mass_kg" in spec.x_cols
        assert spec.x_cols[-1] == "fdm_mass_kg"
        assert spec.x_cols == h_schema.X_COLS

    def test_hybrid_v1_dx_cols_includes_mass_with_zero(self) -> None:
        """AC3: ``(0, 'fdm_d_mass_kgs')`` is present in dx_cols."""
        spec = get("node_adsb_hybrid_v1")
        assert (0, "fdm_d_mass_kgs") in spec.dx_cols


class TestHybridV1DataOdeLong:
    """AC4: data_ode_long emits Newton forces and matches baseline inputs."""

    def _data_ode_long(self, spec_name: str):  # type: ignore[no-untyped-def]
        spec = get(spec_name)
        for layer in spec.layers:
            if layer.name == "data_ode_long":
                return layer
        raise AssertionError(f"data_ode_long layer missing in {spec_name}")

    def test_hybrid_v1_data_ode_long_outputs_newton_forces(self) -> None:
        """AC4: outputs are exactly ``['fdm_t_minus_d_N', 'fdm_lift_N']``."""
        layer = self._data_ode_long("node_adsb_hybrid_v1")
        assert layer.output_cols == ["fdm_t_minus_d_N", "fdm_lift_N"]

    def test_hybrid_v1_data_ode_long_inputs_match_baseline(self) -> None:
        """AC4: NN-backbone input set is identical to NODE_ADSB_V1 (identifiability)."""
        hybrid_inputs = self._data_ode_long("node_adsb_hybrid_v1").input_cols
        baseline_inputs = self._data_ode_long("node_adsb_v1").input_cols
        assert set(hybrid_inputs) == set(baseline_inputs)


class TestHybridV1Physics:
    """AC5: physics layer wires Newton forces, mass, and kinematic inputs."""

    def _physics(self, spec_name: str):  # type: ignore[no-untyped-def]
        spec = get(spec_name)
        for layer in spec.layers:
            if layer.name == "physics":
                return layer
        raise AssertionError(f"physics layer missing in {spec_name}")

    def test_hybrid_v1_physics_inputs_newton_set(self) -> None:
        """AC5: physics inputs include Newton forces, mass, and kinematic state."""
        physics = self._physics("node_adsb_hybrid_v1")
        required = {
            "fdm_t_minus_d_N",
            "fdm_lift_N",
            "fdm_mass_kg",
            "era_tas_ms",
            "fdm_gamma_rad",
            "fdm_phi_bank_rad",
        }
        assert required.issubset(set(physics.input_cols))


class TestHybridV1DerivedOutputs:
    """AC6: derived_output_cols lists Newton forces and excludes a_spec."""

    def test_hybrid_v1_derived_outputs_include_newton_forces(self) -> None:
        """AC6: Newton-force columns are derived; ``fdm_a_spec_ms2`` is not."""
        spec = get("node_adsb_hybrid_v1")
        assert "fdm_t_minus_d_N" in spec.derived_output_cols
        assert "fdm_lift_N" in spec.derived_output_cols
        assert "fdm_a_spec_ms2" not in spec.derived_output_cols


class TestHybridV1OutputCaps:
    """AC7: nn_output_caps enforce hard physical bounds on NN outputs."""

    def test_hybrid_v1_caps_match_doc(self) -> None:
        """AC7: caps match PHASE_1_MASS_ENCODER §7.1 values."""
        spec = get("node_adsb_hybrid_v1")
        assert spec.nn_output_caps["fdm_t_minus_d_N"] == 5.0e5
        assert spec.nn_output_caps["fdm_lift_N"] == 1.5e6


class TestHybridV1FlightFeatureCols:
    """AC8: flight_feature_cols comes from the hybrid schema."""

    def test_hybrid_v1_flight_feature_cols_set(self) -> None:
        """AC8: spec.flight_feature_cols == schema.FLIGHT_FEATURE_COLS."""
        spec = get("node_adsb_hybrid_v1")
        assert spec.flight_feature_cols == FLIGHT_FEATURE_COLS


class TestHybridV1Bounds:
    """AC9, AC10: x_bounds and dx_bounds clamp mass to TCDS-plausible values."""

    def test_hybrid_v1_x_bounds_for_mass(self) -> None:
        """AC9: ``fdm_mass_kg`` clamped to (OEW, MTOW) of the A320."""
        spec = get("node_adsb_hybrid_v1")
        assert spec.x_bounds["fdm_mass_kg"] == (42_600.0, 77_000.0)

    def test_hybrid_v1_dx_bounds_for_mass_zero(self) -> None:
        """AC10: ``fdm_d_mass_kgs`` strictly pinned to zero."""
        spec = get("node_adsb_hybrid_v1")
        assert spec.dx_bounds["fdm_d_mass_kgs"] == (0.0, 0.0)


class TestBaselineUnchanged:
    """AC11: baseline NODE_ADSB_V1 is unaffected by sibling registration."""

    def test_baseline_v1_unchanged_after_hybrid_import(self) -> None:
        """AC11: baseline x_cols stays the 4-entry list (no mass)."""
        baseline = get("node_adsb_v1")
        assert baseline.x_cols == [
            "raw_alt_m",
            "fdm_gamma_rad",
            "era_tas_ms",
            "fdm_heading_rad",
        ]
        assert "fdm_mass_kg" not in baseline.x_cols
