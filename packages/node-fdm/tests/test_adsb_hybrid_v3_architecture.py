from __future__ import annotations

import math

import node_fdm.architectures.adsb  # register node_adsb_v1
import node_fdm.architectures.adsb_hybrid  # register node_adsb_hybrid_v1
import node_fdm.architectures.adsb_hybrid_v2  # register node_adsb_hybrid_v2
import node_fdm.architectures.adsb_hybrid_v3  # noqa: F401  # register node_adsb_hybrid_v3
from node_fdm.architectures.registry import ArchitectureSpec, LayerSpec, get
from node_fdm_data.schemas.adsb_hybrid import (
    FLIGHT_FEATURE_COLS_6,
    FLIGHT_FEATURE_SIGNS_6,
)


def _layer(spec: ArchitectureSpec, name: str) -> LayerSpec:
    for layer in spec.layers:
        if layer.name == name:
            return layer
    raise AssertionError(f"layer {name!r} not found in spec {spec.name!r}")


def test_hybrid_v3_is_registered() -> None:
    """AC1, AC2, AC11: v3 is registered under its canonical name."""
    spec = get("node_adsb_hybrid_v3")
    assert spec.name == "node_adsb_hybrid_v3"


def test_hybrid_v3_state_cols_match_hybrid_v2() -> None:
    """AC3: state/control/env/derivative columns identical to v2."""
    v2 = get("node_adsb_hybrid_v2")
    v3 = get("node_adsb_hybrid_v3")
    assert v3.x_cols == v2.x_cols
    assert v3.dx_cols == v2.dx_cols
    assert v3.u_cols == v2.u_cols
    assert v3.e0_cols == v2.e0_cols
    assert v3.e1_cols == v2.e1_cols


def test_hybrid_v3_data_ode_long_outputs_cl_residual() -> None:
    """AC4: data_ode_long output_cols swap lift_residual_norm for cl_residual."""
    v3 = get("node_adsb_hybrid_v3")
    data_ode_long = _layer(v3, "data_ode_long")
    assert data_ode_long.output_cols == ["fdm_t_minus_d_norm", "fdm_cl_residual"]


def test_hybrid_v3_data_ode_long_inputs_match_baseline() -> None:
    """AC4: data_ode_long inputs identical to NODE_ADSB_V1 (identifiability)."""
    baseline = get("node_adsb_v1")
    v3 = get("node_adsb_hybrid_v3")
    base_inputs = _layer(baseline, "data_ode_long").input_cols
    v3_inputs = _layer(v3, "data_ode_long").input_cols
    assert set(v3_inputs) == set(base_inputs)


def test_hybrid_v3_physics_inputs_use_cl_and_q() -> None:
    """AC5: physics layer uses CL-mode contract (cl_residual + q_pa, no lift_residual_norm)."""
    v3 = get("node_adsb_hybrid_v3")
    physics = _layer(v3, "physics")
    inputs = physics.input_cols
    assert "fdm_t_minus_d_norm" in inputs
    assert "fdm_cl_residual" in inputs
    assert "fdm_q_pa" in inputs
    assert "fdm_mass_kg" in inputs
    assert "era_tas_ms" in inputs
    assert "fdm_gamma_rad" in inputs
    assert "fdm_phi_bank_rad" in inputs
    assert "fdm_lift_residual_norm" not in inputs


def test_hybrid_v3_trajectory_identical_to_v2() -> None:
    """AC6: trajectory LayerSpec byte-identical to v2."""
    v2 = get("node_adsb_hybrid_v2")
    v3 = get("node_adsb_hybrid_v3")
    v2_traj = _layer(v2, "trajectory")
    v3_traj = _layer(v3, "trajectory")
    assert v3_traj.name == v2_traj.name
    assert v3_traj.layer_class == v2_traj.layer_class
    assert v3_traj.input_cols == v2_traj.input_cols
    assert v3_traj.output_cols == v2_traj.output_cols
    assert v3_traj.trainable == v2_traj.trainable
    assert v3_traj.config == v2_traj.config


def test_hybrid_v3_data_ode_lat_identical_to_v2() -> None:
    """AC6: data_ode_lat LayerSpec byte-identical to v2."""
    v2 = get("node_adsb_hybrid_v2")
    v3 = get("node_adsb_hybrid_v3")
    v2_lat = _layer(v2, "data_ode_lat")
    v3_lat = _layer(v3, "data_ode_lat")
    assert v3_lat.name == v2_lat.name
    assert v3_lat.layer_class == v2_lat.layer_class
    assert v3_lat.input_cols == v2_lat.input_cols
    assert v3_lat.output_cols == v2_lat.output_cols
    assert v3_lat.trainable == v2_lat.trainable
    assert v3_lat.config == v2_lat.config


def test_hybrid_v3_derived_outputs_include_cl_residual() -> None:
    """AC7: derived_output_cols lists cl_residual, not lift_residual_norm."""
    v3 = get("node_adsb_hybrid_v3")
    assert "fdm_cl_residual" in v3.derived_output_cols
    assert "fdm_t_minus_d_norm" in v3.derived_output_cols
    assert "fdm_phi_bank_rad" in v3.derived_output_cols
    assert "fdm_lift_residual_norm" not in v3.derived_output_cols


def test_hybrid_v3_caps_match_spec() -> None:
    """AC8: nn_output_caps match the CL-mode contract values."""
    v3 = get("node_adsb_hybrid_v3")
    caps = v3.nn_output_caps
    assert caps["fdm_cl_residual"] == 1.0
    assert caps["fdm_t_minus_d_norm"] == 8.0
    assert caps["fdm_phi_bank_rad"] == math.pi / 3


def test_hybrid_v3_mass_bounds_match_v2() -> None:
    """AC9: mass safety-net bounds identical to v2 (A320 OEW/MTOW, frozen mass derivative)."""
    v3 = get("node_adsb_hybrid_v3")
    assert v3.x_bounds["fdm_mass_kg"] == (42_600.0, 77_000.0)
    assert v3.dx_bounds["fdm_d_mass_kgs"] == (0.0, 0.0)


def test_hybrid_v3_flight_feature_cols_match_v2() -> None:
    """AC10: flight features mirror the 6-feature MassEncoder contract."""
    v3 = get("node_adsb_hybrid_v3")
    assert v3.flight_feature_cols == FLIGHT_FEATURE_COLS_6
    assert v3.flight_feature_signs == FLIGHT_FEATURE_SIGNS_6


def test_hybrid_v2_unchanged_after_v3_import() -> None:
    """AC11: importing v3 does not mutate v2 (independent rollback)."""
    v2 = get("node_adsb_hybrid_v2")
    data_ode_long = _layer(v2, "data_ode_long")
    assert data_ode_long.output_cols == [
        "fdm_t_minus_d_norm",
        "fdm_lift_residual_norm",
    ]
