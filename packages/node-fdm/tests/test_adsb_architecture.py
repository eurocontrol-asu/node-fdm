"""Tests for node_adsb_v1 architecture registration."""

from __future__ import annotations

from node_fdm.architectures.registry import get


class TestNodeAdsbV1Architecture:
    """Tests for node_adsb_v1 architecture auto-registration."""

    def test_node_adsb_v1_registered(self) -> None:
        """'node_adsb_v1' is registered and resolvable."""
        import node_fdm_models.architectures.adsb  # noqa: F401

        spec = get("node_adsb_v1")
        assert spec.name == "node_adsb_v1"

    def test_x_cols_no_distance(self) -> None:
        """State vector has 4 variables (alt, gamma, tas, heading) — no distance."""
        spec = get("node_adsb_v1")
        assert len(spec.x_cols) == 4
        assert "fdm_distance_cum_m" not in spec.x_cols
        assert "fdm_heading_rad" in spec.x_cols

    def test_u_cols_alt_target(self) -> None:
        """Control uses fdm_alt_target_m instead of fdm_mcp_alt_sel_m."""
        spec = get("node_adsb_v1")
        assert spec.u_cols[0] == "fdm_alt_target_m"
        assert "fdm_mcp_alt_sel_m" not in spec.u_cols

    def test_e0_cols_lean(self) -> None:
        """Environment has 4 variables (long_wind, temp, u_wind, v_wind)."""
        spec = get("node_adsb_v1")
        assert len(spec.e0_cols) == 4
        assert "era_u_wind_ms" in spec.e0_cols
        assert "era_v_wind_ms" in spec.e0_cols

    def test_dx_cols_no_gs(self) -> None:
        """Derivatives do not include raw_gs_ms; lateral d_heading is present."""
        spec = get("node_adsb_v1")
        dx_names = [name for _, name in spec.dx_cols]
        assert "raw_gs_ms" not in dx_names
        assert "fdm_d_heading_rads" in dx_names
        assert len(spec.dx_cols) == 4

    def test_four_layers(self) -> None:
        """Architecture has trajectory + data_ode_long + data_ode_lat + physics."""
        spec = get("node_adsb_v1")
        assert len(spec.layers) == 4
        assert spec.layers[0].name == "trajectory"
        assert spec.layers[1].name == "data_ode_long"
        assert spec.layers[2].name == "data_ode_lat"
        assert spec.layers[3].name == "physics"

    def test_data_ode_long_outputs_aero_quantities(self) -> None:
        """data_ode_long outputs (a_spec, n_z_residual)."""
        spec = get("node_adsb_v1")
        assert spec.layers[1].output_cols == [
            "fdm_a_spec_ms2",
            "fdm_n_z_residual",
        ]

    def test_data_ode_lat_outputs_phi_bank(self) -> None:
        """data_ode_lat outputs phi_bank for the lateral channel."""
        spec = get("node_adsb_v1")
        assert spec.layers[2].output_cols == ["fdm_phi_bank_rad"]

    def test_physics_outputs_dx(self) -> None:
        """PhysicsLayer outputs the ODE derivatives consumed by dx_cols."""
        spec = get("node_adsb_v1")
        assert spec.layers[3].output_cols == [
            "fdm_d_tas_ms2",
            "fdm_d_gamma_rads",
            "fdm_n_z",
            "fdm_d_heading_rads",
        ]
        assert spec.layers[3].trainable is False

    def test_col_map_alt_sel(self) -> None:
        """TrajectoryLayer col_map uses fdm_alt_target_m for alt_sel."""
        spec = get("node_adsb_v1")
        raw = spec.layers[0].config["col_map"]
        assert isinstance(raw, dict)
        assert raw["alt_sel"] == "fdm_alt_target_m"


class TestAdsbStructuredInputCols:
    """Tests for data_ode_long layer input column composition."""

    def test_data_ode_long_input_has_g_sin_gamma(self) -> None:
        """data_ode_long input_cols contains fdm_g_sin_gamma_ms2."""
        spec = get("node_adsb_v1")
        assert "fdm_g_sin_gamma_ms2" in spec.layers[1].input_cols

    def test_data_ode_long_input_has_cos_gamma(self) -> None:
        """data_ode_long input_cols contains fdm_cos_gamma."""
        spec = get("node_adsb_v1")
        assert "fdm_cos_gamma" in spec.layers[1].input_cols

    def test_data_ode_long_input_has_q_pa(self) -> None:
        """data_ode_long input_cols contains fdm_q_pa (dynamic pressure)."""
        spec = get("node_adsb_v1")
        assert "fdm_q_pa" in spec.layers[1].input_cols

    def test_data_ode_long_input_has_g_over_v(self) -> None:
        """data_ode_long input_cols contains fdm_g_over_v (g/V ratio)."""
        spec = get("node_adsb_v1")
        assert "fdm_g_over_v" in spec.layers[1].input_cols


class TestAdsbColMapV2:
    """Tests for AXM-808: gamma_sel col_map points to fdm_gamma_target_rad."""

    def test_col_map_gamma_sel_targets_gamma_target(self) -> None:
        """TrajectoryLayer col_map maps gamma_sel → fdm_gamma_target_rad."""
        spec = get("node_adsb_v1")
        col_map = spec.layers[0].config["col_map"]
        assert isinstance(col_map, dict)
        assert col_map["gamma_sel"] == "fdm_gamma_target_rad"

    def test_u_cols_contains_gamma_target(self) -> None:
        """Registered spec u_cols contains fdm_gamma_target_rad."""
        spec = get("node_adsb_v1")
        assert "fdm_gamma_target_rad" in spec.u_cols

    def test_u_cols_no_gamma_sel_rad(self) -> None:
        """Registered spec u_cols does NOT contain fdm_gamma_sel_rad."""
        spec = get("node_adsb_v1")
        assert "fdm_gamma_sel_rad" not in spec.u_cols
