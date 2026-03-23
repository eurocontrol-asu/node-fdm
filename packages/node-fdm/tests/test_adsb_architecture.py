"""Tests for node_adsb_v1 architecture registration."""

from __future__ import annotations

from node_fdm.architectures.registry import get


class TestNodeAdsbV1Architecture:
    """Tests for node_adsb_v1 architecture auto-registration."""

    def test_node_adsb_v1_registered(self) -> None:
        """'node_adsb_v1' is registered and resolvable."""
        import node_fdm.architectures.adsb  # noqa: F401

        spec = get("node_adsb_v1")
        assert spec.name == "node_adsb_v1"

    def test_x_cols_no_distance(self) -> None:
        """State vector has 3 variables (no cumulative distance)."""
        spec = get("node_adsb_v1")
        assert len(spec.x_cols) == 3
        assert "fdm_distance_cum_m" not in spec.x_cols

    def test_u_cols_alt_target(self) -> None:
        """Control uses fdm_alt_target_m instead of fdm_mcp_alt_sel_m."""
        spec = get("node_adsb_v1")
        assert spec.u_cols[0] == "fdm_alt_target_m"
        assert "fdm_mcp_alt_sel_m" not in spec.u_cols

    def test_e0_cols_lean(self) -> None:
        """Environment has 2 variables (no airport distances)."""
        spec = get("node_adsb_v1")
        assert len(spec.e0_cols) == 2

    def test_dx_cols_no_gs(self) -> None:
        """Derivatives do not include raw_gs_ms."""
        spec = get("node_adsb_v1")
        dx_names = [name for _, name in spec.dx_cols]
        assert "raw_gs_ms" not in dx_names
        assert len(spec.dx_cols) == 3

    def test_two_layers(self) -> None:
        """Architecture has trajectory + data_ode layers."""
        spec = get("node_adsb_v1")
        assert len(spec.layers) == 2
        assert spec.layers[0].name == "trajectory"
        assert spec.layers[1].name == "data_ode"

    def test_col_map_alt_sel(self) -> None:
        """TrajectoryLayer col_map uses fdm_alt_target_m for alt_sel."""
        spec = get("node_adsb_v1")
        raw = spec.layers[0].config["col_map"]
        assert isinstance(raw, dict)
        assert raw["alt_sel"] == "fdm_alt_target_m"

    def test_opensky_2025_unchanged(self) -> None:
        """opensky_2025 is still registered and untouched."""
        import node_fdm.architectures.opensky  # noqa: F401

        spec = get("opensky_2025")
        assert spec.name == "opensky_2025"
        assert len(spec.x_cols) == 4
        assert spec.u_cols[0] == "fdm_mcp_alt_sel_m"


class TestAdsbStructuredInputCols:
    """Tests for data_ode layer input column composition."""

    def test_adsb_structured_input_cols(self) -> None:
        """data_ode layer input_cols == X_COLS + U_ODE_COLS + E0_COLS + E1_COLS."""
        from node_fdm_data.schemas.adsb import E0_COLS, E1_COLS, U_ODE_COLS, X_COLS

        spec = get("node_adsb_v1")
        expected = X_COLS + U_ODE_COLS + E0_COLS + E1_COLS
        assert spec.layers[1].input_cols == expected


class TestBothArchitecturesCoexist:
    """Both architectures loaded without conflict."""

    def test_both_registered(self) -> None:
        """Both opensky_2025 and node_adsb_v1 are in the registry."""
        import node_fdm.architectures.adsb
        import node_fdm.architectures.opensky  # noqa: F401

        opensky_spec = get("opensky_2025")
        adsb_spec = get("node_adsb_v1")
        assert opensky_spec.name != adsb_spec.name
