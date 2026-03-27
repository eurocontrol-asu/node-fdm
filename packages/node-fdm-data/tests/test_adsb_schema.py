"""Tests for ADS-B v1 schema column lists."""

from __future__ import annotations

from node_fdm_data.schemas import adsb, opensky


class TestAdsbSchema:
    """Tests for the adsb schema simplifications vs opensky_2025."""

    def test_x_cols_no_distance(self) -> None:
        """X_COLS removes fdm_distance_cum_m, keeps 3 state variables."""
        assert "fdm_distance_cum_m" not in adsb.X_COLS
        assert len(adsb.X_COLS) == 3

    def test_x_cols_content(self) -> None:
        """X_COLS contains the expected state variables."""
        assert adsb.X_COLS == ["raw_alt_m", "fdm_gamma_rad", "era_tas_ms"]

    def test_u_cols_alt_target(self) -> None:
        """U_COLS[0] is fdm_alt_target_m (robust, never NaN)."""
        assert adsb.U_COLS[0] == "fdm_alt_target_m"

    def test_e0_cols_no_airport_dist(self) -> None:
        """E0_COLS has 2 variables, no airport distances."""
        assert len(adsb.E0_COLS) == 2
        assert "fdm_adep_dist_m" not in adsb.E0_COLS
        assert "fdm_ades_dist_m" not in adsb.E0_COLS

    def test_e1_cols_has_error_signals(self) -> None:
        """E1_COLS contains diff error signals and fdm_cas_ms (not bds_ias_ms)."""
        assert "fdm_tas_diff_ms" in adsb.E1_COLS
        assert "fdm_gamma_diff_rad" in adsb.E1_COLS
        assert "fdm_cas_ms" in adsb.E1_COLS
        assert "bds_ias_ms" not in adsb.E1_COLS

    def test_dx_cols_no_gs(self) -> None:
        """DX_COLS does not include raw_gs_ms."""
        dx_names = [name for _, name in adsb.DX_COLS]
        assert "raw_gs_ms" not in dx_names

    def test_dx_cols_content(self) -> None:
        """DX_COLS has 3 derivatives with correct signs."""
        assert adsb.DX_COLS == [
            (1, "fdm_d_alt_ms"),
            (1, "fdm_d_gamma_rads"),
            (1, "fdm_d_tas_ms"),
        ]


class TestUColsV2:
    """Tests for U_COLS after AXM-805 (gamma_target replaces vz_sel)."""

    def test_u_cols_has_gamma_target(self) -> None:
        """U_COLS contains fdm_gamma_target_rad as control input."""
        assert "fdm_gamma_target_rad" in adsb.U_COLS

    def test_u_cols_no_vz_sel(self) -> None:
        """U_COLS no longer contains fdm_vz_sel_ms."""
        assert "fdm_vz_sel_ms" not in adsb.U_COLS


class TestUOdeCols:
    """Tests for the U_ODE_COLS subset used by the ODE layer."""

    def test_u_ode_cols_empty(self) -> None:
        """U_ODE_COLS is empty — no direct control feeds the ODE."""
        assert adsb.U_ODE_COLS == []


class TestE1ColsV2:
    """Tests for E1_COLS after AXM-805 (gamma_diff added)."""

    def test_e1_cols_has_gamma_diff(self) -> None:
        """E1_COLS contains fdm_gamma_diff_rad."""
        assert "fdm_gamma_diff_rad" in adsb.E1_COLS


class TestStructuredInputs:
    """Tests that ODE inputs contain no raw absolute control."""

    def test_structured_no_raw_control(self) -> None:
        """U_ODE_COLS + E1_COLS must not contain absolute control columns."""
        raw_controls = {"fdm_vz_sel_ms", "fdm_alt_target_m", "fdm_tas_target_ms"}
        ode_inputs = set(adsb.U_ODE_COLS) | set(adsb.E1_COLS)
        assert (
            not ode_inputs & raw_controls
        ), f"Raw controls leaked into ODE inputs: {ode_inputs & raw_controls}"


class TestUColsV3:
    """Tests for U_COLS after AXM-808 (fdm_gamma_sel_rad → fdm_gamma_target_rad)."""

    def test_u_cols_no_gamma_sel(self) -> None:
        """U_COLS no longer contains fdm_gamma_sel_rad."""
        assert "fdm_gamma_sel_rad" not in adsb.U_COLS

    def test_u_cols_all_target_convention(self) -> None:
        """All U_COLS entries use the _target_ naming convention."""
        for col in adsb.U_COLS:
            assert "_target_" in col, f"U_COLS entry {col!r} does not follow _target_ convention"

    def test_u_cols_length(self) -> None:
        """U_COLS has 4 entries: 3 targets + gamma_known mask."""
        assert len(adsb.U_COLS) == 4


class TestBothSchemasCoexist:
    """Both schemas can be loaded without conflict."""

    def test_both_loaded(self) -> None:
        """Importing both schemas does not raise."""
        assert len(opensky.X_COLS) == 4
        assert len(adsb.X_COLS) == 3
