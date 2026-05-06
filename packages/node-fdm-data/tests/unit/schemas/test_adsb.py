"""Tests for ADS-B v1 schema column lists."""

from __future__ import annotations

import pytest

from node_fdm_data.schemas import adsb


class TestAdsbSchema:
    """Tests for the adsb schema column lists."""

    def test_x_cols_no_distance(self) -> None:
        """X_COLS removes fdm_distance_cum_m, keeps 4 state variables (Phase 2B)."""
        assert "fdm_distance_cum_m" not in adsb.X_COLS
        assert len(adsb.X_COLS) == 4

    def test_x_cols_content(self) -> None:
        """X_COLS contains the expected state variables incl. heading (Phase 2B)."""
        assert adsb.X_COLS == [
            "raw_alt_m",
            "fdm_gamma_rad",
            "era_tas_ms",
            "fdm_heading_rad",
        ]

    def test_u_cols_alt_target(self) -> None:
        """U_COLS[0] is fdm_alt_target_m (robust, never NaN)."""
        assert adsb.U_COLS[0] == "fdm_alt_target_m"

    def test_e0_cols_no_airport_dist(self) -> None:
        """E0_COLS has 4 variables (long_wind, temp, u_wind, v_wind); no airport distances."""
        assert len(adsb.E0_COLS) == 4
        assert "fdm_adep_dist_m" not in adsb.E0_COLS
        assert "fdm_ades_dist_m" not in adsb.E0_COLS
        assert "era_u_wind_ms" in adsb.E0_COLS
        assert "era_v_wind_ms" in adsb.E0_COLS

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
        """DX_COLS has 4 derivatives (long + lateral d_heading) with correct signs."""
        assert adsb.DX_COLS == [
            (1, "fdm_d_alt_ms"),
            (1, "fdm_d_gamma_rads"),
            (1, "fdm_d_tas_ms2"),
            (1, "fdm_d_heading_rads"),
        ]


class TestUColsV2:
    """Tests for U_COLS after AXM-805 (gamma_target replaces vz_sel)."""

    def test_u_cols_has_gamma_target(self) -> None:
        """U_COLS contains fdm_gamma_target_rad as control input."""
        assert "fdm_gamma_target_rad" in adsb.U_COLS


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
        assert not ode_inputs & raw_controls, (
            f"Raw controls leaked into ODE inputs: {ode_inputs & raw_controls}"
        )


class TestUColsV3:
    """Tests for U_COLS after AXM-808 (fdm_gamma_sel_rad → fdm_gamma_target_rad)."""

    @pytest.mark.parametrize(
        "removed_col",
        [
            pytest.param("fdm_vz_sel_ms", id="axm_805_vz_sel"),
            pytest.param("fdm_gamma_sel_rad", id="axm_808_gamma_sel"),
        ],
    )
    def test_u_cols_removed_legacy_columns(self, removed_col: str) -> None:
        """U_COLS no longer contains legacy raw-control columns."""
        assert removed_col not in adsb.U_COLS

    def test_u_cols_all_target_convention(self) -> None:
        """All U_COLS entries use either ``_target_`` or ``_known`` naming.

        Phase 2B adds ``fdm_heading_known`` (a state-quality flag, not a
        target flag) — kept under ``_known`` for the wrap-aware loss to
        potentially mask catastrophic samples downstream.
        """
        for col in adsb.U_COLS:
            assert "_target_" in col or col.endswith("_known"), (
                f"U_COLS entry {col!r} does not follow _target_/_known convention"
            )

    def test_u_cols_length(self) -> None:
        """U_COLS has 8 entries: 4 targets + 3 target_known flags + heading_known."""
        assert len(adsb.U_COLS) == 8


# ---------------------------------------------------------------------------
# AXM-769: U_ODE_COLS and E1_COLS changes for TAS diff
# ---------------------------------------------------------------------------


class TestUColsHasTasTarget:
    """U_COLS must include the new fdm_tas_target_ms control."""

    def test_u_cols_has_tas_target(self) -> None:
        assert "fdm_tas_target_ms" in adsb.U_COLS


class TestUOdeColsEmpty:
    """U_ODE_COLS is empty — no direct control feeds the ODE (AXM-805)."""

    def test_u_ode_cols_empty(self) -> None:
        assert adsb.U_ODE_COLS == []


class TestE1ColsHasTasDiff:
    """E1_COLS must include the new fdm_tas_diff_ms error signal."""

    def test_e1_cols_has_tas_diff(self) -> None:
        assert "fdm_tas_diff_ms" in adsb.E1_COLS


class TestStructuredNoMachCas:
    """StructuredLayer (layers[1]) must not receive Mach/CAS controls."""

    def test_structured_no_mach_cas(self) -> None:
        from node_fdm.architectures.adsb import NODE_ADSB_V1

        input_cols = NODE_ADSB_V1.layers[1].input_cols
        assert "fdm_mach_sel" not in input_cols
        assert "fdm_cas_sel_ms" not in input_cols
