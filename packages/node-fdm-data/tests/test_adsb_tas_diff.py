"""Tests for AXM-769: U_ODE_COLS and E1_COLS changes for TAS diff.

Validates that:
- U_COLS includes fdm_tas_target_ms (new unified TAS control).
- U_ODE_COLS is reduced to [fdm_vz_sel_ms] only.
- E1_COLS includes fdm_tas_diff_ms (TAS error signal).
- The StructuredLayer (layers[1]) no longer receives Mach/CAS controls.
"""

from __future__ import annotations

from node_fdm_data.schemas import adsb


class TestUColsHasTasTarget:
    """U_COLS must include the new fdm_tas_target_ms control."""

    def test_u_cols_has_tas_target(self) -> None:
        assert "fdm_tas_target_ms" in adsb.U_COLS


class TestUOdeColsVzOnly:
    """U_ODE_COLS should contain only fdm_vz_sel_ms after refactor."""

    def test_u_ode_cols_vz_only(self) -> None:
        assert adsb.U_ODE_COLS == ["fdm_vz_sel_ms"]


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
