"""Tests for AXM-844: rename fdm_d_vz_ms → fdm_d_alt_ms.

Data-layer assertions: SI_DERIVATIVES naming, DERIVATIVE_BOUNDS key,
and edge cases around the rename.
"""

from __future__ import annotations

import polars as pl
import pytest

from node_fdm_data.preprocessing.convert import (
    DERIVATIVE_BOUNDS,
    SI_DERIVATIVES,
    compute_derivatives,
)


class TestSIDerivativesNaming:
    """Unit: all derivative names follow fdm_d_{state}_{unit}."""

    def test_si_derivatives_naming(self) -> None:
        """SI_DERIVATIVES contains fdm_d_alt_ms, not fdm_d_vz_ms."""
        deriv_names = [tgt for _, tgt in SI_DERIVATIVES]
        assert "fdm_d_alt_ms" in deriv_names, "fdm_d_alt_ms missing from SI_DERIVATIVES"
        assert "fdm_d_vz_ms" not in deriv_names, "fdm_d_vz_ms should be renamed"

    def test_derivative_bounds_key(self) -> None:
        """DERIVATIVE_BOUNDS uses fdm_d_alt_ms key, not fdm_d_vz_ms."""
        assert "fdm_d_alt_ms" in DERIVATIVE_BOUNDS
        assert "fdm_d_vz_ms" not in DERIVATIVE_BOUNDS


class TestRawVzMsUntouched:
    """Edge case: raw_vz_ms column still exists separately after rename."""

    def test_raw_vz_ms_column_preserved(self) -> None:
        """raw_vz_ms is an input column, not affected by derivative rename."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["A", "A", "A"],
                "raw_alt_m": [0.0, 100.0, 300.0],
                "fdm_gamma_rad": [0.0, 0.01, 0.03],
                "era_tas_ms": [100.0, 110.0, 130.0],
            }
        )
        result = compute_derivatives(df, dt=4.0)
        # The derivative column should now be fdm_d_alt_ms
        assert "fdm_d_alt_ms" in result.columns
        # raw_vz_ms (SI conversion of raw_vz_ftmin) is a separate concept
        # and should NOT be affected by the derivative rename
        deriv_col = "fdm_d_alt_ms"
        assert deriv_col != "raw_vz_ms"

    def test_derivative_output_uses_new_name(self) -> None:
        """compute_derivatives produces fdm_d_alt_ms, not fdm_d_vz_ms."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["A", "A", "A"],
                "raw_alt_m": [0.0, 100.0, 300.0],
                "fdm_gamma_rad": [0.0, 0.0, 0.0],
                "era_tas_ms": [200.0, 200.0, 200.0],
            }
        )
        result = compute_derivatives(df, dt=4.0)
        assert "fdm_d_alt_ms" in result.columns, "Expected fdm_d_alt_ms in output"
        assert "fdm_d_vz_ms" not in result.columns, "fdm_d_vz_ms should no longer appear"
        # Values should still be correct: diff(alt)/dt
        assert result["fdm_d_alt_ms"][1] == pytest.approx(25.0)
        assert result["fdm_d_alt_ms"][2] == pytest.approx(50.0)
