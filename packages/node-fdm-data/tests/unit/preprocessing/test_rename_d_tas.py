"""Tests for AXM-846: rename fdm_d_tas_ms → fdm_d_tas_ms2.

Data-layer assertions: SI_DERIVATIVES naming, DERIVATIVE_BOUNDS key,
DX_COLS unit suffix, and edge cases around the rename.
"""

from __future__ import annotations

import polars as pl

from node_fdm_data.preprocessing.convert import (
    DERIVATIVE_BOUNDS,
    SI_DERIVATIVES,
    compute_derivatives,
)
from node_fdm_data.schemas.adsb import DX_COLS

# -- Expected unit suffixes for derivative columns --
# d(alt)/dt   → m/s   → _ms
# d(gamma)/dt → rad/s → _rads
# d(tas)/dt   → m/s²  → _ms2
EXPECTED_SUFFIXES: dict[str, str] = {
    "fdm_d_alt_ms": "_ms",
    "fdm_d_gamma_rads": "_rads",
    "fdm_d_tas_ms2": "_ms2",
    # Phase 2B — lateral channel.
    "fdm_d_heading_rads": "_rads",
}


class TestDerivativeColumnUnitSuffix:
    """Unit: all derivative columns have suffixes matching their SI unit."""

    def test_si_derivatives_tas_renamed(self) -> None:
        """SI_DERIVATIVES contains fdm_d_tas_ms2, not fdm_d_tas_ms."""
        deriv_names = [tgt for _, tgt in SI_DERIVATIVES]
        assert "fdm_d_tas_ms2" in deriv_names, "fdm_d_tas_ms2 missing from SI_DERIVATIVES"
        assert "fdm_d_tas_ms" not in deriv_names, "fdm_d_tas_ms should be renamed to fdm_d_tas_ms2"

    def test_derivative_bounds_tas_key(self) -> None:
        """DERIVATIVE_BOUNDS uses fdm_d_tas_ms2 key, not fdm_d_tas_ms."""
        assert "fdm_d_tas_ms2" in DERIVATIVE_BOUNDS
        assert "fdm_d_tas_ms" not in DERIVATIVE_BOUNDS

    def test_dx_cols_suffix_matches_unit(self) -> None:
        """Every DX_COLS entry has a suffix that matches its actual SI unit."""
        dx_names = [name for _, name in DX_COLS]
        for name in dx_names:
            matched = False
            for expected_name, suffix in EXPECTED_SUFFIXES.items():
                if name == expected_name:
                    assert name.endswith(suffix), f"{name} should end with {suffix}"
                    matched = True
                    break
            assert matched, f"Unexpected derivative column {name} — update EXPECTED_SUFFIXES"


class TestAdsbDxColsNaming:
    """Functional: node_adsb_v1 spec uses correctly-suffixed DX_COL names."""

    def test_adsb_dx_cols_naming(self) -> None:
        """All DX_COL names in adsb schema have correct unit suffix."""
        dx_names = [name for _, name in DX_COLS]
        # TAS derivative must reflect m/s² unit
        assert "fdm_d_tas_ms2" in dx_names
        assert "fdm_d_tas_ms" not in dx_names
        # Other derivatives unchanged
        assert "fdm_d_alt_ms" in dx_names
        assert "fdm_d_gamma_rads" in dx_names

    def test_node_adsb_v1_dx_cols(self) -> None:
        """NODE_ADSB_V1 architecture spec picks up the renamed DX_COLS."""
        import node_fdm.architectures.adsb  # noqa: F401
        from node_fdm.architectures.registry import get

        spec = get("node_adsb_v1")
        dx_names = [name for _, name in spec.dx_cols]
        assert "fdm_d_tas_ms2" in dx_names, "node_adsb_v1 should use fdm_d_tas_ms2"
        assert "fdm_d_tas_ms" not in dx_names


class TestOldDeltaTableCompat:
    """Edge case: data with old fdm_d_tas_ms column name."""

    def test_compute_derivatives_produces_new_name(self) -> None:
        """compute_derivatives outputs fdm_d_tas_ms2, not fdm_d_tas_ms."""
        df = pl.DataFrame(
            {
                "meta_flight_id": ["A", "A", "A"],
                "raw_alt_m": [0.0, 100.0, 300.0],
                "fdm_gamma_rad": [0.0, 0.01, 0.03],
                "era_tas_ms": [100.0, 110.0, 130.0],
            }
        )
        result = compute_derivatives(df, dt=4.0)
        assert "fdm_d_tas_ms2" in result.columns, "New derivative name expected"
        assert "fdm_d_tas_ms" not in result.columns, "Old name should not appear"
