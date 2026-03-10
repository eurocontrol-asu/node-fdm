"""Tests for BADA predictor — primarily import validation.

Full functional tests require pyBADA (proprietary) which is not
available in CI.
"""

from __future__ import annotations

from node_fdm_bada.predictor import _HAS_PYBADA, process_single_flight


class TestPredictor:
    """Tests for the BADA predictor module."""

    def test_no_pybada_returns_none(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """If pyBADA is not installed, process_single_flight returns None."""
        if _HAS_PYBADA:
            # Can't test graceful fallback when pyBADA is actually installed
            return

        result = process_single_flight(
            flight_path=tmp_path / "dummy.parquet",
            ac=None,
        )
        assert result is None

    def test_module_imports(self) -> None:
        """Verify the module can be imported without pyBADA."""
        assert callable(process_single_flight)
