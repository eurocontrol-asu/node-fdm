"""Tests for node_fdm_data.schemas.opensky — OpenSky 2025 column schema."""

from __future__ import annotations

from node_fdm_data.schemas import opensky


class TestOpenSkySchema:
    """OpenSky 2025 column schema."""

    def test_x_cols_non_empty(self) -> None:
        assert isinstance(opensky.X_COLS, list)
        assert len(opensky.X_COLS) > 0
        assert all(isinstance(c, str) for c in opensky.X_COLS)

    def test_u_cols_non_empty(self) -> None:
        assert isinstance(opensky.U_COLS, list)
        assert len(opensky.U_COLS) > 0
        assert all(isinstance(c, str) for c in opensky.U_COLS)

    def test_e0_cols_non_empty(self) -> None:
        assert isinstance(opensky.E0_COLS, list)
        assert len(opensky.E0_COLS) > 0
        assert all(isinstance(c, str) for c in opensky.E0_COLS)

    def test_e1_cols_non_empty(self) -> None:
        assert isinstance(opensky.E1_COLS, list)
        assert len(opensky.E1_COLS) > 0
        assert all(isinstance(c, str) for c in opensky.E1_COLS)

    def test_dx_cols_non_empty(self) -> None:
        assert isinstance(opensky.DX_COLS, list)
        assert len(opensky.DX_COLS) > 0
        for sign, name in opensky.DX_COLS:
            assert isinstance(sign, int)
            assert isinstance(name, str)

    def test_si_unit_convention(self) -> None:
        """All columns use SI unit suffixes (m, ms, rad, K)."""
        all_names = opensky.X_COLS + opensky.U_COLS + opensky.E0_COLS + opensky.E1_COLS
        # No non-SI suffixes should remain
        non_si = [c for c in all_names if c.endswith(("_ft", "_kt", "_ftmin", "_nm"))]
        assert non_si == [], f"Non-SI columns found: {non_si}"
