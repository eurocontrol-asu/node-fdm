"""Tests for node_fdm_data.schemas — column lists and conversion registries."""

from __future__ import annotations

import polars as pl

from node_fdm_data.schemas import opensky, qar


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

    def test_conversions_registry(self) -> None:
        """All keys are strings, all values are (callable, str) tuples."""
        assert isinstance(opensky.CONVERSIONS, dict)
        assert len(opensky.CONVERSIONS) > 0
        for key, (fn, target) in opensky.CONVERSIONS.items():
            assert isinstance(key, str)
            assert callable(fn)
            assert isinstance(target, str)

    def test_conversion_fn_returns_expr(self) -> None:
        """Each conversion function returns a pl.Expr."""
        for _key, (fn, _target) in opensky.CONVERSIONS.items():
            result = fn("test_col")
            assert isinstance(result, pl.Expr)


class TestQarSchema:
    """QAR column schema."""

    def test_x_cols_non_empty(self) -> None:
        assert isinstance(qar.X_COLS, list)
        assert len(qar.X_COLS) > 0
        assert all(isinstance(c, str) for c in qar.X_COLS)

    def test_u_cols_non_empty(self) -> None:
        assert isinstance(qar.U_COLS, list)
        assert len(qar.U_COLS) > 0
        assert all(isinstance(c, str) for c in qar.U_COLS)

    def test_e0_cols_non_empty(self) -> None:
        assert isinstance(qar.E0_COLS, list)
        assert len(qar.E0_COLS) > 0

    def test_e1_cols_non_empty(self) -> None:
        assert isinstance(qar.E1_COLS, list)
        assert len(qar.E1_COLS) > 0

    def test_dx_cols_non_empty(self) -> None:
        assert isinstance(qar.DX_COLS, list)
        assert len(qar.DX_COLS) > 0
        for sign, name in qar.DX_COLS:
            assert isinstance(sign, int)
            assert isinstance(name, str)

    def test_conversions_registry(self) -> None:
        """All keys are strings, all values are (callable, str) tuples."""
        assert isinstance(qar.CONVERSIONS, dict)
        assert len(qar.CONVERSIONS) > 0
        for key, (fn, target) in qar.CONVERSIONS.items():
            assert isinstance(key, str)
            assert callable(fn)
            assert isinstance(target, str)

    def test_conversion_fn_returns_expr(self) -> None:
        """Each conversion function returns a pl.Expr."""
        for _key, (fn, _target) in qar.CONVERSIONS.items():
            result = fn("test_col")
            assert isinstance(result, pl.Expr)
