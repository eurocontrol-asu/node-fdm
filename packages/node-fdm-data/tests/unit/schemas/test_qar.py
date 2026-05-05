"""Tests for node_fdm_data.schemas.qar — QAR column schema."""

from __future__ import annotations

import polars as pl

from node_fdm_data.schemas import qar


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

    def test_conversion_fn_evaluates_to_finite_float(self) -> None:
        """Each conversion expression evaluates to a finite float column."""
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        for _key, (fn, _target) in qar.CONVERSIONS.items():
            out = df.select(fn("x").alias("y"))["y"]
            assert out.dtype.is_float()
            assert out.is_finite().all()
            assert out.len() == 3
