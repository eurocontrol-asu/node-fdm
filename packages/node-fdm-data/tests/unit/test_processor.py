"""Tests for node_fdm_data.processor — FlightProcessor pipeline."""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from node_fdm_data.processor import FlightProcessor


class TestFlightProcessor:
    """FlightProcessor pipeline tests."""

    @pytest.fixture()
    def sample_df(self) -> pl.LazyFrame:
        """Small LazyFrame fixture for testing."""
        return pl.LazyFrame(
            {
                "altitude_ft": [100.0, 200.0, 300.0, 400.0],
                "tas_kt": [250.0, 260.0, 270.0, 280.0],
                "distance_m": [0.0, 1000.0, 2000.0, 3000.0],
            }
        )

    def test_processor_identity(self, sample_df: pl.LazyFrame) -> None:
        """Empty transform list → output equals input."""
        proc = FlightProcessor()
        result = proc.process(sample_df).collect()
        expected = sample_df.collect()
        assert_frame_equal(result, expected)

    def test_processor_chain(self, sample_df: pl.LazyFrame) -> None:
        """Two transforms (rename + filter) applied in order."""

        def rename_step(lf: pl.LazyFrame) -> pl.LazyFrame:
            return lf.with_columns(pl.col("altitude_ft").alias("alt_ft")).drop("altitude_ft")

        def filter_step(lf: pl.LazyFrame) -> pl.LazyFrame:
            return lf.filter(pl.col("alt_ft") > 150.0)

        proc = FlightProcessor(steps=[rename_step, filter_step])
        result = proc.process(sample_df).collect()

        assert "alt_ft" in result.columns
        assert "altitude_ft" not in result.columns
        assert len(result) == 3
        assert result["alt_ft"].to_list() == [200.0, 300.0, 400.0]

    def test_processor_add_step_chaining(self) -> None:
        """add_step returns self for fluent API."""
        proc = FlightProcessor()
        result = proc.add_step(lambda lf: lf)
        assert result is proc

    def test_processor_eager_dataframe(self) -> None:
        """Processor auto-converts eager DataFrame to LazyFrame."""
        eager = pl.DataFrame({"x": [1, 2, 3]})
        proc = FlightProcessor(steps=[lambda lf: lf.filter(pl.col("x") > 1)])
        result = proc.process(eager)
        collected = result.collect()
        assert len(collected) == 2

    def test_processor_multiple_add_steps(self, sample_df: pl.LazyFrame) -> None:
        """Multiple add_step calls accumulate transforms."""
        proc = (
            FlightProcessor()
            .add_step(lambda lf: lf.with_columns((pl.col("tas_kt") * 2).alias("tas_x2")))
            .add_step(lambda lf: lf.filter(pl.col("altitude_ft") >= 200.0))
        )
        result = proc.process(sample_df).collect()
        assert "tas_x2" in result.columns
        assert len(result) == 3
