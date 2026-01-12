"""Golden tests for data equivalence between pandas and polars.

These tests compare the refactored Polars code against frozen
outputs from the legacy pandas code.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

GOLDEN_DIR = Path(__file__).parent / "outputs"
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def assert_arrays_approx_equal(
    result: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-6,
    column_name: str = "",
) -> None:
    """Assert two arrays are approximately equal with tolerance."""
    np.testing.assert_allclose(
        result,
        expected,
        rtol=rtol,
        err_msg=f"Mismatch in column '{column_name}'",
    )


@pytest.mark.golden
class TestRawDataEquivalence:
    """Verify raw data loading produces identical results."""

    @pytest.fixture
    def golden_short(self) -> pl.DataFrame:
        """Load golden raw output for short flight."""
        return pl.read_parquet(GOLDEN_DIR / "raw_flight_short.parquet")

    @pytest.fixture
    def golden_medium(self) -> pl.DataFrame:
        """Load golden raw output for medium flight."""
        return pl.read_parquet(GOLDEN_DIR / "raw_flight_medium.parquet")

    @pytest.fixture
    def fixture_short(self) -> pl.DataFrame:
        """Load fixture for short flight."""
        return pl.read_parquet(FIXTURES_DIR / "flight_short.parquet")

    @pytest.fixture
    def fixture_medium(self) -> pl.DataFrame:
        """Load fixture for medium flight."""
        return pl.read_parquet(FIXTURES_DIR / "flight_medium.parquet")

    def test_polars_reads_parquet_identically_short(
        self, fixture_short: pl.DataFrame, golden_short: pl.DataFrame
    ) -> None:
        """Polars reads the same values as pandas wrote for short flight."""
        for col in golden_short.columns:
            result = fixture_short.get_column(col).to_numpy()
            expected = golden_short.get_column(col).to_numpy()
            assert_arrays_approx_equal(result, expected, column_name=col)

    def test_polars_reads_parquet_identically_medium(
        self, fixture_medium: pl.DataFrame, golden_medium: pl.DataFrame
    ) -> None:
        """Polars reads the same values as pandas wrote for medium flight."""
        for col in golden_medium.columns:
            result = fixture_medium.get_column(col).to_numpy()
            expected = golden_medium.get_column(col).to_numpy()
            assert_arrays_approx_equal(result, expected, column_name=col)


@pytest.mark.golden
class TestStatisticsEquivalence:
    """Verify statistics computation matches legacy."""

    @pytest.fixture
    def golden_stats(self) -> dict:
        """Load golden statistics."""
        with open(GOLDEN_DIR / "raw_stats.json") as f:
            return json.load(f)

    def test_mean_matches(self, golden_stats: dict) -> None:
        """Polars computes identical means as pandas."""
        # Load all fixtures and compute stats with Polars
        flights = sorted(FIXTURES_DIR.glob("flight_*.parquet"))
        dfs = [pl.read_parquet(p) for p in flights]
        combined = pl.concat(dfs)

        for col_name, expected in golden_stats.items():
            result_mean = combined.get_column(col_name).mean()
            np.testing.assert_allclose(
                result_mean,
                expected["mean"],
                rtol=1e-6,
                err_msg=f"Mean mismatch for {col_name}",
            )

    def test_std_matches(self, golden_stats: dict) -> None:
        """Polars computes identical std as pandas."""
        flights = sorted(FIXTURES_DIR.glob("flight_*.parquet"))
        dfs = [pl.read_parquet(p) for p in flights]
        combined = pl.concat(dfs)

        for col_name, expected in golden_stats.items():
            # Polars uses ddof=1 by default, same as pandas
            result_std = combined.get_column(col_name).std()
            # Use atol for cases where expected is 0 (constant values)
            np.testing.assert_allclose(
                result_std,
                expected["std"],
                rtol=1e-5,
                atol=1e-15,  # Handle floating point noise near zero
                err_msg=f"Std mismatch for {col_name}",
            )

    def test_min_max_matches(self, golden_stats: dict) -> None:
        """Polars computes identical min/max as pandas."""
        flights = sorted(FIXTURES_DIR.glob("flight_*.parquet"))
        dfs = [pl.read_parquet(p) for p in flights]
        combined = pl.concat(dfs)

        for col_name, expected in golden_stats.items():
            result_min = combined.get_column(col_name).min()
            result_max = combined.get_column(col_name).max()

            np.testing.assert_allclose(
                result_min,
                expected["min"],
                rtol=1e-6,
                err_msg=f"Min mismatch for {col_name}",
            )
            np.testing.assert_allclose(
                result_max,
                expected["max"],
                rtol=1e-6,
                err_msg=f"Max mismatch for {col_name}",
            )
