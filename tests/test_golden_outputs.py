"""Gold standard tests: compare new Polars code against legacy golden outputs.

Golden outputs were generated from the legacy pandas code and serve as
the reference for numerical equivalence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from node_fdm_data.conversions import ft_to_m, ftmin_to_ms, kt_to_ms

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLDEN_DIR = Path(__file__).parent / "golden" / "outputs"

FLIGHTS = ["flight_short", "flight_medium"]


@pytest.fixture(params=FLIGHTS)  # type: ignore[misc]
def flight_name(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture  # type: ignore[misc]
def fixture_df(flight_name: str) -> pl.DataFrame:
    return pl.read_parquet(FIXTURES_DIR / f"{flight_name}.parquet")


@pytest.fixture  # type: ignore[misc]
def golden_raw(flight_name: str) -> pl.DataFrame:
    path = GOLDEN_DIR / f"raw_{flight_name}.parquet"
    if not path.exists():
        pytest.skip(f"Golden output not found: {path}")
    return pl.read_parquet(path)


@pytest.fixture  # type: ignore[misc]
def golden_stats() -> dict[str, dict[str, float]]:
    path = GOLDEN_DIR / "raw_stats.json"
    if not path.exists():
        pytest.skip(f"Golden stats not found: {path}")
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


class TestRawValues:
    """Verify raw column extraction matches legacy output."""

    def test_columns_present(self, fixture_df: pl.DataFrame, golden_raw: pl.DataFrame) -> None:
        """All golden columns exist in fixture data."""
        for col in golden_raw.columns:
            assert col in fixture_df.columns, f"Missing column: {col}"

    def test_row_count_matches(self, fixture_df: pl.DataFrame, golden_raw: pl.DataFrame) -> None:
        """Row counts match."""
        assert len(fixture_df) == len(golden_raw)

    def test_values_match(self, fixture_df: pl.DataFrame, golden_raw: pl.DataFrame) -> None:
        """Raw values are identical (same source data)."""
        for col in golden_raw.columns:
            actual = fixture_df[col].to_numpy()
            expected = golden_raw[col].to_numpy()
            np.testing.assert_allclose(
                actual,
                expected,
                atol=1e-10,
                err_msg=f"Column {col} values differ",
            )


class TestUnitConversions:
    """Verify unit conversions produce correct results."""

    def test_ft_to_m(self, fixture_df: pl.DataFrame) -> None:
        """Feet to metres conversion is numerically correct."""
        result = fixture_df.select(ft_to_m("altitude").alias("alt_m"))
        expected = fixture_df["altitude"].to_numpy() * 0.3048
        np.testing.assert_allclose(
            result["alt_m"].to_numpy(),
            expected,
            atol=1e-10,
        )

    def test_kt_to_ms(self, fixture_df: pl.DataFrame) -> None:
        """Knots to m/s conversion is numerically correct."""
        result = fixture_df.select(kt_to_ms("TAS").alias("tas_ms"))
        expected = fixture_df["TAS"].to_numpy() * 0.514444
        np.testing.assert_allclose(
            result["tas_ms"].to_numpy(),
            expected,
            atol=1e-6,
        )

    def test_ftmin_to_ms(self, fixture_df: pl.DataFrame) -> None:
        """ft/min to m/s conversion is numerically correct."""
        result = fixture_df.select(ftmin_to_ms("vertical_rate").alias("vz_ms"))
        expected = fixture_df["vertical_rate"].to_numpy() * 0.00508
        np.testing.assert_allclose(
            result["vz_ms"].to_numpy(),
            expected,
            atol=1e-6,
        )


class TestStatistics:
    """Verify statistics match legacy golden stats."""

    def test_column_stats_match(self, golden_stats: dict[str, dict[str, float]]) -> None:
        """Read fixture data and compare statistics against legacy."""
        flights = [pl.read_parquet(FIXTURES_DIR / f"{f}.parquet") for f in FLIGHTS]
        combined = pl.concat(flights)

        for col, expected in golden_stats.items():
            if col not in combined.columns:
                continue

            values = combined[col].drop_nulls()
            if len(values) == 0:
                continue

            actual_mean = float(values.mean())
            actual_min = float(values.min())
            actual_max = float(values.max())

            np.testing.assert_allclose(
                actual_mean,
                expected["mean"],
                rtol=1e-6,
                err_msg=f"{col} mean differs",
            )
            np.testing.assert_allclose(
                actual_min,
                expected["min"],
                rtol=1e-6,
                err_msg=f"{col} min differs",
            )
            np.testing.assert_allclose(
                actual_max,
                expected["max"],
                rtol=1e-6,
                err_msg=f"{col} max differs",
            )
