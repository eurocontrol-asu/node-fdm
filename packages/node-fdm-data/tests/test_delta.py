"""Tests for node_fdm_data.delta — Delta Lake read/write helpers and schema registry."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Unit Tests — write_columns
# ---------------------------------------------------------------------------


class TestWriteColumnsCreatesTable:
    """Writing columns to an empty directory creates a Delta table."""

    def test_write_columns_creates_table(self, tmp_path: Path) -> None:
        from node_fdm_data.delta import write_columns

        table_path = tmp_path / "flights.delta"
        df = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01", "2025-01-01"],
                "col_a": [1.0, 2.0],
                "col_b": [10.0, 20.0],
            }
        )

        write_columns(df, table_path)

        # Delta table must exist and contain all 3 columns
        result = pl.read_delta(str(table_path))
        assert result.shape[0] == 2
        assert set(result.columns) == {"meta_batch_date", "col_a", "col_b"}


class TestWriteColumnsAddsColumns:
    """Writing new columns to an existing table merges schema and preserves data."""

    def test_write_columns_adds_columns(self, tmp_path: Path) -> None:
        from node_fdm_data.delta import write_columns

        table_path = tmp_path / "flights.delta"

        # Write initial table with col_a
        df_a = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01"],
                "col_a": [1.0],
            }
        )
        write_columns(df_a, table_path)

        # Write col_b (new column)
        df_b = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01"],
                "col_b": [99.0],
            }
        )
        write_columns(df_b, table_path)

        result = pl.read_delta(str(table_path))
        assert "col_a" in result.columns
        assert "col_b" in result.columns
        # AC3: existing column data must be preserved, not null
        assert result.get_column("col_a").to_list() == [1.0]
        assert result.get_column("col_b").to_list() == [99.0]


class TestWriteColumnsIdempotent:
    """Writing the same column twice keeps latest values, other columns unchanged."""

    def test_write_columns_idempotent(self, tmp_path: Path) -> None:
        from node_fdm_data.delta import write_columns

        table_path = tmp_path / "flights.delta"

        # Initial write: col_a and col_b
        df_initial = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01"],
                "col_a": [1.0],
                "col_b": [10.0],
            }
        )
        write_columns(df_initial, table_path)

        # Overwrite col_b with new value
        df_update = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01"],
                "col_b": [99.0],
            }
        )
        write_columns(df_update, table_path)

        result = pl.read_delta(str(table_path))
        # AC4: col_a must keep its original value (not null)
        assert result.get_column("col_a").to_list() == [1.0]
        # AC4: col_b must have the latest value
        assert result.get_column("col_b").to_list() == [99.0]


class TestWriteColumnsPartition:
    """Writing with meta_batch_date creates partitions."""

    def test_write_columns_partition(self, tmp_path: Path) -> None:
        from node_fdm_data.delta import write_columns

        table_path = tmp_path / "flights.delta"
        df = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01", "2025-01-02"],
                "col_a": [1.0, 2.0],
            }
        )

        write_columns(df, table_path)

        result = pl.read_delta(str(table_path))
        unique_dates = result.get_column("meta_batch_date").unique().sort()
        assert unique_dates.len() == 2


class TestSchemaRegistryValidates:
    """Schema registry rejects type mismatches."""

    def test_schema_registry_validates(self) -> None:
        from node_fdm_data.schema import validate_schema

        # Registry defines raw_alt_ft as Float64; passing str should raise
        df = pl.DataFrame(
            {
                "raw_alt_ft": ["not_a_number", "bad_value"],
            }
        )

        with pytest.raises((TypeError, ValueError)):
            validate_schema(df, step="raw")


# ---------------------------------------------------------------------------
# Functional Tests — roundtrip and time travel
# ---------------------------------------------------------------------------


class TestRoundtripDelta:
    """Write then read a full table — data must be identical."""

    def test_roundtrip_delta(self, tmp_path: Path) -> None:
        from node_fdm_data.delta import read_delta_table, write_columns

        table_path = tmp_path / "flights.delta"
        df = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01", "2025-01-01", "2025-01-02"],
                "altitude_ft": [35000.0, 36000.0, 37000.0],
                "speed_kt": [450.0, 460.0, 470.0],
            }
        )

        write_columns(df, table_path)
        result = read_delta_table(table_path)

        # Sort for deterministic comparison
        df_sorted = df.sort("altitude_ft")
        result_sorted = result.sort("altitude_ft")

        assert df_sorted.shape == result_sorted.shape
        for col in df.columns:
            assert df_sorted.get_column(col).to_list() == result_sorted.get_column(col).to_list()


class TestTimeTravel:
    """Write v1, write v2, read v1 — v1 data must be recovered."""

    def test_time_travel(self, tmp_path: Path) -> None:
        from node_fdm_data.delta import read_delta_table, write_columns

        table_path = tmp_path / "flights.delta"

        # Version 0: initial data
        df_v1 = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01"],
                "altitude_ft": [35000.0],
            }
        )
        write_columns(df_v1, table_path)

        # Version 1: overwrite with different data
        df_v2 = pl.DataFrame(
            {
                "meta_batch_date": ["2025-01-01"],
                "altitude_ft": [40000.0],
            }
        )
        write_columns(df_v2, table_path)

        # Read version 0 — should get v1 data
        result = read_delta_table(table_path, version=0)
        assert result.get_column("altitude_ft").to_list() == [35000.0]
