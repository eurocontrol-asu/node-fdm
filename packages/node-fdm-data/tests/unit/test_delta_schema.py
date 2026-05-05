"""Unit tests for Delta Lake schema validation registry (pure, no I/O)."""

from __future__ import annotations

import polars as pl
import pytest


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


class TestSchemaUnknownStep:
    """validate_schema raises ValueError for unknown pipeline step."""

    def test_schema_unknown_step(self) -> None:
        from node_fdm_data.schema import validate_schema

        df = pl.DataFrame({"raw_alt_ft": [35000.0]})

        with pytest.raises(ValueError, match="Unknown pipeline step"):
            validate_schema(df, step="nonexistent")


class TestSchemaValidPasses:
    """validate_schema accepts a valid DataFrame without raising."""

    def test_schema_valid_passes(self) -> None:
        from node_fdm_data.schema import validate_schema

        df = pl.DataFrame({"raw_alt_ft": [35000.0], "raw_gs_kt": [450.0]})

        # Should not raise
        validate_schema(df, step="raw")


class TestSchemaMissingColumnSkipped:
    """validate_schema skips columns not present in the DataFrame."""

    def test_schema_missing_column_skipped(self) -> None:
        from node_fdm_data.schema import validate_schema

        # Only one of many expected columns — the rest should be skipped
        df = pl.DataFrame({"raw_alt_ft": [35000.0]})

        # Should not raise even though most registry columns are missing
        validate_schema(df, step="raw")
