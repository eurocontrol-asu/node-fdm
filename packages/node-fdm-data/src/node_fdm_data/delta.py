"""Delta Lake read/write helpers with schema merge and partitioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

__all__ = [
    "read_delta_table",
    "table_info",
    "write_columns",
]


_BATCH_KEY = "meta_batch_date"


def _merge_preserved_columns(
    df: pl.DataFrame, existing: pl.DataFrame, preserve_cols: list[str]
) -> pl.DataFrame:
    key = _BATCH_KEY
    if key in existing.columns and key in df.columns:
        idx = "__row_nr"
        existing_idx = existing.with_columns(pl.cum_count(key).over(key).alias(idx))
        df_idx = df.with_columns(pl.cum_count(key).over(key).alias(idx))
        return df_idx.join(
            existing_idx.select([key, idx, *preserve_cols]),
            on=[key, idx],
            how="left",
        ).drop(idx)
    if len(existing) == len(df):
        return df.hstack(existing.select(preserve_cols))
    return df


def _maybe_merge_existing(df: pl.DataFrame, table_path: Path) -> pl.DataFrame:
    if not table_path.exists():
        return df
    existing = pl.read_delta(str(table_path))
    preserve_cols = [c for c in existing.columns if c not in df.columns]
    if not preserve_cols:
        return df
    return _merge_preserved_columns(df, existing, preserve_cols)


def _build_write_options(
    df: pl.DataFrame, table_path: Path, partition_by: list[str] | None
) -> dict[str, object]:
    options: dict[str, object] = {"schema_mode": "merge"}
    if partition_by is None:
        return options
    options["partition_by"] = partition_by
    if table_path.exists():
        dates = df[_BATCH_KEY].unique().sort().to_list()
        quoted = ", ".join(f"'{d}'" for d in dates)
        options["predicate"] = f"{_BATCH_KEY} IN ({quoted})"
    return options


def _cast_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Cast all-null (``pl.Null``) columns to ``Float64`` for Delta.

    A column is typed ``pl.Null`` when every value is null for the batch,
    which happens for an EHS register (BDS field) absent from a small or
    short sample. Delta rejects the ``Null`` type, so we cast such columns
    to ``Float64`` (the EHS fields are numeric); no data is lost since the
    column holds no values.
    """
    null_cols = [name for name, dtype in df.schema.items() if dtype == pl.Null]
    if not null_cols:
        return df
    return df.with_columns(pl.col(c).cast(pl.Float64) for c in null_cols)


def write_columns(df: pl.DataFrame, table_path: Path) -> None:
    """Write columns to a Delta table, preserving columns from previous steps.

    If the table already exists, existing columns not present in *df* are
    preserved by reading and merging them before writing.  Data is
    partitioned by ``meta_batch_date`` when that column is present.

    This makes the function **additive** (AC3) and **idempotent** (AC4):
    re-running a step overwrites only its own columns while leaving the
    rest intact.

    Args:
        df: DataFrame to write (may contain a subset of the table columns).
        table_path: Path to the Delta table directory.
    """
    partition_by = [_BATCH_KEY] if _BATCH_KEY in df.columns else None
    df = _maybe_merge_existing(df, table_path)
    df = _cast_null_columns(df)
    delta_write_options = _build_write_options(df, table_path, partition_by)
    df.write_delta(
        str(table_path),
        mode="overwrite",
        delta_write_options=delta_write_options,
    )


def read_delta_table(table_path: Path, *, version: int | None = None) -> pl.DataFrame:
    """Read a Delta table into a Polars DataFrame.

    Args:
        table_path: Path to the Delta table directory.
        version: Optional version number for time-travel reads.

    Returns:
        Polars DataFrame with the table contents.
    """
    if version is not None:
        return pl.read_delta(str(table_path), version=version)
    return pl.read_delta(str(table_path))


def table_info(table_path: Path) -> dict[str, Any]:
    """Return metadata about a Delta table.

    Args:
        table_path: Path to the Delta table directory.

    Returns:
        Dictionary with ``partitions``, ``columns``, and ``versions`` keys.
    """
    import re

    from deltalake import DeltaTable

    dt = DeltaTable(str(table_path))
    columns = [field.name for field in dt.schema().fields]
    partition_cols = dt.metadata().partition_columns

    # Extract unique partition values from file URIs.
    partitions: list[str] = []
    if partition_cols:
        pattern = re.compile(
            "|".join(rf"{col}=([^/]+)" for col in partition_cols),
        )
        values: set[str] = set()
        for uri in dt.file_uris():
            for m in pattern.finditer(uri):
                values.update(g for g in m.groups() if g is not None)
        partitions = sorted(values)

    versions = dt.version() + 1  # versions are 0-indexed

    return {
        "partitions": partitions,
        "columns": columns,
        "versions": versions,
    }
