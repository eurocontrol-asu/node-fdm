"""Schema registry — column definitions and validation per pipeline step."""

from __future__ import annotations

from typing import Any

import polars as pl
from pydantic import BaseModel

__all__ = [
    "SCHEMA_REGISTRY",
    "StepSchema",
    "validate_schema",
]


class StepSchema(BaseModel, frozen=True):
    """Column-type mapping for a single pipeline step.

    Attributes:
        columns: Mapping of column name to expected Polars dtype string
                 (e.g. ``{"raw_alt_ft": "Float64"}``).
    """

    columns: dict[str, str]


# ---------------------------------------------------------------------------
# Registry — one entry per pipeline step
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, StepSchema] = {
    "raw": StepSchema(
        columns={
            "raw_alt_ft": "Float64",
            "raw_gs_kt": "Float64",
            "raw_trk_deg": "Float64",
            "raw_ias_kt": "Float64",
            "raw_tas_kt": "Float64",
            "raw_mach": "Float64",
            "raw_baro_alt_rate_ftmin": "Float64",
        },
    ),
    "preprocessed": StepSchema(
        columns={
            "altitude_ft": "Float64",
            "groundspeed_kt": "Float64",
            "track_deg": "Float64",
            "latitude": "Float64",
            "longitude": "Float64",
        },
    ),
    "processed": StepSchema(
        columns={
            "altitude_m": "Float64",
            "speed_ms": "Float64",
            "gamma_rad": "Float64",
            "mass_kg": "Float64",
        },
    ),
}

# ---------------------------------------------------------------------------
# Polars dtype name → set of compatible dtype strings
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, set[str]] = {
    "Float64": {"Float64", "Float32"},
    "Float32": {"Float32"},
    "Int64": {"Int64", "Int32", "Int16", "Int8"},
    "Int32": {"Int32", "Int16", "Int8"},
    "Utf8": {"Utf8", "String"},
    "String": {"Utf8", "String"},
}


def _polars_dtype_name(dtype: Any) -> str:
    """Return a canonical Polars dtype name string."""
    return str(dtype).replace("polars.", "").replace("datatypes.classes.", "")


def validate_schema(df: pl.DataFrame, *, step: str) -> None:
    """Validate a DataFrame's columns against the schema registry.

    Only columns **present in both** the DataFrame and the registry entry
    are checked.

    Args:
        df: DataFrame to validate.
        step: Pipeline step key (must exist in ``SCHEMA_REGISTRY``).

    Raises:
        ValueError: If *step* is unknown or a column type mismatches.
    """
    if step not in SCHEMA_REGISTRY:
        msg = f"Unknown pipeline step: {step!r}"
        raise ValueError(msg)

    expected = SCHEMA_REGISTRY[step].columns

    for col_name, expected_dtype in expected.items():
        if col_name not in df.columns:
            continue

        actual_dtype = _polars_dtype_name(df.schema[col_name])
        compatible = _DTYPE_MAP.get(expected_dtype, {expected_dtype})

        if actual_dtype not in compatible:
            msg = f"Column {col_name!r}: expected {expected_dtype}, got {actual_dtype}"
            raise TypeError(msg)
