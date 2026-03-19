"""Table-info command — inspect a Delta table's metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from node_fdm_data.delta import table_info as _table_info

__all__ = [
    "table_info",
]


def table_info(table_path: Path) -> dict[str, Any]:
    """Return metadata about a Delta table.

    Delegates to :func:`node_fdm_data.delta.table_info`.
    """
    return _table_info(table_path)
