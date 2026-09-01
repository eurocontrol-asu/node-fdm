"""Append-only audit trail for a resumable fleet download campaign."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["append_event"]


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Durably append one JSON event without rewriting campaign history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(UTC).isoformat(), **event}
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
