"""Append-only audit trail for a resumable fleet download campaign."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

__all__ = ["ManifestCorruption", "append_event", "read_events"]


class ManifestCorruption(ValueError):  # noqa: N818
    """Raised when a fleet manifest is corrupt before its torn tail."""


def append_event(path: Path, event: dict[str, Any]) -> None:
    """Durably append one JSON event without rewriting campaign history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(UTC).isoformat(), **event}
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_events(path: Path) -> list[dict[str, object]]:
    """Read complete manifest events, tolerating one torn terminal record."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    events: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        try:
            event = cast("dict[str, object]", json.loads(line))
        except json.JSONDecodeError as exc:
            is_torn_tail = index == len(lines) - 1 and not line.endswith(("\n", "\r"))
            if is_torn_tail:
                break
            raise ManifestCorruption(f"invalid manifest event on line {index + 1}") from exc
        events.append(event)
    return events
