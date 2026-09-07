from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from pydantic import BaseModel, ConfigDict

__all__ = ["DayGridCache", "GridCacheStats", "GridStillReferenced"]

type GridKey = tuple[str, str]


class GridCacheStats(BaseModel):
    """Immutable snapshot of cumulative grid openings."""

    model_config = ConfigDict(frozen=True)

    opens: dict[GridKey, int]
    total_opens: int
    resident_entries: int = 0


class GridStillReferenced(RuntimeError):  # noqa: N818 - domain exception name
    """Raised when a source day still has active consumers."""

    def __init__(self, source_day: str, remaining_consumers: int) -> None:
        self.source_day = source_day
        self.remaining_consumers = remaining_consumers
        super().__init__(
            f"cannot evict source day {source_day}: "
            f"{remaining_consumers} consumer(s) still referenced"
        )


@dataclass(slots=True)
class _GridEntry[HandleT]:
    handle: HandleT
    consumers: int = 1


class DayGridCache[HandleT]:
    """Share opened day grids until every consumer releases its reference."""

    def __init__(
        self,
        *,
        opener: Callable[[str, str], HandleT],
        closer: Callable[[HandleT], None],
    ) -> None:
        self._opener = opener
        self._closer = closer
        self._entries: dict[GridKey, _GridEntry[HandleT]] = {}
        self._opens: dict[GridKey, int] = {}
        self._lock = RLock()

    def acquire(self, source_day: str, field: str) -> HandleT:
        """Acquire one consumer reference and return the shared grid handle."""
        key = (source_day, field)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                handle = self._opener(source_day, field)
                self._entries[key] = _GridEntry(handle=handle)
                self._opens[key] = self._opens.get(key, 0) + 1
                return handle
            entry.consumers += 1
            return entry.handle

    def release(self, source_day: str, field: str) -> None:
        """Release one consumer reference for an acquired grid."""
        key = (source_day, field)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise KeyError(f"grid {source_day}/{field} is not acquired")
            if entry.consumers == 0:
                raise RuntimeError(f"grid {source_day}/{field} has already been released")
            entry.consumers -= 1

    def evict(self, source_day: str) -> tuple[str, ...]:
        """Close and remove every unreferenced field for one source day."""
        with self._lock:
            keys = sorted(key for key in self._entries if key[0] == source_day)
            remaining = sum(self._entries[key].consumers for key in keys)
            if remaining:
                raise GridStillReferenced(source_day, remaining)

            evicted: list[str] = []
            for key in keys:
                entry = self._entries[key]
                self._closer(entry.handle)
                del self._entries[key]
                evicted.append(key[1])
            return tuple(evicted)

    def close(self) -> None:
        """Close and remove every resident grid, including referenced entries."""
        with self._lock:
            entries = tuple(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._closer(entry.handle)

    def stats(self) -> GridCacheStats:
        """Return a stable snapshot of cumulative grid-open counts."""
        with self._lock:
            opens = dict(self._opens)
            resident_entries = len(self._entries)
        return GridCacheStats(
            opens=opens,
            total_opens=sum(opens.values()),
            resident_entries=resident_entries,
        )
