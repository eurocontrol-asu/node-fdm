from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from node_fdm_pipeline.commands._day_weather import GridCacheStats
from node_fdm_pipeline.commands._fleet_journal import (
    JournalEvent,
    RunState,
    record_state,
)
from node_fdm_pipeline.commands._trino_lease import LeaseNotOwned

__all__ = ["interrupt_campaign", "release_owned_lease"]


class Closable(Protocol):
    """Local resource that can be closed deterministically."""

    def close(self) -> None:
        """Close this resource."""


class GridCache(Protocol):
    """Weather-grid cache boundary required during campaign shutdown."""

    def close(self) -> None:
        """Close every resident grid."""

    def stats(self) -> GridCacheStats:
        """Return the cache's current residency snapshot."""


class LeaseHandle(Protocol):
    """Owned Trino lease boundary required during campaign shutdown."""

    def release(self) -> None:
        """Release the lease while ownership is still valid."""


def release_owned_lease(lease: LeaseHandle | None) -> bool:
    """Release a lease only if its handle still owns the durable record."""
    if lease is None:
        return False
    try:
        lease.release()
    except LeaseNotOwned:
        return False
    return True


def interrupt_campaign[ResultT](  # noqa: PLR0913
    run: Callable[[], ResultT],
    *,
    acquisition_key: str,
    journal_path: Path,
    receipt_dir: Path,
    grid_cache: GridCache | None = None,
    readers: Iterable[Closable] = (),
    lease: LeaseHandle | None = None,
) -> ResultT:
    """Run a campaign and durably close its in-flight step on interruption."""
    try:
        return run()
    except KeyboardInterrupt:
        record_state(
            journal_path,
            receipt_dir,
            JournalEvent(
                acquisition_key=acquisition_key,
                state=RunState.INTERRUPTED,
                timestamp=datetime.now(UTC),
                receipt={"reason": "keyboard_interrupt"},
            ),
        )
        if grid_cache is not None:
            grid_cache.close()
        for reader in readers:
            reader.close()
        release_owned_lease(lease)
        raise
