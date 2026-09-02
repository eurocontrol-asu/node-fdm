from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import structlog
from pydantic import BaseModel

from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigest,
    check_resume_compatible,
    compute_resume_digest,
)
from node_fdm_pipeline.commands._trino_lease import (
    acquire_lease,
    require_shared_lease_path,
)
from node_fdm_pipeline.config import FleetRunConfig

__all__ = ["AcquisitionPreflight", "preflight_acquisition"]

log = structlog.get_logger()


class AcquisitionPreflight(BaseModel, frozen=True):
    """Validated inputs required before entering remote acquisition."""

    lease_path: Path
    resume_digest: ResumeDigest


def preflight_acquisition(
    *,
    recorded_digest: ResumeDigest,
    selection_digest: str,
    resolved_config: DigestInput,
    profile: DigestInput,
    fleet_config: FleetRunConfig,
) -> AcquisitionPreflight:
    """Validate resume identity and the shared lease location before acquisition."""
    current_digest = compute_resume_digest(selection_digest, resolved_config, profile)
    check_resume_compatible(recorded_digest, current_digest)
    lease_path = require_shared_lease_path(fleet_config.lease_path)
    preflight = AcquisitionPreflight(
        lease_path=lease_path,
        resume_digest=current_digest,
    )
    log.debug("acquisition_preflight_ok", lease_path=str(lease_path))
    return preflight


class _AcquisitionJournal:
    """Optional durable lifecycle sink for one guarded acquisition."""

    __slots__ = ("_acquisition_key", "_journal_path", "_owner", "_receipt_dir")

    def __init__(
        self,
        journal_path: Path | None,
        receipt_dir: Path | None,
        acquisition_key: str | None,
        owner: str,
    ) -> None:
        self._journal_path = journal_path
        self._receipt_dir = receipt_dir
        self._acquisition_key = acquisition_key
        self._owner = owner

    @classmethod
    def from_options(
        cls,
        journal_path: Path | None,
        receipt_dir: Path | None,
        acquisition_key: str | None,
        owner: str,
    ) -> _AcquisitionJournal:
        if journal_path is None and receipt_dir is None and acquisition_key is None:
            return cls(None, None, None, owner)
        if journal_path is None or receipt_dir is None or acquisition_key is None:
            raise ValueError(
                "Journal path, receipt directory, and acquisition key must be provided together"
            )
        return cls(journal_path, receipt_dir, acquisition_key, owner)

    def record_acquiring(self) -> None:
        self._record("acquiring", {"owner": self._owner})

    def record_processing(self) -> None:
        self._record("processing", {"owner": self._owner})

    def record_interrupted(self) -> None:
        self._record("interrupted", {"exception_class": "KeyboardInterrupt"})

    def record_failed(self, exc: Exception) -> None:
        self._record("failed", {"exception_class": type(exc).__name__})

    def record_cleanup_failed(self, exc: Exception) -> None:
        self._record(
            "cleanup_failed",
            {
                "exception_class": type(exc).__name__,
                "reason": str(exc),
            },
        )

    def _record(self, state: str, receipt: dict[str, str]) -> None:
        if self._journal_path is None:
            return
        assert self._receipt_dir is not None
        assert self._acquisition_key is not None

        from datetime import UTC, datetime

        from node_fdm_pipeline.commands._fleet_journal import (
            JournalEvent,
            RunState,
            record_state,
        )

        event = JournalEvent.model_validate(
            {
                "acquisition_key": self._acquisition_key,
                "state": RunState(state),
                "timestamp": datetime.now(UTC),
                "receipt": receipt,
            }
        )
        record_state(self._journal_path, self._receipt_dir, event)


@contextmanager
def acquisition_section(  # noqa: PLR0913
    lease_path: Path,
    *,
    owner: str,
    ttl_s: float,
    journal_path: Path | None = None,
    receipt_dir: Path | None = None,
    acquisition_key: str | None = None,
) -> Iterator[None]:
    """Hold an exclusive shared lease for the complete acquisition section."""
    if ttl_s <= 0:
        raise ValueError("Lease TTL must be positive")

    journal = _AcquisitionJournal.from_options(
        journal_path,
        receipt_dir,
        acquisition_key,
        owner,
    )

    lease = acquire_lease(
        lease_path,
        owner=owner,
        ttl_s=ttl_s,
        now=time.time(),
    )
    journal.record_acquiring()
    stop_heartbeat = threading.Event()
    heartbeat_errors: list[Exception] = []

    def renew_lease() -> None:
        while not stop_heartbeat.wait(ttl_s / 3):
            try:
                lease.heartbeat(now=time.time())
            except Exception as exc:  # noqa: BLE001
                heartbeat_errors.append(exc)
                return

    heartbeat = threading.Thread(
        target=renew_lease,
        name=f"lease-heartbeat-{owner}",
        daemon=True,
    )
    heartbeat.start()
    body_failed = False
    try:
        journal.record_processing()
        yield
    except KeyboardInterrupt:
        body_failed = True
        journal.record_interrupted()
        raise
    except Exception as exc:
        body_failed = True
        journal.record_failed(exc)
        raise
    except BaseException:
        body_failed = True
        raise
    finally:
        stop_heartbeat.set()
        heartbeat.join()
        release_error: Exception | None = None
        try:
            lease.release()
        except Exception as exc:  # noqa: BLE001
            release_error = exc
        if not body_failed:
            if release_error is not None:
                journal.record_cleanup_failed(release_error)
                raise release_error
            if heartbeat_errors:
                raise heartbeat_errors[0]
