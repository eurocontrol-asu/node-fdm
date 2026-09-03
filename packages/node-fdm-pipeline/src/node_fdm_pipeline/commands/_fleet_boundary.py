from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel

from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigest,
    check_resume_compatible,
    compute_resume_digest,
    load_campaign_identity,
    record_campaign_identity,
)
from node_fdm_pipeline.commands._trino_lease import (
    Lease,
    LeaseUnavailable,
    acquire_lease,
    decide_lease_wait,
    require_shared_lease_path,
)
from node_fdm_pipeline.config import FleetRunConfig

__all__ = [
    "AcquisitionPreflight",
    "CampaignGuardIncomplete",
    "FleetGuardDecision",
    "preflight_acquisition",
    "resolve_fleet_guard",
]

log = structlog.get_logger()


class AcquisitionPreflight(BaseModel, frozen=True):
    """Validated inputs required before entering remote acquisition."""

    lease_path: Path
    resume_digest: ResumeDigest


class CampaignGuardIncomplete(ValueError):  # noqa: N818
    """Raised when only part of the campaign input set is supplied."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = tuple(sorted(missing))
        super().__init__(f"Incomplete campaign inputs; missing: {', '.join(self.missing)}")


class FleetGuardDecision(BaseModel, frozen=True):
    """Resolved acquisition mode and validated campaign state."""

    mode: Literal["historical", "campaign"]
    preflight: AcquisitionPreflight | None = None
    resume_digest: ResumeDigest | None = None


def resolve_fleet_guard(
    *,
    selection: Path | None,
    resolved_config: Path | None,
    profile: Path | None,
    lease_path: Path | None,
) -> FleetGuardDecision:
    """Resolve historical or campaign mode before acquisition writes begin."""
    campaign_inputs = {
        "selection": selection,
        "resolved_config": resolved_config,
        "profile": profile,
        "lease_path": lease_path,
    }
    present = {name for name, value in campaign_inputs.items() if value is not None}
    if not present:
        return FleetGuardDecision(mode="historical")

    missing = sorted(campaign_inputs.keys() - present)
    if missing:
        raise CampaignGuardIncomplete(missing)

    assert selection is not None
    assert resolved_config is not None
    assert profile is not None
    assert lease_path is not None

    shared_lease_path = require_shared_lease_path(lease_path)
    selection_digest = selection.read_text(encoding="utf-8")
    resolved_config_digest = resolved_config.read_text(encoding="utf-8")
    profile_digest = profile.read_text(encoding="utf-8")
    resume_digest = compute_resume_digest(
        selection_digest,
        resolved_config_digest,
        profile_digest,
    )
    preflight = AcquisitionPreflight(
        lease_path=shared_lease_path,
        resume_digest=resume_digest,
    )
    return FleetGuardDecision(
        mode="campaign",
        preflight=preflight,
        resume_digest=resume_digest,
    )


def preflight_acquisition(  # noqa: PLR0913
    *,
    recorded_digest: ResumeDigest,
    selection_digest: str,
    resolved_config: DigestInput,
    profile: DigestInput,
    fleet_config: FleetRunConfig,
    campaign_root: Path | None = None,
) -> AcquisitionPreflight:
    """Validate resume identity and the shared lease location before acquisition."""
    current_digest = compute_resume_digest(selection_digest, resolved_config, profile)
    authoritative_digest = recorded_digest
    if campaign_root is not None:
        try:
            authoritative_digest = load_campaign_identity(campaign_root)
        except FileNotFoundError:
            record_campaign_identity(campaign_root, current_digest)
            authoritative_digest = current_digest
    check_resume_compatible(authoritative_digest, current_digest)
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

    def record_waiting(self, holder: str) -> None:
        self._record("waiting", {"owner": self._owner, "holder": holder})

    def record_boundary_entered(self, *, enabled: bool) -> None:
        if enabled:
            self._record("boundary_entered", {"owner": self._owner})

    def record_lease_released(self, *, enabled: bool) -> None:
        if enabled:
            self._record("lease_released", {"owner": self._owner})

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
            AcquisitionState,
            JournalEvent,
            RunState,
            record_state,
        )

        lifecycle_state: RunState | AcquisitionState
        try:
            lifecycle_state = RunState(state)
        except ValueError:
            lifecycle_state = AcquisitionState(state)
        event = JournalEvent.model_validate(
            {
                "acquisition_key": self._acquisition_key,
                "state": lifecycle_state,
                "timestamp": datetime.now(UTC),
                "receipt": receipt,
            }
        )
        record_state(self._journal_path, self._receipt_dir, event)


def _acquire_with_wait(  # noqa: PLR0913
    lease_path: Path,
    *,
    owner: str,
    ttl_s: float,
    wait_budget_s: float,
    poll_interval_s: float,
    journal: _AcquisitionJournal,
) -> Lease:
    started = time.monotonic()
    while True:
        now = time.time()
        try:
            return acquire_lease(lease_path, owner=owner, ttl_s=ttl_s, now=now)
        except LeaseUnavailable as exc:
            if exc.record is None:
                raise
            decision = decide_lease_wait(
                exc.record,
                now=now,
                elapsed_s=time.monotonic() - started,
                wait_budget_s=wait_budget_s,
                poll_interval_s=poll_interval_s,
            )
            journal.record_waiting(decision.holder)
            if not decision.should_wait:
                raise
            time.sleep(decision.delay_s)


@contextmanager
def acquisition_section(  # noqa: PLR0913, PLR0915
    lease_path: Path,
    *,
    owner: str,
    ttl_s: float,
    journal_path: Path | None = None,
    receipt_dir: Path | None = None,
    acquisition_key: str | None = None,
    wait_budget_s: float = 0.0,
    poll_interval_s: float = 0.1,
    record_boundary_events: bool = False,
) -> Iterator[None]:
    """Hold an exclusive shared lease for the complete acquisition section."""
    if ttl_s <= 0:
        raise ValueError("Lease TTL must be positive")
    if wait_budget_s < 0:
        raise ValueError("Lease wait budget must not be negative")
    if poll_interval_s <= 0:
        raise ValueError("Lease poll interval must be positive")

    journal = _AcquisitionJournal.from_options(
        journal_path,
        receipt_dir,
        acquisition_key,
        owner,
    )

    lease = _acquire_with_wait(
        lease_path,
        owner=owner,
        ttl_s=ttl_s,
        wait_budget_s=wait_budget_s,
        poll_interval_s=poll_interval_s,
        journal=journal,
    )
    journal.record_acquiring()
    journal.record_boundary_entered(enabled=record_boundary_events)
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
        else:
            journal.record_lease_released(enabled=record_boundary_events)
        if not body_failed:
            if release_error is not None:
                journal.record_cleanup_failed(release_error)
                raise release_error
            if heartbeat_errors:
                raise heartbeat_errors[0]
