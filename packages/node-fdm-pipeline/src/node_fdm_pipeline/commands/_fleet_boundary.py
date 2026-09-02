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


@contextmanager
def acquisition_section(
    lease_path: Path,
    *,
    owner: str,
    ttl_s: float,
) -> Iterator[None]:
    """Hold an exclusive shared lease for the complete acquisition section."""
    if ttl_s <= 0:
        raise ValueError("Lease TTL must be positive")

    lease = acquire_lease(
        lease_path,
        owner=owner,
        ttl_s=ttl_s,
        now=time.time(),
    )
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
        yield
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
                raise release_error
            if heartbeat_errors:
                raise heartbeat_errors[0]
