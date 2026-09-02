from __future__ import annotations

from pathlib import Path

import structlog
from pydantic import BaseModel

from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigest,
    check_resume_compatible,
    compute_resume_digest,
)
from node_fdm_pipeline.commands._trino_lease import require_shared_lease_path
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
