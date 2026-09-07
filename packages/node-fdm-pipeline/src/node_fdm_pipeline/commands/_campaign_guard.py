"""Central operator preflight contract for fleet campaigns."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import structlog

from node_fdm_pipeline.commands._fleet_boundary import CampaignGuardIncomplete
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigest,
    ResumeDigestMismatch,
    check_resume_compatible,
    compute_resume_digest,
    load_campaign_identity,
)
from node_fdm_pipeline.commands._trino_lease import require_shared_lease_path
from node_fdm_pipeline.config import FleetRunConfig

__all__ = [
    "CampaignGuardIncomplete",
    "CampaignMode",
    "CampaignStateIncompatible",
    "CampaignStateMissing",
    "UnknownCampaignMode",
    "preflight_campaign",
    "resolve_campaign_mode",
]

_LOGGER = structlog.get_logger(__name__)
_LIVE_MODES = frozenset({"run", "resume"})


class CampaignMode(StrEnum):
    """Operator actions accepted by the campaign entry point."""

    PLAN = "plan"
    RUN = "run"
    RESUME = "resume"
    STATUS = "status"
    VALIDATE = "validate"


class UnknownCampaignMode(ValueError):  # noqa: N818
    """Raised when an operator requests a mode outside the closed vocabulary."""


class CampaignStateIncompatible(RuntimeError):  # noqa: N818
    """Raised when a new run conflicts with an existing campaign identity."""


class CampaignStateMissing(RuntimeError):  # noqa: N818
    """Raised when resume is requested without a durable campaign identity."""


def resolve_campaign_mode(value: str) -> CampaignMode:
    """Resolve an exact operator spelling into its typed campaign mode."""
    try:
        return CampaignMode(value)
    except ValueError as exc:
        accepted = ", ".join(mode.value for mode in CampaignMode)
        raise UnknownCampaignMode(
            f"Unknown campaign mode {value!r}; expected one of: {accepted}"
        ) from exc


def preflight_campaign(
    *,
    mode: str | CampaignMode,
    state_dir: Path,
    selection_digest: str,
    resolved_config: FleetRunConfig,
    profile: DigestInput,
) -> ResumeDigest:
    """Validate live-run configuration and durable state before remote I/O."""
    campaign_mode = resolve_campaign_mode(mode)
    if campaign_mode.value in _LIVE_MODES:
        _validate_live_config(resolved_config)

    current = compute_resume_digest(selection_digest, resolved_config, profile)
    if campaign_mode is CampaignMode.RUN:
        _check_run_state(state_dir, current)
    elif campaign_mode is CampaignMode.RESUME:
        _check_resume_state(state_dir, current)
    return current


def _validate_live_config(config: FleetRunConfig) -> None:
    """Reject incomplete deployment guards before resolving their values."""
    missing: list[str] = []
    if config.lease_path is None:
        missing.append("lease_path")
    if config.min_free_gib is None:
        missing.append("min_free_gib")
    if missing:
        raise CampaignGuardIncomplete(missing)

    require_shared_lease_path(config.lease_path)


def _check_run_state(state_dir: Path, current: ResumeDigest) -> None:
    """Allow a fresh run, but reject reuse of incompatible durable state."""
    try:
        recorded = load_campaign_identity(state_dir)
    except FileNotFoundError:
        return

    try:
        check_resume_compatible(recorded, current)
    except ResumeDigestMismatch as exc:
        raise CampaignStateIncompatible(str(exc)) from exc


def _check_resume_state(state_dir: Path, current: ResumeDigest) -> None:
    """Require a durable identity and preserve component mismatch details."""
    try:
        recorded = load_campaign_identity(state_dir)
    except FileNotFoundError as exc:
        raise CampaignStateMissing(f"No campaign identity recorded in {state_dir}") from exc
    check_resume_compatible(recorded, current)
