from __future__ import annotations

from pathlib import Path

import pytest

from node_fdm_pipeline.commands import _fleet_boundary as boundary
from node_fdm_pipeline.commands._fleet_digest import (
    ResumeDigest,
    ResumeDigestMismatch,
    compute_resume_digest,
    load_campaign_identity,
    record_campaign_identity,
)
from node_fdm_pipeline.config import FleetRunConfig

pytestmark = pytest.mark.integration

_SELECTION = "selection-v1"
_RESOLVED_CONFIG = "resolved-config-v1"
_PROFILE = "profile-v1"


def _fleet_config(lease_path: Path) -> FleetRunConfig:
    return FleetRunConfig(
        lease_path=lease_path,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )


def _digest(
    selection: str = _SELECTION,
    resolved_config: str = _RESOLVED_CONFIG,
    profile: str = _PROFILE,
) -> ResumeDigest:
    return compute_resume_digest(selection, resolved_config, profile)


def _mutated_inputs(dimension: str) -> tuple[str, str, str]:
    values = {
        "selection": _SELECTION,
        "resolved_config": _RESOLVED_CONFIG,
        "profile": _PROFILE,
    }
    original = values[dimension]
    values[dimension] = f"{original[:-1]}2"
    return values["selection"], values["resolved_config"], values["profile"]


@pytest.mark.parametrize(
    "dimensions",
    [("selection", "resolved_config", "profile")],
    ids=["selection-resolved-config-profile"],
)
def test_preflight_rejects_each_recorded_identity_changed_by_one_byte(
    tmp_path: Path,
    dimensions: tuple[str, str, str],
) -> None:
    """AC1: one changed byte in each identity dimension rejects the resume."""
    campaign_root = tmp_path / "campaign"
    record_campaign_identity(campaign_root, _digest())
    lease_path = campaign_root / "fleet.lease"

    for dimension in dimensions:
        selection, resolved_config, profile = _mutated_inputs(dimension)
        current = _digest(selection, resolved_config, profile)

        with pytest.raises(ResumeDigestMismatch):
            boundary.preflight_acquisition(
                campaign_root=campaign_root,
                recorded_digest=current,
                selection_digest=selection,
                resolved_config=resolved_config,
                profile=profile,
                fleet_config=_fleet_config(lease_path),
            )


def test_rejected_identity_creates_no_lease_or_fallback_lock(tmp_path: Path) -> None:
    """AC2: identity rejection leaves neither a lease record nor a fallback lock."""
    campaign_root = tmp_path / "campaign"
    record_campaign_identity(campaign_root, _digest())
    before = {path.relative_to(campaign_root) for path in campaign_root.rglob("*")}
    selection, resolved_config, profile = _mutated_inputs("selection")
    current = _digest(selection, resolved_config, profile)
    lease_path = campaign_root / "fleet.lease"

    with pytest.raises(ResumeDigestMismatch):
        boundary.preflight_acquisition(
            campaign_root=campaign_root,
            recorded_digest=current,
            selection_digest=selection,
            resolved_config=resolved_config,
            profile=profile,
            fleet_config=_fleet_config(lease_path),
        )

    after = {path.relative_to(campaign_root) for path in campaign_root.rglob("*")}
    assert after == before
    assert not lease_path.exists()
    assert list(campaign_root.rglob("*.lock")) == []


def test_first_preflight_records_current_campaign_identity(tmp_path: Path) -> None:
    """AC3: a first preflight durably records all three current identity values."""
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    lease_path = campaign_root / "fleet.lease"
    current = _digest()

    result = boundary.preflight_acquisition(
        campaign_root=campaign_root,
        recorded_digest=current,
        selection_digest=_SELECTION,
        resolved_config=_RESOLVED_CONFIG,
        profile=_PROFILE,
        fleet_config=_fleet_config(lease_path),
    )

    assert result.lease_path == lease_path
    assert result.resume_digest == current
    assert load_campaign_identity(campaign_root) == current


def test_identity_mismatch_precedes_unavailable_shared_lease(tmp_path: Path) -> None:
    """AC4: identity mismatch wins before an unavailable shared lease is resolved."""
    campaign_root = tmp_path / "campaign"
    record_campaign_identity(campaign_root, _digest())
    selection, resolved_config, profile = _mutated_inputs("profile")
    current = _digest(selection, resolved_config, profile)
    lease_path = tmp_path / "unavailable" / "fleet.lease"

    with pytest.raises(ResumeDigestMismatch):
        boundary.preflight_acquisition(
            campaign_root=campaign_root,
            recorded_digest=current,
            selection_digest=selection,
            resolved_config=resolved_config,
            profile=profile,
            fleet_config=_fleet_config(lease_path),
        )

    assert not lease_path.parent.exists()
