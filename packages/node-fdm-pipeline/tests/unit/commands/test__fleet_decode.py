"""Unit tests for the fleet decode acquisition boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from node_fdm_pipeline.commands import _fleet_decode
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigestMismatch,
    compute_resume_digest,
)
from node_fdm_pipeline.config import FleetRunConfig

_SELECTION_DIGEST = "recorded-offline-selection"
_RESOLVED_CONFIG: DigestInput = {"workers": 1, "mode": "recorded-offline"}
_PROFILE: DigestInput = {"aircraft": "A320", "version": 1}


def test_decode_fleet_rejects_mismatching_recorded_digest() -> None:
    """AC1: decode rejects when the recorded config digest differs from the current one."""
    current = compute_resume_digest(_SELECTION_DIGEST, _RESOLVED_CONFIG, _PROFILE)
    recorded = current.model_copy(update={"config": "recorded-config-digest"})
    fleet_config = FleetRunConfig(
        lease_path=Path("/shared/decode-fleet.lease"),
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )

    with pytest.raises(ResumeDigestMismatch, match="config changed"):
        _fleet_decode.decode_fleet(
            Path("/recorded-offline/fleet"),
            workers=1,
            fleet_config=fleet_config,
            recorded_digest=recorded,
            selection_digest=_SELECTION_DIGEST,
            resolved_config=_RESOLVED_CONFIG,
            profile=_PROFILE,
        )
