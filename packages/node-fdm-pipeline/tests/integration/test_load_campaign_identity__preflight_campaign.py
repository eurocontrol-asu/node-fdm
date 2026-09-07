"""Integration tests for campaign preflight against durable local state."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from node_fdm_pipeline import config as config_module
from node_fdm_pipeline.commands import _fleet_digest as fleet_digest

pytestmark = pytest.mark.integration


def _campaign_guard() -> ModuleType:
    """Resolve the newly introduced module from inside the test execution."""
    return importlib.import_module("node_fdm_pipeline.commands._campaign_guard")


def _fleet_config(tmp_path: Path) -> config_module.FleetRunConfig:
    """Return a complete live campaign config rooted in the test directory."""
    return config_module.FleetRunConfig(
        lease_path=tmp_path / "trino.lease",
        lease_ttl_s=60,
        disk_min_gib=1.0,
        min_free_gib=2.0,
    )


def test_run_refuses_an_incompatible_recorded_state(tmp_path: Path) -> None:
    """AC2: run rejects a different durable identity without mutating state."""
    campaign_guard = _campaign_guard()
    state_dir = tmp_path / "state"
    config = _fleet_config(tmp_path)
    profile = {"worker": "local"}
    recorded = fleet_digest.compute_resume_digest("selection-old", config, profile)
    fleet_digest.record_campaign_identity(state_dir, recorded)
    files_before = {path.relative_to(state_dir) for path in state_dir.rglob("*")}

    with pytest.raises(campaign_guard.CampaignStateIncompatible):
        campaign_guard.preflight_campaign(
            mode="run",
            state_dir=state_dir,
            selection_digest="selection-new",
            resolved_config=config,
            profile=profile,
        )

    assert {path.relative_to(state_dir) for path in state_dir.rglob("*")} == files_before


def test_resume_refuses_a_missing_state(tmp_path: Path) -> None:
    """AC3: resume rejects an absent identity before checking compatibility."""
    campaign_guard = _campaign_guard()
    state_dir = tmp_path / "empty-state"
    config = _fleet_config(tmp_path)

    with pytest.raises(campaign_guard.CampaignStateMissing):
        campaign_guard.preflight_campaign(
            mode="resume",
            state_dir=state_dir,
            selection_digest="selection",
            resolved_config=config,
            profile={"worker": "local"},
        )


def test_resume_refuses_a_divergent_digest(tmp_path: Path) -> None:
    """AC3: resume preserves ResumeDigestMismatch for divergent identity data."""
    campaign_guard = _campaign_guard()
    state_dir = tmp_path / "state"
    config = _fleet_config(tmp_path)
    profile = {"worker": "local"}
    recorded = fleet_digest.compute_resume_digest("selection-old", config, profile)
    fleet_digest.record_campaign_identity(state_dir, recorded)

    with pytest.raises(fleet_digest.ResumeDigestMismatch):
        campaign_guard.preflight_campaign(
            mode="resume",
            state_dir=state_dir,
            selection_digest="selection-new",
            resolved_config=config,
            profile=profile,
        )
