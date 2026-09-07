"""Unit tests for the campaign guard operator contract."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from node_fdm_pipeline import config as config_module


def _campaign_guard() -> ModuleType:
    """Resolve the newly introduced module from inside the test execution."""
    return importlib.import_module("node_fdm_pipeline.commands._campaign_guard")


def _fleet_config(
    *,
    lease_path: Path | None,
    min_free_gib: float | None,
) -> config_module.FleetRunConfig:
    """Build guard inputs without letting model validation mask guard errors."""
    return config_module.FleetRunConfig.model_construct(
        lease_path=lease_path,
        lease_ttl_s=60,
        disk_min_gib=1.0,
        min_free_gib=min_free_gib,
    )


def test_the_five_operator_modes_resolve() -> None:
    """AC1: the five accepted spellings resolve and the vocabulary is closed."""
    campaign_guard = _campaign_guard()
    expected = {
        "plan": campaign_guard.CampaignMode.PLAN,
        "run": campaign_guard.CampaignMode.RUN,
        "resume": campaign_guard.CampaignMode.RESUME,
        "status": campaign_guard.CampaignMode.STATUS,
        "validate": campaign_guard.CampaignMode.VALIDATE,
    }

    assert {value: campaign_guard.resolve_campaign_mode(value) for value in expected} == expected

    with pytest.raises(campaign_guard.UnknownCampaignMode) as exc_info:
        campaign_guard.resolve_campaign_mode("download")
    assert all(value in str(exc_info.value) for value in expected)


def test_an_unknown_mode_is_refused() -> None:
    """AC1: an unknown mode names every accepted operator mode."""
    campaign_guard = _campaign_guard()

    with pytest.raises(campaign_guard.UnknownCampaignMode) as exc_info:
        campaign_guard.resolve_campaign_mode("download")

    message = str(exc_info.value)
    assert all(mode in message for mode in ("plan", "run", "resume", "status", "validate"))


def test_a_live_run_without_shared_lease_path_is_refused(tmp_path: Path) -> None:
    """AC4: live guards name both mandatory keys when each is absent."""
    campaign_guard = _campaign_guard()
    missing_lease = _fleet_config(lease_path=None, min_free_gib=2.0)
    with pytest.raises(campaign_guard.CampaignGuardIncomplete, match="lease_path"):
        campaign_guard.preflight_campaign(
            mode="run",
            state_dir=tmp_path,
            selection_digest="selection",
            resolved_config=missing_lease,
            profile={"worker": "local"},
        )

    missing_disk_guard = _fleet_config(
        lease_path=tmp_path / "trino.lease",
        min_free_gib=None,
    )
    with pytest.raises(campaign_guard.CampaignGuardIncomplete, match="min_free_gib"):
        campaign_guard.preflight_campaign(
            mode="resume",
            state_dir=tmp_path,
            selection_digest="selection",
            resolved_config=missing_disk_guard,
            profile={"worker": "local"},
        )


def test_a_live_run_without_min_free_gib_is_refused(tmp_path: Path) -> None:
    """AC4: a resume without min_free_gib identifies that missing key."""
    campaign_guard = _campaign_guard()
    config = _fleet_config(
        lease_path=tmp_path / "trino.lease",
        min_free_gib=None,
    )

    with pytest.raises(campaign_guard.CampaignGuardIncomplete, match="min_free_gib"):
        campaign_guard.preflight_campaign(
            mode="resume",
            state_dir=tmp_path,
            selection_digest="selection",
            resolved_config=config,
            profile={"worker": "local"},
        )
