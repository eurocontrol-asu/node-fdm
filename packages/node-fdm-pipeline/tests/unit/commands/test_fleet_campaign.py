from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from pytest_mock import MockerFixture


def _fleet_campaign() -> ModuleType:
    """Resolve the ticket's new public module inside each collected test."""
    return importlib.import_module("node_fdm_pipeline.commands.fleet_campaign")


def test_plan_mode_returns_the_plan_contract_object(mocker: MockerFixture) -> None:
    """AC1: plan returns the exact object produced by campaign_plan."""
    campaign = _fleet_campaign()
    config = Path("already-loaded-campaign.yaml")
    expected = object()
    plan = mocker.patch.object(campaign, "campaign_plan", return_value=expected)

    result = campaign.run_fleet_campaign(config, "plan")

    assert result is expected
    plan.assert_called_once_with(config)


def test_read_only_modes_return_their_contract_objects(mocker: MockerFixture) -> None:
    """AC1: plan, status, and validate preserve their contract objects unchanged."""
    campaign = _fleet_campaign()
    config = Path("already-loaded-campaign.yaml")
    expected_by_mode = {
        "plan": object(),
        "status": object(),
        "validate": object(),
    }
    contracts = {
        mode: mocker.patch.object(campaign, contract, return_value=expected_by_mode[mode])
        for mode, contract in {
            "plan": "campaign_plan",
            "status": "campaign_status",
            "validate": "campaign_validate",
        }.items()
    }

    returned = {
        mode: campaign.run_fleet_campaign(config, mode) for mode in ("plan", "status", "validate")
    }

    assert returned == expected_by_mode
    for contract in contracts.values():
        contract.assert_called_once_with(config)
