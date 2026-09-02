from __future__ import annotations

from pathlib import Path

import pytest

from node_fdm_pipeline.commands import _fleet_boundary as boundary
from node_fdm_pipeline.commands._fleet_digest import compute_resume_digest

pytestmark = pytest.mark.integration

_RESOLVE_GUARD_SYMBOL = "resolve_fleet_guard"
_DECISION_SYMBOL = "FleetGuardDecision"


@pytest.fixture
def campaign_inputs(tmp_path: Path) -> dict[str, Path]:
    selection = tmp_path / "selection.json"
    resolved_config = tmp_path / "resolved-config.json"
    profile = tmp_path / "profile.json"
    shared = tmp_path / "shared"
    selection.write_text('{"selection":"v1"}', encoding="utf-8")
    resolved_config.write_text('{"workers":2}', encoding="utf-8")
    profile.write_text('{"aircraft":"A320"}', encoding="utf-8")
    shared.mkdir()
    return {
        "selection": selection,
        "resolved_config": resolved_config,
        "profile": profile,
        "lease_path": shared / "fleet.lease",
    }


def test_resolve_fleet_guard_uses_complete_campaign_inputs(
    campaign_inputs: dict[str, Path],
) -> None:
    """AC3: complete campaign inputs produce their exact delegated resume digest."""
    decision_type = getattr(boundary, _DECISION_SYMBOL)
    resolve_fleet_guard = getattr(boundary, _RESOLVE_GUARD_SYMBOL)

    decision = resolve_fleet_guard(
        selection=campaign_inputs["selection"],
        resolved_config=campaign_inputs["resolved_config"],
        profile=campaign_inputs["profile"],
        lease_path=campaign_inputs["lease_path"],
    )

    expected = compute_resume_digest(
        campaign_inputs["selection"].read_text(encoding="utf-8"),
        campaign_inputs["resolved_config"].read_text(encoding="utf-8"),
        campaign_inputs["profile"].read_text(encoding="utf-8"),
    )
    assert isinstance(decision, decision_type)
    assert decision.mode == "campaign"
    assert decision.resume_digest == expected


@pytest.mark.parametrize(
    "input_names",
    [("profile", "resolved_config", "selection")],
    ids=["profile-resolved-config-selection"],
)
def test_resolve_fleet_guard_digest_changes_after_one_byte_rewrite(
    campaign_inputs: dict[str, Path],
    input_names: tuple[str, str, str],
) -> None:
    """AC4: one rewritten byte in each campaign input changes the resume digest."""
    resolve_fleet_guard = getattr(boundary, _RESOLVE_GUARD_SYMBOL)
    originals = {
        name: campaign_inputs[name].read_bytes()
        for name in ("selection", "resolved_config", "profile")
    }

    for input_name in input_names:
        for name, content in originals.items():
            campaign_inputs[name].write_bytes(content)
        first = resolve_fleet_guard(
            selection=campaign_inputs["selection"],
            resolved_config=campaign_inputs["resolved_config"],
            profile=campaign_inputs["profile"],
            lease_path=campaign_inputs["lease_path"],
        )
        content = originals[input_name]
        campaign_inputs[input_name].write_bytes(bytes([content[0] ^ 1]) + content[1:])

        second = resolve_fleet_guard(
            selection=campaign_inputs["selection"],
            resolved_config=campaign_inputs["resolved_config"],
            profile=campaign_inputs["profile"],
            lease_path=campaign_inputs["lease_path"],
        )

        assert second.resume_digest != first.resume_digest, input_name
