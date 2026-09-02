from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigestMismatch,
    compute_resume_digest,
)
from node_fdm_pipeline.commands._trino_lease import LeaseConfigError
from node_fdm_pipeline.config import FleetRunConfig

_SELECTION_DIGEST = "selection-v1"
_RESOLVED_CONFIG: dict[str, DigestInput] = {"workers": 2, "mode": "fleet"}
_PROFILE: dict[str, DigestInput] = {"aircraft": "A320", "version": 1}
_RESOLVE_GUARD_SYMBOL = "resolve_fleet_guard"
_DECISION_SYMBOL = "FleetGuardDecision"
_INCOMPLETE_SYMBOL = "CampaignGuardIncomplete"


def _boundary_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_boundary")


@pytest.mark.parametrize(
    "components",
    [("selection", "config", "profile")],
    ids=["selection-config-profile"],
)
def test_preflight_rejects_each_independent_resume_component(
    components: tuple[str, str, str],
) -> None:
    """AC1: chaque divergence isolée selection, config ou profile interdit la reprise."""
    boundary = _boundary_module()
    recorded = compute_resume_digest(_SELECTION_DIGEST, _RESOLVED_CONFIG, _PROFILE)
    changed_inputs = {
        "selection": ("selection-v2", _RESOLVED_CONFIG, _PROFILE),
        "config": (_SELECTION_DIGEST, {**_RESOLVED_CONFIG, "workers": 3}, _PROFILE),
        "profile": (
            _SELECTION_DIGEST,
            _RESOLVED_CONFIG,
            {**_PROFILE, "version": 2},
        ),
    }
    fleet_config = FleetRunConfig(
        lease_path=Path("/shared/fleet.lease"),
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )

    for component in components:
        selection_digest, resolved_config, profile = changed_inputs[component]

        with pytest.raises(ResumeDigestMismatch, match=rf"\b{component}\b"):
            boundary.preflight_acquisition(
                recorded_digest=recorded,
                selection_digest=selection_digest,
                resolved_config=resolved_config,
                profile=profile,
                fleet_config=fleet_config,
            )


def test_preflight_rejects_unset_shared_lease_path() -> None:
    """AC2: une configuration sans chemin de lease partagé est refusée en mémoire."""
    boundary = _boundary_module()
    recorded = compute_resume_digest(_SELECTION_DIGEST, _RESOLVED_CONFIG, _PROFILE)
    fleet_config = FleetRunConfig.model_construct(
        lease_path=None,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )

    with pytest.raises(LeaseConfigError, match="shared lease path"):
        boundary.preflight_acquisition(
            recorded_digest=recorded,
            selection_digest=_SELECTION_DIGEST,
            resolved_config=_RESOLVED_CONFIG,
            profile=_PROFILE,
            fleet_config=fleet_config,
        )


def test_resolve_fleet_guard_defaults_to_historical_mode() -> None:
    """AC1: an absent campaign input set resolves to historical mode."""
    boundary = _boundary_module()
    resolve_fleet_guard = getattr(boundary, _RESOLVE_GUARD_SYMBOL)
    decision_type = getattr(boundary, _DECISION_SYMBOL)

    decision = resolve_fleet_guard(
        selection=None,
        resolved_config=None,
        profile=None,
        lease_path=None,
    )

    assert isinstance(decision, decision_type)
    assert decision.mode == "historical"
    assert decision.preflight is None


def test_resolve_fleet_guard_names_all_missing_campaign_inputs() -> None:
    """AC2: a partial campaign set reports every missing input before any I/O."""
    boundary = _boundary_module()
    resolve_fleet_guard = getattr(boundary, _RESOLVE_GUARD_SYMBOL)
    incomplete_error = getattr(boundary, _INCOMPLETE_SYMBOL)

    with pytest.raises(incomplete_error) as exc_info:
        resolve_fleet_guard(
            selection=Path("selection.json"),
            resolved_config=None,
            profile=None,
            lease_path=None,
        )

    message = str(exc_info.value)
    assert "resolved_config" in message
    assert "profile" in message
    assert "lease_path" in message
