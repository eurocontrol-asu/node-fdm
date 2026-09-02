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
