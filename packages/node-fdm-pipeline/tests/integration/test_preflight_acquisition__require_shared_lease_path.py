from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigest,
    compute_resume_digest,
)
from node_fdm_pipeline.commands._trino_lease import LeaseConfigError
from node_fdm_pipeline.config import FleetRunConfig

_SELECTION_DIGEST = "selection-v1"
_RESOLVED_CONFIG: dict[str, DigestInput] = {"workers": 2, "mode": "fleet"}
_PROFILE: dict[str, DigestInput] = {"aircraft": "A320", "version": 1}
_COMPONENT_DIGESTS = {
    "selection": "11" * 32,
    "config": "22" * 32,
    "profile": "33" * 32,
}


def _boundary_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_boundary")


def _fleet_config(lease_path: Path) -> FleetRunConfig:
    return FleetRunConfig(
        lease_path=lease_path,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )


@pytest.mark.integration
def test_preflight_rejects_absent_shared_parent_without_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3: un parent partagé absent est refusé sans créer de lease locale."""
    boundary = _boundary_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    lease_path = tmp_path / "missing" / "fleet.lease"
    recorded = compute_resume_digest(_SELECTION_DIGEST, _RESOLVED_CONFIG, _PROFILE)

    with pytest.raises(LeaseConfigError, match="parent is unavailable"):
        boundary.preflight_acquisition(
            recorded_digest=recorded,
            selection_digest=_SELECTION_DIGEST,
            resolved_config=_RESOLVED_CONFIG,
            profile=_PROFILE,
            fleet_config=_fleet_config(lease_path),
        )

    assert not lease_path.parent.exists()
    assert list(run_dir.rglob("*")) == []


@pytest.mark.integration
def test_preflight_returns_exact_shared_path_and_component_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC4: le préflight accepté restitue le chemin et les trois digests littéraux."""
    boundary = _boundary_module()
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    lease_path = shared_dir / "fleet.lease"
    recorded = ResumeDigest(
        **_COMPONENT_DIGESTS,
        composite="44" * 32,
    )
    monkeypatch.setattr(
        boundary,
        "compute_resume_digest",
        lambda selection_digest, resolved_config, profile: recorded,
    )

    result = boundary.preflight_acquisition(
        recorded_digest=recorded,
        selection_digest=_SELECTION_DIGEST,
        resolved_config=_RESOLVED_CONFIG,
        profile=_PROFILE,
        fleet_config=_fleet_config(lease_path),
    )

    assert isinstance(result, boundary.AcquisitionPreflight)
    assert result.lease_path == lease_path
    assert {
        key: getattr(result.resume_digest, key) for key in ("selection", "config", "profile")
    } == _COMPONENT_DIGESTS
