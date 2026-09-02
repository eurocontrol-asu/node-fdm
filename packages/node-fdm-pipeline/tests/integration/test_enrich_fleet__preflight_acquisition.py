"""Integration tests for the fleet enrich acquisition preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _fleet_enrich
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    compute_resume_digest,
)
from node_fdm_pipeline.commands._trino_lease import LeaseConfigError
from node_fdm_pipeline.config import FleetRunConfig

_SELECTION_DIGEST = "recorded-offline-selection"
_RESOLVED_CONFIG: DigestInput = {"workers": 1, "mode": "recorded-offline"}
_PROFILE: DigestInput = {"aircraft": "A320", "version": 1}


@pytest.mark.integration
def test_enrich_fleet_rejects_unset_lease_before_provider_or_lock(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC3: an unset shared lease rejects before weather access or fallback locking."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    provider_factory = mocker.Mock(name="recorded_weather_provider")
    runtime = _fleet_enrich.EnrichmentRuntime(
        provider_factory=provider_factory,
        status_checker=mocker.Mock(),
        enrich_function=mocker.Mock(),
    )
    plan = _fleet_enrich.EnrichmentPlan(
        days={},
        cache_root=run_dir / "era5-cache",
        features=("temperature",),
    )
    fleet_config = FleetRunConfig.model_construct(
        lease_path=None,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )
    recorded = compute_resume_digest(_SELECTION_DIGEST, _RESOLVED_CONFIG, _PROFILE)
    kwargs: dict[str, Any] = {
        "fleet_config": fleet_config,
        "recorded_digest": recorded,
        "selection_digest": _SELECTION_DIGEST,
        "resolved_config": _RESOLVED_CONFIG,
        "profile": _PROFILE,
        "journal_path": run_dir / "enrich-fleet.journal.jsonl",
        "receipt_dir": run_dir / "receipts",
        "runtime": runtime,
    }

    with pytest.raises(LeaseConfigError, match="shared lease path"):
        _fleet_enrich.enrich_fleet(plan, **kwargs)

    assert provider_factory.call_count == 0
    assert list(run_dir.rglob("*.lock")) == []
