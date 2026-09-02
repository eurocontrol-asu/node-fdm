"""Integration tests for the fleet enrich acquisition preflight."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _fleet_enrich
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    compute_resume_digest,
)
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._trino_lease import (
    LeaseConfigError,
    acquire_lease,
)
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


def _campaign_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    selection = tmp_path / "selection.txt"
    resolved_config = tmp_path / "resolved-config.json"
    profile = tmp_path / "profile.json"
    selection.write_text(_SELECTION_DIGEST, encoding="utf-8")
    resolved_config.write_text('{"workers": 1}', encoding="utf-8")
    profile.write_text('{"aircraft": "A320"}', encoding="utf-8")
    return selection, resolved_config, profile


def _cli_plan(run_dir: Path) -> _fleet_enrich.EnrichmentPlan:
    return _fleet_enrich.EnrichmentPlan(
        days={},
        cache_root=run_dir / "era5-cache",
        features=("temperature",),
    )


@pytest.mark.integration
def test_enrich_fleet_campaign_rejects_held_lease_before_provider_or_artifact(
    tmp_path: Path,
    mocker: MockerFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC1: a held shared lease aborts before weather access or enriched output."""
    from node_fdm_pipeline.cli import enrich_fleet

    run_dir = tmp_path / "run"
    shared_dir = tmp_path / "shared"
    run_dir.mkdir()
    shared_dir.mkdir()
    selection, resolved_config, profile = _campaign_files(tmp_path)
    lease_path = shared_dir / "enrich-fleet.lease"
    holder = acquire_lease(
        lease_path,
        owner="other-owner",
        ttl_s=60,
        now=time.time(),
    )
    mocker.patch(
        "node_fdm_pipeline.commands._fleet_plan.discover_cohorts",
        return_value=[],
    )
    mocker.patch.object(
        _fleet_enrich,
        "build_enrichment_plan",
        return_value=_cli_plan(run_dir),
    )
    provider_factory = mocker.patch.object(_fleet_enrich, "_default_provider")

    try:
        with pytest.raises(SystemExit) as raised:
            enrich_fleet(
                fleet_dir=tmp_path,
                data_root=run_dir,
                start_date="",
                end_date="",
                dry_run=False,
                selection=selection,
                resolved_config=resolved_config,
                profile=profile,
                lease_path=lease_path,
                lease_ttl_s=60,
                disk_min_gib=1.0,
            )
    finally:
        holder.release()

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert "Lease is held by other-owner" in captured.err
    assert provider_factory.call_count == 0
    assert list(run_dir.rglob("*.parquet")) == []


@pytest.mark.integration
def test_enrich_fleet_rejects_unusable_lease_before_plan_or_journal_event(
    tmp_path: Path,
    mocker: MockerFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC3: an unusable lease directory aborts before plan construction or journalling."""
    from node_fdm_pipeline.cli import enrich_fleet

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    journal_path = run_dir / "enrich-fleet.journal.jsonl"
    journal_path.write_text("", encoding="utf-8")
    selection, resolved_config, profile = _campaign_files(tmp_path)
    lease_path = tmp_path / "absent" / "enrich-fleet.lease"
    mocker.patch(
        "node_fdm_pipeline.commands._fleet_plan.discover_cohorts",
        return_value=[],
    )
    plan_builder = mocker.patch.object(
        _fleet_enrich,
        "build_enrichment_plan",
        return_value=_cli_plan(run_dir),
    )
    provider_factory = mocker.patch.object(_fleet_enrich, "_default_provider")

    with pytest.raises(SystemExit) as raised:
        enrich_fleet(
            fleet_dir=tmp_path,
            data_root=run_dir,
            start_date="",
            end_date="",
            dry_run=False,
            selection=selection,
            resolved_config=resolved_config,
            profile=profile,
            lease_path=lease_path,
            lease_ttl_s=60,
            disk_min_gib=1.0,
        )

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert "Shared lease parent is unavailable" in captured.err
    assert plan_builder.call_count == 0
    assert provider_factory.call_count == 0
    assert not lease_path.parent.exists()
    assert read_events(journal_path) == []
