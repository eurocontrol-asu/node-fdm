"""Integration tests for the fleet download acquisition boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from node_fdm_pipeline.commands import _fleet_fetch
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    compute_resume_digest,
)
from node_fdm_pipeline.commands._fleet_journal import RunState, replay_journal
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._trino_lease import LeaseConfigError
from node_fdm_pipeline.config import FleetRunConfig

_SELECTION_DIGEST = "recorded-offline-selection"
_RESOLVED_CONFIG: DigestInput = {"workers": 1, "mode": "recorded-offline"}
_PROFILE: DigestInput = {"aircraft": "A320", "version": 1}


def _plan() -> Any:
    return SimpleNamespace(
        cohorts=(SimpleNamespace(name="recorded"),),
        dates={"20240101": ["abc123"], "20240102": ["def456"]},
        aircraft_days=2,
        requests_saved=lambda: (2, 2),
    )


def _recorded_fetch(calls: list[str]) -> Any:
    def fetch(plan: Any, date: str, *, force: bool = False) -> Any:
        calls.append(date)
        return SimpleNamespace(
            date=date,
            requested=1,
            written=1,
            requests=1,
            empty_kinds=(),
            error=None,
        )

    return fetch


def _download_kwargs(
    run_dir: Path,
    fleet_config: FleetRunConfig,
    calls: list[str],
) -> dict[str, Any]:
    return {
        "fleet_config": fleet_config,
        "recorded_digest": compute_resume_digest(
            _SELECTION_DIGEST,
            _RESOLVED_CONFIG,
            _PROFILE,
        ),
        "selection_digest": _SELECTION_DIGEST,
        "resolved_config": _RESOLVED_CONFIG,
        "profile": _PROFILE,
        "manifest_path": run_dir / "download-fleet.manifest.jsonl",
        "journal_path": run_dir / "download-fleet.journal.jsonl",
        "receipt_dir": run_dir / "receipts",
        "fetch_boundary": _recorded_fetch(calls),
    }


@pytest.mark.integration
def test_download_fleet_rejects_unset_lease_before_fetch_or_lock(
    tmp_path: Path,
) -> None:
    """AC2: an unset shared lease rejects before fetching or creating a fallback lock."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[str] = []
    fleet_config = FleetRunConfig.model_construct(
        lease_path=None,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )

    with pytest.raises(LeaseConfigError, match="shared lease path"):
        _fleet_fetch.download_fleet(
            _plan(),
            **_download_kwargs(run_dir, fleet_config, calls),
        )

    assert calls == []
    assert list(run_dir.rglob("*.lock")) == []


@pytest.mark.integration
def test_download_fleet_journals_lifecycle_and_releases_lease(
    tmp_path: Path,
) -> None:
    """AC3: a recorded run replays acquiring then processing and releases its lease."""
    run_dir = tmp_path / "run"
    shared_dir = tmp_path / "shared"
    run_dir.mkdir()
    shared_dir.mkdir()
    lease_path = shared_dir / "download-fleet.lease"
    journal_path = run_dir / "download-fleet.journal.jsonl"
    calls: list[str] = []
    fleet_config = FleetRunConfig(
        lease_path=lease_path,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )

    outcomes = _fleet_fetch.download_fleet(
        _plan(),
        **_download_kwargs(run_dir, fleet_config, calls),
    )

    events = read_events(journal_path)
    run_keys = {str(event["acquisition_key"]) for event in events}
    assert len(run_keys) == 1
    run_key = run_keys.pop()
    assert [RunState(str(event["state"])) for event in events] == [
        RunState.ACQUIRING,
        RunState.PROCESSING,
    ]
    assert replay_journal(journal_path).states[run_key] is RunState.PROCESSING
    assert [outcome.date for outcome in outcomes] == calls
    assert not lease_path.exists()
