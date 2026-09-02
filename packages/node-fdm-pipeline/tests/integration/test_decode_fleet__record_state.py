"""Integration tests for durable fleet decode acquisition state."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _fleet_decode, _fleet_journal, _fleet_manifest
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    compute_resume_digest,
)
from node_fdm_pipeline.config import FleetRunConfig

_SELECTION_DIGEST = "recorded-offline-selection"
_RESOLVED_CONFIG: DigestInput = {"workers": 1, "mode": "recorded-offline"}
_PROFILE: DigestInput = {"aircraft": "A320", "version": 1}


class _RecordedExecutor:
    """Synchronous executor for the recorded offline decode outcome."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> _RecordedExecutor:
        return self

    def __exit__(self, *args: object) -> None:
        del args

    def submit(self, function: object, job: _fleet_decode.CohortDecode) -> Future[Any]:
        del function
        future: Future[Any] = Future()
        future.set_result(_fleet_decode.DecodeOutcome(name=job.name, seconds=0.0))
        return future


@pytest.mark.integration
def test_decode_fleet_records_lifecycle_and_releases_lease(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC4: decode durably records acquiring then processing and releases its lease."""
    run_dir = tmp_path / "run"
    fleet_dir = run_dir / "recorded-offline"
    shared_dir = tmp_path / "shared"
    fleet_dir.mkdir(parents=True)
    shared_dir.mkdir()
    lease_path = shared_dir / "decode-fleet.lease"
    journal_path = run_dir / "decode-fleet.journal.jsonl"
    job = _fleet_decode.CohortDecode(
        name="recorded",
        config_path=fleet_dir / "config.yaml",
        start_date="2024-01-01",
        end_date="2024-01-02",
        days=1,
    )
    mocker.patch.object(_fleet_decode, "plan_decodes", return_value=[job])
    mocker.patch.object(_fleet_decode, "ProcessPoolExecutor", _RecordedExecutor)
    fleet_config = FleetRunConfig(
        lease_path=lease_path,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )
    recorded = compute_resume_digest(_SELECTION_DIGEST, _RESOLVED_CONFIG, _PROFILE)

    _fleet_decode.decode_fleet(
        fleet_dir,
        workers=1,
        fleet_config=fleet_config,
        recorded_digest=recorded,
        selection_digest=_SELECTION_DIGEST,
        resolved_config=_RESOLVED_CONFIG,
        profile=_PROFILE,
        journal_path=journal_path,
        receipt_dir=run_dir / "receipts",
    )

    events = _fleet_manifest.read_events(journal_path)
    run_keys = {str(event["acquisition_key"]) for event in events}
    assert len(run_keys) == 1
    run_key = run_keys.pop()
    assert [_fleet_journal.RunState(str(event["state"])) for event in events[:2]] == [
        _fleet_journal.RunState.ACQUIRING,
        _fleet_journal.RunState.PROCESSING,
    ]
    assert (
        _fleet_journal.replay_journal(journal_path).states[run_key]
        is _fleet_journal.RunState.PROCESSING
    )
    assert not lease_path.exists()
