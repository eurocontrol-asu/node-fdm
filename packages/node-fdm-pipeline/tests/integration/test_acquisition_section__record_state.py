"""Integration tests for durable acquisition-section lifecycle recording."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import cast

import pytest

from node_fdm_pipeline.commands import (
    _fleet_boundary,
    _fleet_journal,
    _fleet_manifest,
    _trino_lease,
)

KEY = "flight:AFR001:2026-09-02"
OWNER = "integration-test-owner"


@pytest.fixture
def lifecycle_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Provide isolated lease, journal, and receipt paths."""
    return (
        tmp_path / "lease.json",
        tmp_path / "journal.jsonl",
        tmp_path / "receipts",
    )


def _section(
    paths: tuple[Path, Path, Path],
) -> AbstractContextManager[None]:
    lease_path, journal_path, receipt_dir = paths
    return _fleet_boundary.acquisition_section(
        lease_path,
        owner=OWNER,
        ttl_s=60.0,
        journal_path=journal_path,
        receipt_dir=receipt_dir,
        acquisition_key=KEY,
    )


def _events(journal_path: Path) -> list[dict[str, object]]:
    return _fleet_manifest.read_events(journal_path)


def _receipt_payload(event: dict[str, object]) -> dict[str, object]:
    receipt_path = Path(str(event["receipt"]))
    return cast("dict[str, object]", json.loads(receipt_path.read_text(encoding="utf-8")))


@pytest.mark.integration
def test_completed_section_records_acquiring_then_processing(
    lifecycle_paths: tuple[Path, Path, Path],
) -> None:
    """AC1: a completed section replays acquiring then processing in order."""
    _, journal_path, _ = lifecycle_paths

    with _section(lifecycle_paths):
        pass

    events = _events(journal_path)
    snapshot = _fleet_journal.replay_journal(journal_path)

    assert [event["state"] for event in events] == [
        _fleet_journal.RunState.ACQUIRING.value,
        _fleet_journal.RunState.PROCESSING.value,
    ]
    assert snapshot.states[KEY] is _fleet_journal.RunState.PROCESSING


@pytest.mark.integration
def test_failing_body_records_failed_with_exception_class(
    lifecycle_paths: tuple[Path, Path, Path],
) -> None:
    """AC2: a body exception records failed and its exception class durably."""
    _, journal_path, _ = lifecycle_paths

    with pytest.raises(RuntimeError, match="boom"):
        with _section(lifecycle_paths):
            raise RuntimeError("boom")

    events = _events(journal_path)
    snapshot = _fleet_journal.replay_journal(journal_path)

    assert snapshot.states[KEY] is _fleet_journal.RunState.FAILED
    assert events[-1]["state"] == _fleet_journal.RunState.FAILED.value
    assert "RuntimeError" in json.dumps(_receipt_payload(events[-1]), sort_keys=True)


@pytest.mark.integration
def test_keyboard_interrupt_records_interrupted_not_failed(
    lifecycle_paths: tuple[Path, Path, Path],
) -> None:
    """AC3: KeyboardInterrupt records interrupted and never records failed."""
    _, journal_path, _ = lifecycle_paths

    with pytest.raises(KeyboardInterrupt):
        with _section(lifecycle_paths):
            raise KeyboardInterrupt

    events = _events(journal_path)
    snapshot = _fleet_journal.replay_journal(journal_path)
    states = [event["state"] for event in events]

    assert snapshot.states[KEY] is _fleet_journal.RunState.INTERRUPTED
    assert states[-1] == _fleet_journal.RunState.INTERRUPTED.value
    assert _fleet_journal.RunState.FAILED.value not in states


@pytest.mark.integration
def test_release_failure_records_cleanup_failed_with_reason(
    lifecycle_paths: tuple[Path, Path, Path],
) -> None:
    """AC4: a release failure records cleanup_failed and its reason durably."""
    lease_path, journal_path, _ = lifecycle_paths

    with pytest.raises(_trino_lease.LeaseNotOwned, match="The lease no longer exists") as raised:
        with _section(lifecycle_paths):
            lease_path.unlink()

    events = _events(journal_path)
    snapshot = _fleet_journal.replay_journal(journal_path)

    assert snapshot.states[KEY] is _fleet_journal.RunState.CLEANUP_FAILED
    assert events[-1]["state"] == _fleet_journal.RunState.CLEANUP_FAILED.value
    assert str(raised.value) in json.dumps(_receipt_payload(events[-1]), sort_keys=True)
