from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any


def _journal() -> Any:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_journal")


def test_run_state_declares_the_durable_states_in_order() -> None:
    """AC1: expose exactly the ten durable states in their contractual order."""
    journal = _journal()

    assert [state.value for state in journal.RunState] == [
        "pending",
        "acquiring",
        "processing",
        "staged",
        "publishing",
        "committed",
        "cleaned",
        "failed",
        "interrupted",
        "cleanup_failed",
    ]


def test_next_incomplete_step_follows_the_last_complete_receipt() -> None:
    """AC5: resume at the durable step following the last complete receipt."""
    journal = _journal()
    key = "flight:AFR001:2026-09-02"
    cases = [
        (journal.RunState.ACQUIRING, "processing"),
        (journal.RunState.STAGED, "publishing"),
        (journal.RunState.COMMITTED, "cleaned"),
    ]

    for state, expected_step in cases:
        snapshot = journal.RunSnapshot(
            states={key: state},
            artifacts={key: ("receipt.json",)},
        )
        assert journal.next_incomplete_step(snapshot, key) == expected_step


def test_replay_attempts_preserves_recording_order_and_metadata() -> None:
    """AC1: rebuild attempts in journal order with their batch, delay, and branch."""
    journal = _journal()
    events = [
        {
            "event": "fetch_attempt",
            "attempt_id": "history:a001-a050:1",
            "date": "20260902",
            "kind": "history",
            "batch": ["a001", "a050"],
            "batch_label": "a001-a050",
            "attempt": 1,
            "outcome": "retry_later",
            "observed_delay_s": 1.25,
            "branch_name": "root.left",
        },
        {
            "event": "fetch_attempt",
            "attempt_id": "history:a051-a100:1",
            "date": "20260902",
            "kind": "history",
            "batch": ["a051", "a100"],
            "batch_label": "a051-a100",
            "attempt": 1,
            "outcome": "success",
            "observed_delay_s": 0.0,
            "branch_name": "root.right",
        },
        {
            "event": "fetch_attempt",
            "attempt_id": "history:a001-a050:2",
            "date": "20260902",
            "kind": "history",
            "batch": ["a001", "a050"],
            "batch_label": "a001-a050",
            "attempt": 2,
            "outcome": "success",
            "observed_delay_s": 2.5,
            "branch_name": "root.left",
        },
    ]

    attempts = journal.replay_attempts(events)

    assert [
        (attempt.batch_label, attempt.observed_delay_s, attempt.branch_name)
        for attempt in attempts
    ] == [
        ("a001-a050", 1.25, "root.left"),
        ("a051-a100", 0.0, "root.right"),
        ("a001-a050", 2.5, "root.left"),
    ]


def test_replay_journal_ignores_fetch_manifest_events(tmp_path: Path) -> None:
    """A shared append-only journal may contain state and fetch events."""
    journal = _journal()
    path = tmp_path / "campaign.jsonl"
    events = [
        {"event": "run_started", "plan_sha256": "abc"},
        {
            "acquisition_key": "campaign:abc",
            "state": "acquiring",
            "timestamp": "2026-09-07T12:00:00+00:00",
            "receipt": "/tmp/receipt.json",
        },
        {
            "acquisition_key": "campaign:abc",
            "state": "boundary_entered",
            "timestamp": "2026-09-07T12:00:01+00:00",
            "receipt": "/tmp/boundary.json",
        },
        {"event": "fetch_attempt", "attempt_id": "history:a001:1"},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    snapshot = journal.replay_journal(path)

    assert snapshot.states == {"campaign:abc": journal.RunState.ACQUIRING}
    assert snapshot.artifacts == {"campaign:abc": (Path("/tmp/receipt.json"),)}
