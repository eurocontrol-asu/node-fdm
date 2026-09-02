from __future__ import annotations

import importlib
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
