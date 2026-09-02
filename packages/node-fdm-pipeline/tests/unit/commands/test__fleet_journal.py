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
