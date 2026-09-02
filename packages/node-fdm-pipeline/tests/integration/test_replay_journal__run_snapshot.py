from __future__ import annotations

import importlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

KEY = "flight:AFR001:2026-09-02"
WHEN = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


def _journal() -> Any:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_journal")


def _event(journal: Any, key: str, state: Any, sequence: int) -> Any:
    return journal.JournalEvent(
        acquisition_key=key,
        state=state,
        timestamp=WHEN,
        receipt={"sequence": sequence},
    )


@pytest.mark.integration
def test_replay_journal_reconstructs_each_durable_state(tmp_path: Path) -> None:
    """AC2: reconstruct every durable state from the on-disk journal alone."""
    journal = _journal()

    for sequence, state in enumerate(journal.RunState):
        case_dir = tmp_path / state.value
        journal_path = case_dir / "journal.jsonl"
        journal.record_state(
            journal_path,
            case_dir / "receipts",
            _event(journal, KEY, state, sequence),
        )

        snapshot = journal.replay_journal(journal_path)

        assert snapshot.states[KEY] == state


@pytest.mark.integration
def test_operator_summary_is_complete_and_deterministic(tmp_path: Path) -> None:
    """AC3: render every key, state, and receipt path deterministically."""
    journal = _journal()
    journal_path = tmp_path / "journal.jsonl"
    expected_states = {
        "flight:AFR001:2026-09-02": journal.RunState.PROCESSING,
        "flight:BAW002:2026-09-02": journal.RunState.STAGED,
    }
    sequence = 0
    for key, final_state in expected_states.items():
        for state in (journal.RunState.ACQUIRING, final_state):
            journal.record_state(
                journal_path,
                tmp_path / "receipts",
                _event(journal, key, state, sequence),
            )
            sequence += 1

    first_snapshot = journal.replay_journal(journal_path)
    first_summary = journal.render_operator_summary(first_snapshot)
    second_summary = journal.render_operator_summary(journal.replay_journal(journal_path))

    assert first_summary == second_summary
    assert len(first_summary.splitlines()) == len(expected_states)
    for key, state in expected_states.items():
        assert key in first_summary
        assert state.value in first_summary
        assert len(first_snapshot.artifacts[key]) == 2
        for receipt_path in first_snapshot.artifacts[key]:
            assert str(receipt_path) in first_summary


@pytest.mark.integration
def test_replay_ignores_a_torn_terminal_event(tmp_path: Path) -> None:
    """AC4: stop at the final complete event when the journal tail is torn."""
    journal = _journal()
    journal_path = tmp_path / "journal.jsonl"
    journal.record_state(
        journal_path,
        tmp_path / "receipts",
        _event(journal, KEY, journal.RunState.STAGED, 1),
    )
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(
            '{"acquisition_key":"flight:PARTIAL","state":"committed","timestamp":"2026-09'
        )

    snapshot = journal.replay_journal(journal_path)

    assert snapshot.states[KEY] == journal.RunState.STAGED
    assert "flight:PARTIAL" not in snapshot.states


@pytest.mark.integration
def test_record_state_fault_boundaries_never_replay_a_mixed_state(
    tmp_path: Path,
    mocker: Any,
) -> None:
    """AC6: replay only an old or new complete state across durable-write faults."""
    journal = _journal()
    original_append: Callable[[Path, dict[str, object]], None] = journal.append_event

    def fail_before_event(_path: Path, _event_data: dict[str, object]) -> None:
        raise OSError("injected fault before journal append")

    def write_torn_event(path: Path, _event_data: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"acquisition_key":"flight:AFR001')
            handle.flush()
        raise OSError("injected fault during journal append")

    boundaries = [
        ("receipt-only", fail_before_event),
        ("torn-journal", write_torn_event),
    ]
    for case_name, injected_append in boundaries:
        case_dir = tmp_path / case_name
        journal_path = case_dir / "journal.jsonl"
        journal.record_state(
            journal_path,
            case_dir / "receipts",
            _event(journal, KEY, journal.RunState.ACQUIRING, 1),
        )
        old_snapshot = journal.replay_journal(journal_path)

        patched_append = mocker.patch.object(
            journal,
            "append_event",
            side_effect=injected_append,
        )
        try:
            with pytest.raises(OSError, match="injected fault"):
                journal.record_state(
                    journal_path,
                    case_dir / "receipts",
                    _event(journal, KEY, journal.RunState.PROCESSING, 2),
                )
        finally:
            mocker.stop(patched_append)

        replayed = journal.replay_journal(journal_path)
        assert replayed.states[KEY] in {
            journal.RunState.ACQUIRING,
            journal.RunState.PROCESSING,
        }
        if replayed.states[KEY] == journal.RunState.ACQUIRING:
            assert replayed.artifacts == old_snapshot.artifacts
        else:
            assert len(replayed.artifacts[KEY]) == len(old_snapshot.artifacts[KEY]) + 1

        journal.append_event = original_append
