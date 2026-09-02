from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_DATE = "20260902"


class _RecordingSource:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def history(
        self,
        _start: object,
        _end: object,
        *,
        icao24: list[str],
        cached: bool,
    ) -> None:
        assert cached is False
        self.calls.append(list(icao24))


def _attempt(
    batch: list[str],
    *,
    attempt: int,
    outcome: str,
    delay_s: float,
) -> dict[str, object]:
    batch_label = f"{batch[0]}-{batch[-1]}"
    return {
        "event": "fetch_attempt",
        "attempt_id": f"history:{batch_label}:{attempt}",
        "date": _DATE,
        "kind": "history",
        "batch": batch,
        "batch_label": batch_label,
        "attempt": attempt,
        "outcome": outcome,
        "observed_delay_s": delay_s,
        "branch_name": "root.left",
    }


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(event, sort_keys=True)}\n" for event in events),
        encoding="utf-8",
    )


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _run_resumed_fetch(
    monkeypatch: pytest.MonkeyPatch,
    journal_path: Path,
    source: _RecordingSource,
) -> tuple[Any, Any]:
    fleet_fetch = importlib.import_module("node_fdm_pipeline.commands._fleet_fetch")
    aircraft = [f"a{index:03d}" for index in range(1, 101)]
    plan = SimpleNamespace(
        dates={_DATE: aircraft},
        owner={icao24: object() for icao24 in aircraft},
    )
    monkeypatch.setattr(fleet_fetch, "_KINDS", ("history",))
    monkeypatch.setattr(fleet_fetch, "_get_opensky", lambda: source)

    outcome = fleet_fetch.fetch_one_date(
        plan,
        _DATE,
        force=True,
        manifest_path=journal_path,
    )
    return outcome, fleet_fetch


@pytest.mark.integration
def test_fetch_one_date_skips_a_received_batch_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2: resume never submits or re-journals a received batch, but submits the next."""
    received = [f"a{index:03d}" for index in range(1, 51)]
    pending = [f"a{index:03d}" for index in range(51, 101)]
    initial_events = [
        _attempt(
            received,
            attempt=1,
            outcome="success",
            delay_s=0.0,
        )
    ]
    journal_path = tmp_path / "fetch.jsonl"
    _write_events(journal_path, initial_events)
    source = _RecordingSource()

    outcome, _fleet_fetch = _run_resumed_fetch(monkeypatch, journal_path, source)

    appended = _read_events(journal_path)[len(initial_events) :]
    assert outcome.error is None
    assert source.calls == [pending]
    assert all(event.get("batch") != received for event in appended)
    assert any(event.get("batch") == pending for event in appended)


@pytest.mark.integration
def test_fetch_one_date_extends_attempt_history_without_duplicate_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3: resumed history keeps its exact prefix and never duplicates an attempt id."""
    received = [f"a{index:03d}" for index in range(1, 51)]
    initial_events = [
        _attempt(
            received,
            attempt=1,
            outcome="retry_later",
            delay_s=1.0,
        ),
        _attempt(
            received,
            attempt=2,
            outcome="retry_later",
            delay_s=2.0,
        ),
        _attempt(
            received,
            attempt=3,
            outcome="success",
            delay_s=0.0,
        ),
    ]
    journal_path = tmp_path / "fetch.jsonl"
    _write_events(journal_path, initial_events)
    before_ids = [str(event["attempt_id"]) for event in initial_events]
    source = _RecordingSource()

    outcome, _fleet_fetch = _run_resumed_fetch(monkeypatch, journal_path, source)
    journal = importlib.import_module("node_fdm_pipeline.commands._fleet_journal")
    attempts = journal.replay_attempts(_read_events(journal_path))
    after_ids = [attempt.attempt_id for attempt in attempts]

    assert outcome.error is None
    assert after_ids[: len(before_ids)] == before_ids
    assert len(after_ids) > len(before_ids)
    assert len(after_ids) == len(set(after_ids))
