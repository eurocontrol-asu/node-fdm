"""Integration tests for the append-only fleet manifest journal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from node_fdm_pipeline.commands import _fleet_manifest


@pytest.mark.integration
def test_read_events_ignores_only_a_torn_terminal_line(tmp_path: Path) -> None:
    """AC1: ignore a partial terminal line and preserve all complete events in order."""
    path = tmp_path / "fleet.manifest.jsonl"
    payloads: list[dict[str, str | int]] = [
        {"event": "run_started", "campaign": "alpha"},
        {"event": "date_finished", "date": "20240101"},
        {"event": "run_finished", "failed": 0},
    ]
    for payload in payloads:
        _fleet_manifest.append_event(path, payload)

    complete_events = [json.loads(line) for line in path.read_text().splitlines()]
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event":"date_finished","date":"2024')

    assert _fleet_manifest.read_events(path) == complete_events


@pytest.mark.integration
def test_read_events_rejects_a_corrupt_non_terminal_line(tmp_path: Path) -> None:
    """AC2: raise ManifestCorruption when an unreadable line is not terminal."""
    path = tmp_path / "fleet.manifest.jsonl"
    path.write_text(
        '{"event":"run_started"}\n{"event":broken}\n{"event":"run_finished"}\n',
        encoding="utf-8",
    )

    with pytest.raises(_fleet_manifest.ManifestCorruption):
        _fleet_manifest.read_events(path)


@pytest.mark.integration
def test_appended_events_round_trip_in_order_with_intact_payloads(tmp_path: Path) -> None:
    """AC3: successive durable appends round-trip every payload in write order."""
    path = tmp_path / "fleet.manifest.jsonl"
    payloads = [
        {"event": "date_finished", "index": index, "nested": {"value": f"flight-{index}"}}
        for index in range(5)
    ]

    for payload in payloads:
        _fleet_manifest.append_event(path, payload)

    events = _fleet_manifest.read_events(path)

    assert len(events) == len(payloads)
    assert [
        {key: value for key, value in event.items() if key != "timestamp"} for event in events
    ] == payloads
