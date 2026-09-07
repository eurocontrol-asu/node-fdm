from __future__ import annotations

import importlib
from datetime import UTC, datetime
from types import ModuleType

from pydantic import JsonValue

from node_fdm_pipeline.commands._fleet_journal import JournalEvent, RunSnapshot, RunState


def _report_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._campaign_report")


def _event(
    *,
    step: str = "decode",
    state: RunState = RunState.COMMITTED,
    duration: float = 2.0,
    byte_count: int = 400,
) -> JournalEvent:
    receipt: dict[str, JsonValue] = {
        "duration": duration,
        "bytes": byte_count,
        "rows": 20,
        "flight_identities": ["flight-001", "flight-002"],
        "rejects": 3,
        "split_batches": 2,
        "cache_hits": 7,
        "cache_misses": 1,
        "errors": ["one recoverable failure"],
        "caches": ["opensky-history", "era5"],
    }
    return JournalEvent(
        acquisition_key=step,
        state=state,
        timestamp=datetime(2026, 9, 7, tzinfo=UTC),
        receipt=receipt,
    )


def test_fold_step_report_carries_the_twelve_operator_figures() -> None:
    """AC1: one folded step exposes every figure needed by an operator."""
    report_module = _report_module()

    report = report_module.fold_step_report((_event(),), planned_bytes=1_000)

    assert report.duration == 2.0
    assert report.bytes == 400
    assert report.rows == 20
    assert report.flight_identities == ["flight-001", "flight-002"]
    assert report.rejects == 3
    assert report.split_batches == 2
    assert report.cache_hits == 7
    assert report.cache_misses == 1
    assert report.errors == ["one recoverable failure"]
    assert report.caches == ["opensky-history", "era5"]
    assert report.next_actions == []


def test_fold_step_report_derives_throughput_and_eta() -> None:
    """AC1: throughput and ETA are derived from duration, bytes and the plan."""
    report_module = _report_module()

    report = report_module.fold_step_report((_event(),), planned_bytes=1_000)

    assert report.throughput == 200.0
    assert report.eta == 3.0


def test_next_actions_for_names_the_interrupted_step() -> None:
    """AC2: the durable interrupted key is the exact step offered for resume."""
    report_module = _report_module()
    snapshot = RunSnapshot(
        states={"decode": RunState.INTERRUPTED},
        artifacts={"decode": ()},
    )

    actions = report_module.next_actions_for(snapshot)

    assert actions == ["resume decode"]
