"""Integration contracts for durable, resumable cleanup."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import pytest

from node_fdm_pipeline.commands import _day_commit, _day_plan
from node_fdm_pipeline.commands._fleet_manifest import read_events

DAY = "2026-09-01"
StepHook = Callable[[str, str], None]


def _cleanup() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_cleanup")


def _committed_snapshot() -> _day_commit.DaySnapshot:
    key = _day_plan.DayPartitionKey(cohort="cohort-a", meta_selection_day=DAY)
    marker = _day_commit.DayCommit(
        meta_selection_day=DAY,
        partition_keys=(key,),
        profile_manifest={},
    )
    return _day_commit.DaySnapshot(
        meta_selection_day=DAY,
        published_keys=(key,),
        day_committed=marker,
    )


def _write_counters(path: Path, counters: Mapping[str, int]) -> None:
    path.write_text(json.dumps(counters, sort_keys=True) + "\n", encoding="utf-8")


def _read_counters(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return {str(key): int(value) for key, value in payload.items()}


def _artifacts(root: Path, ids: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for artifact_id in ids:
        path = root / artifact_id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload:{artifact_id}".encode())
        result[artifact_id] = path
    return result


def _published_digests(root: Path) -> dict[Path, str]:
    paths = (root / "cohort-a.parquet", root / "cohort-b.parquet")
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"committed-partition-{index}".encode())
    return {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _event_names(events: list[dict[str, object]]) -> list[str]:
    return [str(event.get("event", "")) for event in events]


@pytest.mark.integration
def test_cleanup_preserves_artifact_with_positive_reference_count(tmp_path: Path) -> None:
    """AC1: cleanup deletes exactly zero-reference artefacts and preserves a referenced raw."""
    ids = (
        "raw/20260901/history",
        "raw/20260901/flightlist",
        "era5/20260901/t2m",
    )
    artifacts = _artifacts(tmp_path / "inputs", ids)
    counters_path = tmp_path / "dependency-counters.json"
    _write_counters(counters_path, {ids[0]: 1, ids[1]: 0, ids[2]: 0})

    outcome = _cleanup().cleanup_day(
        _committed_snapshot(),
        journal_path=tmp_path / "cleanup.jsonl",
        counters_path=counters_path,
        artifacts=artifacts,
        decrements=(),
    )

    assert artifacts[ids[0]].is_file()
    assert not artifacts[ids[1]].exists()
    assert not artifacts[ids[2]].exists()
    assert outcome.deleted == tuple(sorted((ids[1], ids[2])))


@pytest.mark.integration
def test_failure_after_decrement_preserves_committed_partition_digests(
    tmp_path: Path,
) -> None:
    """AC2: a post-decrement fault is journalled without changing committed outputs."""
    artifact_id = "raw/20260901/history"
    artifacts = _artifacts(tmp_path / "inputs", (artifact_id,))
    counters_path = tmp_path / "dependency-counters.json"
    journal_path = tmp_path / "cleanup.jsonl"
    _write_counters(counters_path, {artifact_id: 2})
    committed = _published_digests(tmp_path / "published")

    def fail_after_decrement(step: str, _artifact_id: str) -> None:
        if step == "decrement":
            raise RuntimeError("injected after durable decrement")

    with pytest.raises(RuntimeError, match="injected after durable decrement"):
        _cleanup().cleanup_day(
            _committed_snapshot(),
            journal_path=journal_path,
            counters_path=counters_path,
            artifacts=artifacts,
            decrements=(artifact_id,),
            step_hook=fail_after_decrement,
        )

    assert "cleanup_failed" in _event_names(read_events(journal_path))
    assert all(path.is_file() for path in committed)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in committed} == committed


@pytest.mark.integration
def test_resume_after_deletion_finishes_only_remaining_cleanup_steps(
    tmp_path: Path,
) -> None:
    """AC3: resume deletes only remaining zeros without acquisition or publication."""
    ids = ("era5/20260901/t2m", "raw/20260901/flightlist", "raw/20260901/history")
    artifacts = _artifacts(tmp_path / "inputs", ids)
    counters_path = tmp_path / "dependency-counters.json"
    journal_path = tmp_path / "cleanup.jsonl"
    _write_counters(counters_path, dict.fromkeys(ids, 0))
    failed_once = False

    def fail_after_one_deletion(step: str, _artifact_id: str) -> None:
        nonlocal failed_once
        if step == "delete" and not failed_once:
            failed_once = True
            raise RuntimeError("injected after artefact deletion")

    with pytest.raises(RuntimeError, match="injected after artefact deletion"):
        _cleanup().cleanup_day(
            _committed_snapshot(),
            journal_path=journal_path,
            counters_path=counters_path,
            artifacts=artifacts,
            decrements=(),
            step_hook=fail_after_one_deletion,
        )

    existing_after_failure = {
        artifact_id for artifact_id, path in artifacts.items() if path.exists()
    }
    events_after_failure = read_events(journal_path)
    failure_index = _event_names(events_after_failure).index("cleanup_failed")
    resumed_deletions: list[str] = []

    def observe_resume(step: str, artifact_id: str) -> None:
        if step == "delete":
            resumed_deletions.append(artifact_id)

    outcome = _cleanup().resume_cleanup(
        _committed_snapshot(),
        journal_path=journal_path,
        counters_path=counters_path,
        artifacts=artifacts,
        decrements=(),
        step_hook=observe_resume,
    )

    resumed_events = read_events(journal_path)[failure_index + 1 :]
    assert outcome.state == "cleaned"
    assert set(resumed_deletions) == existing_after_failure
    assert all(not path.exists() for path in artifacts.values())
    assert not {
        name
        for name in _event_names(resumed_events)
        if name.startswith(("acquisition", "publication"))
    }


@pytest.mark.integration
def test_resume_does_not_repeat_a_durable_decrement(tmp_path: Path) -> None:
    """AC4: resume applies each decrement once and never drives a counter negative."""
    ids = ("raw/20260901/history", "era5/20260901/t2m")
    artifacts = _artifacts(tmp_path / "inputs", ids)
    counters_path = tmp_path / "dependency-counters.json"
    journal_path = tmp_path / "cleanup.jsonl"
    _write_counters(counters_path, {ids[0]: 2, ids[1]: 1})
    failed_once = False

    def fail_after_first_decrement(step: str, _artifact_id: str) -> None:
        nonlocal failed_once
        if step == "decrement" and not failed_once:
            failed_once = True
            raise RuntimeError("injected after first decrement")

    with pytest.raises(RuntimeError, match="injected after first decrement"):
        _cleanup().cleanup_day(
            _committed_snapshot(),
            journal_path=journal_path,
            counters_path=counters_path,
            artifacts=artifacts,
            decrements=ids,
            step_hook=fail_after_first_decrement,
        )

    counters_after_failure = _read_counters(counters_path)
    assert sorted(counters_after_failure.values()) == [1, 1]

    _cleanup().resume_cleanup(
        _committed_snapshot(),
        journal_path=journal_path,
        counters_path=counters_path,
        artifacts=artifacts,
        decrements=ids,
    )

    final_counters = _read_counters(counters_path)
    assert final_counters == {ids[0]: 1, ids[1]: 0}
    assert min(final_counters.values()) >= 0
