from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from node_fdm_pipeline.commands._day_commit import DaySnapshot, assert_purge_allowed
from node_fdm_pipeline.commands._day_plan import DayPartitionKey
from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events

__all__ = [
    "CleanupOutcome",
    "cleanup_day",
    "resume_cleanup",
    "zero_reference_artifacts",
]

type StepHook = Callable[[str, str], None]
type CounterMap = dict[str, int]
type CleanupState = Literal["cleaned"]


@dataclass
class _PurgeGuard:
    meta_selection_day: str
    published_keys: tuple[DayPartitionKey, ...]
    day_committed: object | None


class CleanupOutcome(BaseModel):
    """Immutable result of a completed day cleanup."""

    model_config = ConfigDict(frozen=True)

    state: CleanupState
    deleted: tuple[str, ...] = ()


def zero_reference_artifacts(counters: Mapping[str, int]) -> tuple[str, ...]:
    """Return artefact identifiers whose durable future-reference count is zero."""
    return tuple(sorted(artifact_id for artifact_id, count in counters.items() if count == 0))


def _read_counters(path: Path) -> CounterMap:
    payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError("dependency counters must be a JSON object")
    counters: CounterMap = {}
    for raw_artifact_id, raw_count in payload.items():
        if not isinstance(raw_artifact_id, str):
            raise ValueError("dependency counter identifiers must be strings")
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(f"invalid dependency counter for {raw_artifact_id}")
        counters[raw_artifact_id] = raw_count
    return counters


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_counters(path: Path, counters: Mapping[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(counters, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _events(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return read_events(path)


def _event_matches(event: Mapping[str, object], day: str, name: str) -> bool:
    return event.get("day") == day and event.get("event") == name


def _completed_artifacts(
    events: Iterable[Mapping[str, object]],
    day: str,
    event_name: str,
) -> set[str]:
    return {
        artifact_id
        for event in events
        if _event_matches(event, day, event_name)
        and isinstance((artifact_id := event.get("artifact_id")), str)
    }


def _pending_intent(
    events: Iterable[Mapping[str, object]],
    day: str,
    artifact_id: str,
) -> tuple[int, int] | None:
    for event in reversed(tuple(events)):
        if not _event_matches(event, day, "counter_decrement_intent"):
            continue
        if event.get("artifact_id") != artifact_id:
            continue
        before = event.get("before")
        after = event.get("after")
        if isinstance(before, int) and not isinstance(before, bool):
            if isinstance(after, int) and not isinstance(after, bool):
                return before, after
        raise ValueError(f"invalid decrement intent for {artifact_id}")
    return None


def _append_counter_event(  # noqa: PLR0913
    journal_path: Path,
    name: str,
    day: str,
    artifact_id: str,
    before: int,
    after: int,
) -> None:
    append_event(
        journal_path,
        {
            "event": name,
            "day": day,
            "artifact_id": artifact_id,
            "before": before,
            "after": after,
        },
    )


def _decrement_once(  # noqa: PLR0913
    *,
    day: str,
    artifact_id: str,
    counters: CounterMap,
    counters_path: Path,
    journal_path: Path,
    events: list[dict[str, object]],
    step_hook: StepHook | None,
) -> None:
    completed = _completed_artifacts(events, day, "counter_decremented")
    if artifact_id in completed:
        return

    current = counters.get(artifact_id)
    if current is None:
        raise KeyError(f"missing dependency counter for {artifact_id}")

    intent = _pending_intent(events, day, artifact_id)
    if intent is None:
        if current == 0:
            raise ValueError(f"dependency counter underflow for {artifact_id}")
        before, after = current, current - 1
        _append_counter_event(
            journal_path,
            "counter_decrement_intent",
            day,
            artifact_id,
            before,
            after,
        )
    else:
        before, after = intent

    if current == before:
        counters[artifact_id] = after
        _write_counters(counters_path, counters)
    elif current != after:
        raise ValueError(f"inconsistent dependency counter for {artifact_id}")

    _append_counter_event(
        journal_path,
        "counter_decremented",
        day,
        artifact_id,
        before,
        after,
    )
    events.append(
        {
            "event": "counter_decremented",
            "day": day,
            "artifact_id": artifact_id,
            "before": before,
            "after": after,
        }
    )
    if step_hook is not None:
        step_hook("decrement", artifact_id)


def _delete_once(  # noqa: PLR0913
    *,
    day: str,
    artifact_id: str,
    path: Path,
    journal_path: Path,
    deleted: set[str],
    step_hook: StepHook | None,
) -> None:
    if artifact_id in deleted:
        return
    existed = path.exists()
    if existed:
        path.unlink()
    append_event(
        journal_path,
        {
            "event": "artifact_deleted",
            "day": day,
            "artifact_id": artifact_id,
        },
    )
    deleted.add(artifact_id)
    if step_hook is not None:
        step_hook("delete", artifact_id)


def _run_cleanup(  # noqa: PLR0913
    snapshot: DaySnapshot,
    *,
    journal_path: Path,
    counters_path: Path,
    artifacts: Mapping[str, Path],
    decrements: Iterable[str],
    step_hook: StepHook | None,
) -> CleanupOutcome:
    assert_purge_allowed(
        _PurgeGuard(
            meta_selection_day=snapshot.meta_selection_day,
            published_keys=snapshot.published_keys,
            day_committed=snapshot.day_committed,
        )
    )
    day = snapshot.meta_selection_day
    events = _events(journal_path)
    try:
        counters = _read_counters(counters_path)
        for artifact_id in dict.fromkeys(decrements):
            _decrement_once(
                day=day,
                artifact_id=artifact_id,
                counters=counters,
                counters_path=counters_path,
                journal_path=journal_path,
                events=events,
                step_hook=step_hook,
            )

        zero_references = zero_reference_artifacts(counters)
        missing_paths = tuple(
            artifact_id for artifact_id in zero_references if artifact_id not in artifacts
        )
        if missing_paths:
            raise KeyError(f"missing cleanup paths for {', '.join(missing_paths)}")

        deleted = _completed_artifacts(events, day, "artifact_deleted")
        for artifact_id in zero_references:
            _delete_once(
                day=day,
                artifact_id=artifact_id,
                path=artifacts[artifact_id],
                journal_path=journal_path,
                deleted=deleted,
                step_hook=step_hook,
            )

        append_event(journal_path, {"event": "cleanup_completed", "day": day})
        return CleanupOutcome(state="cleaned", deleted=zero_references)
    except Exception as exc:
        append_event(
            journal_path,
            {
                "event": "cleanup_failed",
                "day": day,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def cleanup_day(  # noqa: PLR0913
    snapshot: DaySnapshot,
    *,
    journal_path: Path,
    counters_path: Path,
    artifacts: Mapping[str, Path],
    decrements: Iterable[str],
    step_hook: StepHook | None = None,
) -> CleanupOutcome:
    """Release one committed day's zero-reference inputs through a durable journal."""
    return _run_cleanup(
        snapshot,
        journal_path=journal_path,
        counters_path=counters_path,
        artifacts=artifacts,
        decrements=decrements,
        step_hook=step_hook,
    )


def resume_cleanup(  # noqa: PLR0913
    snapshot: DaySnapshot,
    *,
    journal_path: Path,
    counters_path: Path,
    artifacts: Mapping[str, Path],
    decrements: Iterable[str],
    step_hook: StepHook | None = None,
) -> CleanupOutcome:
    """Resume only unfinished decrements and deletions after a cleanup failure."""
    return _run_cleanup(
        snapshot,
        journal_path=journal_path,
        counters_path=counters_path,
        artifacts=artifacts,
        decrements=decrements,
        step_hook=step_hook,
    )
