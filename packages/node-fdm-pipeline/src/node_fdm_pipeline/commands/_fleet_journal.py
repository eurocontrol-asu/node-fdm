from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events

__all__ = [
    "AttemptRecord",
    "JournalEvent",
    "RunSnapshot",
    "RunState",
    "next_incomplete_step",
    "record_state",
    "render_operator_summary",
    "replay_attempts",
    "replay_journal",
]

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class RunState(StrEnum):
    """Durable states recorded for one acquisition."""

    PENDING = "pending"
    ACQUIRING = "acquiring"
    PROCESSING = "processing"
    STAGED = "staged"
    PUBLISHING = "publishing"
    COMMITTED = "committed"
    CLEANED = "cleaned"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CLEANUP_FAILED = "cleanup_failed"


class AcquisitionState(StrEnum):
    """Auditable coordination observations outside the durable run lifecycle."""

    WAITING = "waiting"
    BOUNDARY_ENTERED = "boundary_entered"
    LEASE_RELEASED = "lease_released"


class JournalEvent(BaseModel):
    """One durable state transition and the receipt that attests it."""

    model_config = ConfigDict(frozen=True)

    acquisition_key: str
    state: RunState | AcquisitionState
    timestamp: datetime
    receipt: dict[str, JsonValue]


class RunSnapshot(BaseModel):
    """State reconstructed exclusively from complete journal events."""

    model_config = ConfigDict(frozen=True)

    states: dict[str, RunState]
    artifacts: dict[str, tuple[Path, ...]]


class _PersistedEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    acquisition_key: str
    state: RunState
    timestamp: datetime
    receipt: Path


_NEXT_STATE: dict[RunState, RunState | None] = {
    RunState.PENDING: RunState.ACQUIRING,
    RunState.ACQUIRING: RunState.PROCESSING,
    RunState.PROCESSING: RunState.STAGED,
    RunState.STAGED: RunState.PUBLISHING,
    RunState.PUBLISHING: RunState.COMMITTED,
    RunState.COMMITTED: RunState.CLEANED,
    RunState.CLEANED: None,
    RunState.FAILED: RunState.ACQUIRING,
    RunState.INTERRUPTED: RunState.ACQUIRING,
    RunState.CLEANUP_FAILED: RunState.CLEANED,
}


class AttemptRecord(BaseModel):
    """One typed fetch attempt reconstructed from an append-only journal."""

    model_config = ConfigDict(frozen=True)

    event: str
    attempt_id: str
    date: str
    kind: str
    batch: tuple[str, ...]
    batch_label: str
    attempt: int
    outcome: str
    observed_delay_s: float
    branch_name: str
    acquisition_keys: tuple[str, ...] = ()


def replay_attempts(events: list[dict[str, object]]) -> tuple[AttemptRecord, ...]:
    """Rebuild fetch attempts without changing their journal order."""
    return tuple(
        AttemptRecord.model_validate(raw_event)
        for raw_event in events
        if raw_event.get("event") == "fetch_attempt"
    )


def replay_journal(path: Path) -> RunSnapshot:
    """Rebuild the current states and receipt paths from complete events."""
    states: dict[str, RunState] = {}
    artifacts: dict[str, list[Path]] = {}
    for raw_event in read_events(path):
        if not {"acquisition_key", "state", "receipt"}.issubset(raw_event):
            continue
        if raw_event.get("state") in {state.value for state in AcquisitionState}:
            continue
        event = _PersistedEvent.model_validate(raw_event)
        states[event.acquisition_key] = event.state
        artifacts.setdefault(event.acquisition_key, []).append(event.receipt)

    return RunSnapshot(
        states=states,
        artifacts={key: tuple(paths) for key, paths in artifacts.items()},
    )


def render_operator_summary(snapshot: RunSnapshot) -> str:
    """Render one stable, self-contained line per acquisition key."""
    lines: list[str] = []
    for key in sorted(snapshot.states):
        receipt_paths = ", ".join(sorted(str(path) for path in snapshot.artifacts.get(key, ())))
        lines.append(f"{key}: state={snapshot.states[key].value}; receipts=[{receipt_paths}]")
    return "\n".join(lines)


def next_incomplete_step(snapshot: RunSnapshot, key: str) -> RunState | None:
    """Return the durable step following the last completed receipt."""
    current = snapshot.states.get(key, RunState.PENDING)
    return _NEXT_STATE[current]


def record_state(journal_path: Path, receipt_dir: Path, event: JournalEvent) -> None:
    """Persist a receipt atomically before making its state visible in the journal."""
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_payload = json.dumps(
        event.receipt,
        sort_keys=True,
        separators=(",", ":"),
    )
    identity = "|".join(
        (
            event.acquisition_key,
            event.state.value,
            event.timestamp.isoformat(),
            receipt_payload,
        )
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    receipt_path = receipt_dir / f"{digest}.json"
    temporary_path = _write_temporary_receipt(receipt_dir, digest, receipt_payload)
    try:
        os.replace(temporary_path, receipt_path)
        _sync_directory(receipt_dir)
    finally:
        temporary_path.unlink(missing_ok=True)

    persisted: dict[str, object] = {
        "acquisition_key": event.acquisition_key,
        "state": event.state.value,
        "timestamp": event.timestamp.isoformat(),
        "receipt": str(receipt_path),
    }
    append_event(journal_path, persisted)


def _write_temporary_receipt(receipt_dir: Path, digest: str, payload: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=receipt_dir,
        prefix=f".{digest}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
