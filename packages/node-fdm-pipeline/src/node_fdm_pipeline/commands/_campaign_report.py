from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from node_fdm_pipeline.commands import _day_publish
from node_fdm_pipeline.commands._campaign_plan import campaign_plan
from node_fdm_pipeline.commands._day_commit import load_day_snapshot
from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan
from node_fdm_pipeline.commands._day_publish import missing_partitions
from node_fdm_pipeline.commands._fleet_journal import (
    JournalEvent,
    RunSnapshot,
    RunState,
    next_incomplete_step,
    replay_journal,
)
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.config import PipelineConfig

__all__ = [
    "CampaignReport",
    "CampaignStepReport",
    "campaign_status",
    "campaign_validate",
    "fold_step_report",
    "next_actions_for",
]


class CampaignStepReport(BaseModel):
    """Complete operator-facing reconstruction of one journalled step."""

    model_config = ConfigDict(frozen=True)

    step: str
    duration: float
    bytes: int
    rows: int
    flight_identities: list[str]
    rejects: int
    split_batches: int
    cache_hits: int
    cache_misses: int
    throughput: float
    eta: float
    errors: list[str]
    caches: list[str]
    next_actions: list[str]


class CampaignReport(BaseModel):
    """Offline status or validation result for a recorded campaign."""

    model_config = ConfigDict(frozen=True)

    steps: list[CampaignStepReport]
    next_actions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    verdict: Literal["status", "valid", "invalid"] = "status"


class _StepReceipt(BaseModel):
    """Typed local receipt used to fold durable operator figures."""

    model_config = ConfigDict(frozen=True)

    duration: float = 0.0
    bytes: int = 0
    rows: int = 0
    flight_identities: list[str] = Field(default_factory=list)
    rejects: int = 0
    split_batches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: list[str] = Field(default_factory=list)
    caches: list[str] = Field(default_factory=list)
    day_journals: list[Path] = Field(default_factory=list)


class _PublishedRecord(BaseModel):
    """Digest and location persisted by one day publication event."""

    model_config = ConfigDict(frozen=True)

    event: Literal["publish_partition"]
    cohort: str
    meta_selection_day: str
    path: Path | None = None
    digest: str
    row_count: int


def _unique_strings(groups: Sequence[list[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for group in groups:
        for value in group:
            if value not in seen:
                seen.add(value)
                result.append(value)
    return result


def _fold_receipts(
    step: str,
    receipts: Sequence[_StepReceipt],
    *,
    planned_bytes: int,
    actions: list[str],
) -> CampaignStepReport:
    duration = sum(receipt.duration for receipt in receipts)
    byte_count = sum(receipt.bytes for receipt in receipts)
    throughput = byte_count / duration if duration > 0 else 0.0
    remaining_bytes = max(planned_bytes - byte_count, 0)
    eta = remaining_bytes / throughput if throughput > 0 else 0.0
    return CampaignStepReport(
        step=step,
        duration=duration,
        bytes=byte_count,
        rows=sum(receipt.rows for receipt in receipts),
        flight_identities=_unique_strings([receipt.flight_identities for receipt in receipts]),
        rejects=sum(receipt.rejects for receipt in receipts),
        split_batches=sum(receipt.split_batches for receipt in receipts),
        cache_hits=sum(receipt.cache_hits for receipt in receipts),
        cache_misses=sum(receipt.cache_misses for receipt in receipts),
        throughput=throughput,
        eta=eta,
        errors=_unique_strings([receipt.errors for receipt in receipts]),
        caches=_unique_strings([receipt.caches for receipt in receipts]),
        next_actions=list(actions),
    )


def fold_step_report(
    events: Sequence[JournalEvent],
    *,
    planned_bytes: int,
) -> CampaignStepReport:
    """Fold in-memory journal events into one complete step report."""
    if not events:
        raise ValueError("at least one journal event is required")
    step = events[0].acquisition_key
    if any(event.acquisition_key != step for event in events):
        raise ValueError("all journal events must belong to the same step")
    receipts = [_StepReceipt.model_validate(event.receipt) for event in events]
    actions = [f"resume {step}"] if events[-1].state == RunState.INTERRUPTED else []
    return _fold_receipts(
        step,
        receipts,
        planned_bytes=planned_bytes,
        actions=actions,
    )


def next_actions_for(snapshot: RunSnapshot) -> list[str]:
    """Return deterministic resume actions derived from durable step states."""
    actions: list[str] = []
    for step, state in snapshot.states.items():
        following = next_incomplete_step(snapshot, step)
        if state == RunState.INTERRUPTED and following is not None:
            actions.append(f"resume {step}")
    return actions


def _config_path(config: Path) -> Path:
    return config.expanduser().resolve()


def _input_path(config: Path, value: Path) -> Path:
    return value.expanduser() if value.is_absolute() else config.parent / value


def _campaign_snapshot(config: Path) -> tuple[RunSnapshot, int]:
    resolved = PipelineConfig.from_yaml(config)
    fleet_run = resolved.fleet_run
    plan = campaign_plan(config)
    if fleet_run is None or fleet_run.acquisition_journal is None:
        return RunSnapshot(states={}, artifacts={}), plan.estimated_disk_bytes
    journal = _input_path(config, fleet_run.acquisition_journal)
    snapshot = (
        replay_journal(journal) if journal.is_file() else RunSnapshot(states={}, artifacts={})
    )
    return snapshot, plan.estimated_disk_bytes


def _receipt(path: Path) -> _StepReceipt:
    return _StepReceipt.model_validate_json(path.read_bytes())


def campaign_status(config: Path) -> CampaignReport:
    """Reconstruct a complete campaign report exclusively from local state."""
    config_path = _config_path(config)
    snapshot, planned_bytes = _campaign_snapshot(config_path)
    actions = next_actions_for(snapshot)
    steps: list[CampaignStepReport] = []
    for step in snapshot.states:
        step_actions = [action for action in actions if action == f"resume {step}"]
        receipts = [_receipt(path) for path in snapshot.artifacts.get(step, ())]
        steps.append(
            _fold_receipts(
                step,
                receipts,
                planned_bytes=planned_bytes,
                actions=step_actions,
            )
        )
    return CampaignReport(steps=steps, next_actions=actions)


def _publication_records(journal: Path) -> list[_PublishedRecord]:
    records: list[_PublishedRecord] = []
    for event in read_events(journal):
        if event.get("event") == "publish_partition":
            records.append(_PublishedRecord.model_validate(event))
    return records


def _published_path(root: Path, record: _PublishedRecord) -> Path:
    if record.path is None:
        key = DayPartitionKey(record.cohort, record.meta_selection_day)
        return _day_publish._partition_path(root, key)
    return record.path if record.path.is_absolute() else root / record.path


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_day_journal(journal: Path, published_root: Path) -> list[str]:
    snapshot = load_day_snapshot(journal)
    plan = DayPlan(
        meta_selection_day=snapshot.meta_selection_day,
        partition_keys=snapshot.published_keys,
        selection_ids=frozenset(),
        selection_ids_by_key={key: frozenset() for key in snapshot.published_keys},
        source_days=(),
    )
    absent_by_layout = set(missing_partitions(plan, published_root))
    errors: list[str] = []
    for record in _publication_records(journal):
        key = DayPartitionKey(record.cohort, record.meta_selection_day)
        path = _published_path(published_root, record)
        identity = f"{record.cohort}/{record.meta_selection_day}"
        if key in absent_by_layout and not path.is_file():
            errors.append(f"{identity}: published partition is missing at {path}")
            continue
        if not path.is_file():
            errors.append(f"{identity}: published partition is missing at {path}")
            continue
        observed = _file_digest(path)
        if observed != record.digest:
            errors.append(
                f"{identity}: digest mismatch recorded={record.digest} observed={observed}"
            )
    return errors


def _day_journals(snapshot: RunSnapshot, config: Path) -> list[Path]:
    journals: list[Path] = []
    seen: set[Path] = set()
    for artifacts in snapshot.artifacts.values():
        for artifact in artifacts:
            receipt = _receipt(artifact)
            for journal in receipt.day_journals:
                resolved = _input_path(config, journal)
                if resolved not in seen:
                    seen.add(resolved)
                    journals.append(resolved)
    return journals


def campaign_validate(config: Path) -> CampaignReport:
    """Compare journalled publications with the local published tree."""
    config_path = _config_path(config)
    status = campaign_status(config_path)
    resolved = PipelineConfig.from_yaml(config_path)
    published_root = Path(resolved.paths.data_dir)
    if not published_root.is_absolute():
        published_root = config_path.parent / published_root
    snapshot, _ = _campaign_snapshot(config_path)
    errors: list[str] = []
    for journal in _day_journals(snapshot, config_path):
        errors.extend(_validate_day_journal(journal, published_root))
    verdict: Literal["valid", "invalid"] = "invalid" if errors else "valid"
    return CampaignReport(
        steps=status.steps,
        next_actions=status.next_actions,
        errors=errors,
        verdict=verdict,
    )
