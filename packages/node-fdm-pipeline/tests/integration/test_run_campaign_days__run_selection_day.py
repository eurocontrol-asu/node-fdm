from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from node_fdm_pipeline.commands._day_bounds import ResourceBudget
from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan
from node_fdm_pipeline.commands._day_runner import run_selection_day
from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events


class _CampaignBudget(ResourceBudget, frozen=True):
    min_free_gib: float = 0.0


class _JournalExecutor:
    def __init__(
        self,
        journal_path: Path,
        *,
        cleanup_failure_day: str | None = None,
    ) -> None:
        self.journal_path = journal_path
        self.cleanup_failure_day = cleanup_failure_day
        self.day_runner = run_selection_day

    def estimate_resident_gib(self, plan: DayPlan) -> float:
        del plan
        return 0.1

    def free_disk_gib(self) -> float:
        return 1_000.0

    def planned_acquisition_batches(self, plan: DayPlan) -> tuple[str, ...]:
        del plan
        return ()

    @contextmanager
    def acquisition_section(self, plan: DayPlan, batch: str) -> Iterator[None]:
        del plan, batch
        yield

    def acquire(self, plan: DayPlan, batch: str) -> None:
        del plan, batch

    def run_day(self, plan: DayPlan) -> str:
        day = plan.meta_selection_day
        append_event(
            self.journal_path,
            {"event": "day_started", "meta_selection_day": day},
        )
        if day == self.cleanup_failure_day:
            append_event(
                self.journal_path,
                {"event": "cleanup_failed", "day": day},
            )
            return "cleanup_failed"
        append_event(
            self.journal_path,
            {"event": "cleanup_completed", "day": day},
        )
        return "completed"


def _campaign_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._campaign_runner")


def _plan(day: str) -> DayPlan:
    key = DayPartitionKey("cohort", day)
    return DayPlan(
        meta_selection_day=day,
        partition_keys=(key,),
        selection_ids=frozenset({f"selection-{day}"}),
        selection_ids_by_key={key: frozenset({f"selection-{day}"})},
        source_days=(day,),
    )


def _budget() -> _CampaignBudget:
    return _CampaignBudget(
        local_workers=1,
        max_resident_gib=8.0,
        min_free_gib=0.0,
    )


@pytest.mark.integration
def test_cleanup_failure_stops_campaign(tmp_path: Path) -> None:
    """AC5: a durable cleanup_failed outcome blocks every later day."""
    campaign_runner = _campaign_module()
    journal_path = tmp_path / "campaign.jsonl"
    blocking_day = "2024-01-01"
    executor = _JournalExecutor(
        journal_path,
        cleanup_failure_day=blocking_day,
    )

    report = campaign_runner.run_campaign_days(
        [_plan("2024-01-02"), _plan(blocking_day)],
        _budget(),
        executor,
    )

    started = [
        event["meta_selection_day"]
        for event in read_events(journal_path)
        if event.get("event") == "day_started"
    ]
    assert started == [blocking_day]
    assert blocking_day in str(report)


@pytest.mark.integration
def test_day_order_survives_real_journal(tmp_path: Path) -> None:
    """AC1: durable day_started events preserve ascending selection order."""
    campaign_runner = _campaign_module()
    journal_path = tmp_path / "campaign.jsonl"
    executor = _JournalExecutor(journal_path)

    campaign_runner.run_campaign_days(
        [_plan("2024-01-03"), _plan("2024-01-01"), _plan("2024-01-02")],
        _budget(),
        executor,
    )

    started = [
        event["meta_selection_day"]
        for event in read_events(journal_path)
        if event.get("event") == "day_started"
    ]
    assert started == ["2024-01-01", "2024-01-02", "2024-01-03"]
