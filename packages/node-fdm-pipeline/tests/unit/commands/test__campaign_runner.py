from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

from node_fdm_pipeline.commands._day_bounds import ResourceBudget
from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan


class _CampaignBudget(ResourceBudget, frozen=True):
    min_free_gib: float = 0.0


class _RecordingExecutor:
    def __init__(
        self,
        *,
        resident_by_day: dict[str, float] | None = None,
        free_disk_gib: float = 1_000.0,
        acquisition_batches: int = 0,
        delay_s: float = 0.02,
    ) -> None:
        self.resident_by_day = resident_by_day or {}
        self.free_disk = free_disk_gib
        self.batch_count = acquisition_batches
        self.delay_s = delay_s
        self.events: list[tuple[str, str]] = []
        self.started_days: list[str] = []
        self.active_local = 0
        self.max_active_local = 0
        self.active_acquisitions = 0
        self.max_active_acquisitions = 0
        self._lock = threading.Lock()

    def estimate_resident_gib(self, plan: DayPlan) -> float:
        return self.resident_by_day.get(plan.meta_selection_day, 0.1)

    def free_disk_gib(self) -> float:
        return self.free_disk

    def planned_acquisition_batches(self, plan: DayPlan) -> tuple[str, ...]:
        return tuple(f"{plan.meta_selection_day}:{index}" for index in range(self.batch_count))

    @contextmanager
    def acquisition_section(self, plan: DayPlan, batch: str) -> Iterator[None]:
        del plan, batch
        with self._lock:
            self.active_acquisitions += 1
            self.max_active_acquisitions = max(
                self.max_active_acquisitions,
                self.active_acquisitions,
            )
        try:
            yield
        finally:
            with self._lock:
                self.active_acquisitions -= 1

    def acquire(self, plan: DayPlan, batch: str) -> None:
        del plan, batch
        time.sleep(self.delay_s)

    def run_day(self, plan: DayPlan) -> str:
        day = plan.meta_selection_day
        with self._lock:
            self.events.append(("started", day))
            self.started_days.append(day)
            self.active_local += 1
            self.max_active_local = max(self.max_active_local, self.active_local)
        time.sleep(self.delay_s)
        with self._lock:
            self.events.append(("completed", day))
            self.active_local -= 1
        return "completed"


def _campaign_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._campaign_runner")


def _plan(day: str, *, source_days: tuple[str, ...] | None = None) -> DayPlan:
    key = DayPartitionKey("cohort", day)
    return DayPlan(
        meta_selection_day=day,
        partition_keys=(key,),
        selection_ids=frozenset({f"selection-{day}"}),
        selection_ids_by_key={key: frozenset({f"selection-{day}"})},
        source_days=source_days or (day,),
    )


def _budget(
    *,
    local_workers: int = 1,
    max_resident_gib: float = 8.0,
    min_free_gib: float = 0.0,
) -> _CampaignBudget:
    return _CampaignBudget(
        local_workers=local_workers,
        max_resident_gib=max_resident_gib,
        min_free_gib=min_free_gib,
    )


def test_shuffled_day_plans_start_in_selection_order() -> None:
    """AC1: shuffled plans start strictly in ascending selection-day order."""
    campaign_runner = _campaign_module()
    executor = _RecordingExecutor()

    campaign_runner.run_campaign_days(
        [_plan("2024-01-03"), _plan("2024-01-01"), _plan("2024-01-02")],
        _budget(),
        executor,
    )

    assert executor.started_days == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_cross_midnight_dependent_waits_for_predecessor() -> None:
    """AC2: a cross-midnight dependent starts only after predecessor completion."""
    campaign_runner = _campaign_module()
    executor = _RecordingExecutor()
    predecessor = _plan("2024-01-01")
    dependent = _plan(
        "2024-01-02",
        source_days=("2024-01-01", "2024-01-02"),
    )

    campaign_runner.run_campaign_days(
        [dependent, predecessor],
        _budget(local_workers=2),
        executor,
    )

    assert executor.events.index(("completed", "2024-01-01")) < executor.events.index(
        ("started", "2024-01-02")
    )


def test_local_concurrency_never_exceeds_local_workers() -> None:
    """AC3: campaign-local slices never exceed the configured worker bound."""
    campaign_runner = _campaign_module()
    executor = _RecordingExecutor(delay_s=0.05)
    plans = [_plan(f"2024-01-{day:02d}") for day in range(1, 9)]

    campaign_runner.run_campaign_days(
        plans,
        _budget(local_workers=2),
        executor,
    )

    assert executor.max_active_local == 2


def test_slice_above_max_resident_gib_is_not_started() -> None:
    """AC4: a slice exceeding max_resident_gib is blocked and reported."""
    campaign_runner = _campaign_module()
    day = "2024-01-01"
    executor = _RecordingExecutor(resident_by_day={day: 2.0})

    report = campaign_runner.run_campaign_days(
        [_plan(day)],
        _budget(max_resident_gib=1.0),
        executor,
    )

    assert executor.started_days == []
    assert "max_resident_gib" in str(report)


def test_low_free_disk_blocks_new_slices() -> None:
    """AC4: free disk below min_free_gib blocks starts and is reported."""
    campaign_runner = _campaign_module()
    executor = _RecordingExecutor(free_disk_gib=0.5)

    report = campaign_runner.run_campaign_days(
        [_plan("2024-01-01")],
        _budget(min_free_gib=1.0),
        executor,
    )

    assert executor.started_days == []
    assert "min_free_gib" in str(report)


def test_trino_acquisitions_never_overlap() -> None:
    """AC6: planned Trino acquisition sections are globally serialised."""
    campaign_runner = _campaign_module()
    executor = _RecordingExecutor(acquisition_batches=1, delay_s=0.05)
    plans = [_plan(f"2024-01-0{day}") for day in range(1, 4)]

    campaign_runner.run_campaign_days(
        plans,
        _budget(local_workers=3),
        executor,
    )

    assert executor.max_active_acquisitions == 1
