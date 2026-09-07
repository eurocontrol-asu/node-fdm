from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from node_fdm_pipeline.commands._day_bounds import ResourceBudget, admit_slice
from node_fdm_pipeline.commands._day_plan import DayPlan

__all__ = [
    "CampaignBudget",
    "CampaignExecutor",
    "CampaignRunReport",
    "CampaignStepRun",
    "run_campaign_days",
]


class CampaignBudget(Protocol):
    """Resolved resource limits applied across the complete campaign."""

    local_workers: int
    max_resident_gib: float
    min_free_gib: float


class CampaignExecutor(Protocol):
    """Typed boundary between campaign scheduling and one-day execution."""

    def estimate_resident_gib(self, plan: DayPlan) -> float:
        """Estimate the resident-memory increment of one day slice."""
        ...

    def free_disk_gib(self) -> float:
        """Return currently available disk space in GiB."""
        ...

    def planned_acquisition_batches(self, plan: DayPlan) -> tuple[str, ...]:
        """Return the Trino acquisition batches needed by a day."""
        ...

    def acquisition_section(
        self,
        plan: DayPlan,
        batch: str,
    ) -> AbstractContextManager[None]:
        """Return the lease-protected context for one acquisition batch."""
        ...

    def acquire(self, plan: DayPlan, batch: str) -> None:
        """Run one Trino acquisition batch."""
        ...

    def run_day(self, plan: DayPlan) -> str:
        """Run the local day slice and return its terminal outcome."""
        ...


class CampaignStepRun(BaseModel, frozen=True):
    """One named campaign step selected for this invocation."""

    step: str


class CampaignRunReport(BaseModel, frozen=True):
    """Durable scheduling observations for one campaign invocation."""

    started_days: tuple[str, ...]
    completed_days: tuple[str, ...]
    blocked_reasons: tuple[str, ...]
    blocking_day: str | None
    max_observed_concurrency: int
    steps: tuple[CampaignStepRun, ...] = ()


@dataclass(frozen=True)
class _ActiveDay:
    plan: DayPlan
    resident_gib: float


def _run_plan(
    plan: DayPlan,
    executor: CampaignExecutor,
    acquisition_lock: threading.Lock,
) -> str:
    for batch in executor.planned_acquisition_batches(plan):
        with acquisition_lock, executor.acquisition_section(plan, batch):
            executor.acquire(plan, batch)
    return executor.run_day(plan)


def _dependencies(plan: DayPlan, planned_days: frozenset[str]) -> frozenset[str]:
    return frozenset(
        source_day
        for source_day in plan.source_days
        if source_day != plan.meta_selection_day and source_day in planned_days
    )


def _memory_block_reason(
    budget: CampaignBudget,
    *,
    active_resident_gib: float,
    slice_gib: float,
) -> str | None:
    resource_budget = ResourceBudget(
        local_workers=budget.local_workers,
        max_resident_gib=budget.max_resident_gib,
    )
    admission = admit_slice(
        budget=resource_budget,
        resident_gib=active_resident_gib,
        slice_gib=slice_gib,
    )
    return admission.reason if admission.deferred else None


def _disk_block_reason(budget: CampaignBudget, free_disk_gib: float) -> str | None:
    if free_disk_gib >= budget.min_free_gib:
        return None
    return f"free disk {free_disk_gib:.3f} GiB is under min_free_gib={budget.min_free_gib:.3f}"


def run_campaign_days(  # noqa: PLR0915
    plan: Iterable[DayPlan],
    budget: CampaignBudget,
    executor: CampaignExecutor,
) -> CampaignRunReport:
    """Run campaign days in selection order within all global resource bounds."""
    pending = deque(sorted(plan, key=lambda item: item.meta_selection_day))
    planned_days = frozenset(item.meta_selection_day for item in pending)
    completed: set[str] = set()
    completed_order: list[str] = []
    started: list[str] = []
    blocked_reasons: list[str] = []
    active: dict[Future[str], _ActiveDay] = {}
    active_resident_gib = 0.0
    blocking_day: str | None = None
    max_observed_concurrency = 0
    acquisition_lock = threading.Lock()
    halted = False

    with ThreadPoolExecutor(max_workers=budget.local_workers) as pool:
        while pending or active:
            while pending and len(active) < budget.local_workers and not halted:
                candidate = pending[0]
                dependencies = _dependencies(candidate, planned_days)
                if not dependencies.issubset(completed):
                    break

                slice_gib = executor.estimate_resident_gib(candidate)
                memory_reason = _memory_block_reason(
                    budget,
                    active_resident_gib=active_resident_gib,
                    slice_gib=slice_gib,
                )
                if memory_reason is not None:
                    if active and slice_gib <= budget.max_resident_gib:
                        break
                    blocked_reasons.append(memory_reason)
                    halted = True
                    break

                disk_reason = _disk_block_reason(budget, executor.free_disk_gib())
                if disk_reason is not None:
                    blocked_reasons.append(disk_reason)
                    halted = True
                    break

                pending.popleft()
                future = pool.submit(_run_plan, candidate, executor, acquisition_lock)
                active[future] = _ActiveDay(candidate, slice_gib)
                active_resident_gib += slice_gib
                started.append(candidate.meta_selection_day)
                max_observed_concurrency = max(max_observed_concurrency, len(active))

            if not active:
                break

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active_day = active.pop(future)
                active_resident_gib -= active_day.resident_gib
                outcome = future.result()
                day = active_day.plan.meta_selection_day
                if outcome == "cleanup_failed":
                    blocking_day = day
                    halted = True
                    continue
                completed.add(day)
                completed_order.append(day)

    return CampaignRunReport(
        started_days=tuple(started),
        completed_days=tuple(completed_order),
        blocked_reasons=tuple(blocked_reasons),
        blocking_day=blocking_day,
        max_observed_concurrency=max_observed_concurrency,
    )
