from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from node_fdm_pipeline.commands._fleet_decode import _available_gib

__all__ = [
    "Admission",
    "ExternalAccessFromWorker",
    "ResourceBudget",
    "RunReport",
    "admit_slice",
    "register_parent_process",
    "require_parent_process",
    "run_bounded_slices",
]

_PARENT_PID_ENV = "NODE_FDM_PIPELINE_PARENT_PID"


class ResourceBudget(BaseModel, frozen=True):
    """Hard bounds applied to one selection day's local slice execution."""

    local_workers: int = Field(gt=0)
    max_resident_gib: float = Field(gt=0)


class Admission(BaseModel, frozen=True):
    """Decision describing whether a slice may enter the local process pool."""

    deferred: bool
    reason: str


class RunReport(BaseModel, frozen=True):
    """Results and measured bounds for one completed batch of slices."""

    results: tuple[object, ...]
    max_observed_concurrency: int
    deferred_then_run: tuple[str, ...]


class ExternalAccessFromWorker(RuntimeError):  # noqa: N818
    """A worker attempted an external operation reserved for its parent."""

    action: str
    pid: int

    def __init__(self, action: str, pid: int) -> None:
        self.action = action
        self.pid = pid
        super().__init__(action, pid)

    def __str__(self) -> str:
        return f"{self.action} is parent-process only; offending pid={self.pid}"


class _BoundedSlice(Protocol):
    @property
    def slice_id(self) -> str: ...

    @property
    def slice_gib(self) -> float: ...


@dataclass(frozen=True)
class _ActiveSlice:
    index: int
    slice_id: str
    slice_gib: float


def register_parent_process() -> int:
    """Register and return the process that exclusively owns external access."""
    parent_pid = os.getpid()
    os.environ[_PARENT_PID_ENV] = str(parent_pid)
    return parent_pid


def require_parent_process(action: str) -> None:
    """Reject an external action when invoked outside the registered parent."""
    current_pid = os.getpid()
    registered = os.environ.get(_PARENT_PID_ENV)
    if registered is None:
        raise RuntimeError("register_parent_process() must be called before external access")
    if current_pid != int(registered):
        raise ExternalAccessFromWorker(action, current_pid)


def admit_slice(
    *,
    budget: ResourceBudget,
    resident_gib: float,
    slice_gib: float,
) -> Admission:
    """Decide whether a slice fits both the configured and live memory bounds."""
    projected_gib = resident_gib + slice_gib
    if projected_gib > budget.max_resident_gib:
        return Admission(
            deferred=True,
            reason=(
                f"projected resident memory {projected_gib:.3f} GiB exceeds "
                f"max_resident_gib={budget.max_resident_gib:.3f}"
            ),
        )
    available_gib = _available_gib()
    if slice_gib > available_gib:
        return Admission(
            deferred=True,
            reason=(
                f"slice requires {slice_gib:.3f} GiB but only "
                f"{available_gib:.3f} GiB is currently available"
            ),
        )
    return Admission(deferred=False, reason="")


def _prepare_pending[SliceT: _BoundedSlice](
    slices: Iterable[SliceT],
    budget: ResourceBudget,
) -> deque[tuple[int, SliceT]]:
    items = tuple(slices)
    seen: set[str] = set()
    for slice_ in items:
        if slice_.slice_id in seen:
            raise ValueError(f"duplicate slice_id: {slice_.slice_id}")
        if slice_.slice_gib <= 0:
            raise ValueError(f"slice_gib must be positive for {slice_.slice_id}")
        if slice_.slice_gib > budget.max_resident_gib:
            raise ValueError(
                f"slice {slice_.slice_id} can never fit max_resident_gib={budget.max_resident_gib}"
            )
        seen.add(slice_.slice_id)
    return deque(enumerate(items))


def _submit_admissible[SliceT: _BoundedSlice, ResultT](  # noqa: PLR0913
    *,
    pending: deque[tuple[int, SliceT]],
    active: dict[Future[ResultT], _ActiveSlice],
    pool: ProcessPoolExecutor,
    budget: ResourceBudget,
    fn: Callable[[SliceT], ResultT],
    deferred_ids: list[str],
    deferred_seen: set[str],
) -> None:
    candidates = len(pending)
    for _ in range(candidates):
        if len(active) >= budget.local_workers:
            return
        index, slice_ = pending.popleft()
        resident_gib = sum(item.slice_gib for item in active.values())
        decision = admit_slice(
            budget=budget,
            resident_gib=resident_gib,
            slice_gib=slice_.slice_gib,
        )
        if decision.deferred:
            pending.append((index, slice_))
            if slice_.slice_id not in deferred_seen:
                deferred_ids.append(slice_.slice_id)
                deferred_seen.add(slice_.slice_id)
            continue
        future = pool.submit(fn, slice_)
        active[future] = _ActiveSlice(index, slice_.slice_id, slice_.slice_gib)


def _collect_completed[ResultT](
    active: dict[Future[ResultT], _ActiveSlice],
    results: list[object],
) -> None:
    completed, _ = wait(active, return_when=FIRST_COMPLETED)
    for future in completed:
        item = active.pop(future)
        results[item.index] = future.result()


def run_bounded_slices[SliceT: _BoundedSlice, ResultT](
    slices: Iterable[SliceT],
    budget: ResourceBudget,
    fn: Callable[[SliceT], ResultT],
) -> RunReport:
    """Run all slices in real worker processes without exceeding either bound."""
    pending = _prepare_pending(slices, budget)
    results: list[object] = [None] * len(pending)
    active: dict[Future[ResultT], _ActiveSlice] = {}
    deferred_ids: list[str] = []
    deferred_seen: set[str] = set()
    max_observed_concurrency = 0
    register_parent_process()

    with ProcessPoolExecutor(max_workers=budget.local_workers) as pool:
        while pending or active:
            _submit_admissible(
                pending=pending,
                active=active,
                pool=pool,
                budget=budget,
                fn=fn,
                deferred_ids=deferred_ids,
                deferred_seen=deferred_seen,
            )
            max_observed_concurrency = max(max_observed_concurrency, len(active))
            if active:
                _collect_completed(active, results)
            elif pending:
                time.sleep(0.05)

    return RunReport(
        results=tuple(results),
        max_observed_concurrency=max_observed_concurrency,
        deferred_then_run=tuple(deferred_ids),
    )
