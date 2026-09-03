from __future__ import annotations

import importlib
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest


@dataclass(frozen=True)
class _RecordedSlice:
    slice_id: str
    slice_gib: float
    record_dir: Path
    delay_seconds: float


def _day_bounds() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_bounds")


def _record_interval(slice_: _RecordedSlice) -> str:
    started_ns = time.monotonic_ns()
    (slice_.record_dir / f"{slice_.slice_id}.start").write_text(str(started_ns))
    time.sleep(slice_.delay_seconds)
    ended_ns = time.monotonic_ns()
    (slice_.record_dir / f"{slice_.slice_id}.end").write_text(str(ended_ns))
    return slice_.slice_id


def _attempt_opensky_fetch() -> None:
    _day_bounds().require_parent_process("opensky_fetch")


def _intervals(slices: tuple[_RecordedSlice, ...]) -> dict[str, tuple[int, int]]:
    return {
        slice_.slice_id: (
            int((slice_.record_dir / f"{slice_.slice_id}.start").read_text()),
            int((slice_.record_dir / f"{slice_.slice_id}.end").read_text()),
        )
        for slice_ in slices
    }


@pytest.mark.integration
def test_run_bounded_slices_never_exceeds_local_workers(tmp_path: Path) -> None:
    """AC2: eight real worker slices never overlap more than local_workers at once."""
    day_bounds = _day_bounds()
    slices = tuple(
        _RecordedSlice(
            slice_id=f"slice-{index}",
            slice_gib=1.0,
            record_dir=tmp_path,
            delay_seconds=0.15,
        )
        for index in range(8)
    )
    budget = day_bounds.ResourceBudget(local_workers=2, max_resident_gib=8.0)

    report = day_bounds.run_bounded_slices(slices=slices, budget=budget, fn=_record_interval)

    intervals = _intervals(slices)
    overlap_counts = [
        sum(start <= instant < end for start, end in intervals.values())
        for instant in (start for start, _ in intervals.values())
    ]
    assert report.max_observed_concurrency == 2
    assert max(overlap_counts) == 2


@pytest.mark.integration
def test_require_parent_process_refuses_worker_opensky_fetch() -> None:
    """AC3: a child process cannot perform the parent-owned OpenSky fetch."""
    day_bounds = _day_bounds()
    parent_pid = day_bounds.register_parent_process()

    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_attempt_opensky_fetch)
        with pytest.raises(day_bounds.ExternalAccessFromWorker) as exc_info:
            future.result()

    message = str(exc_info.value)
    assert "opensky_fetch" in message
    assert str(parent_pid) not in message
    assert str(exc_info.value.pid) in message
    assert exc_info.value.pid != parent_pid


@pytest.mark.integration
def test_run_bounded_slices_retries_resident_deferred_slice(tmp_path: Path) -> None:
    """AC4: the sole memory-deferred slice runs after admitted work releases budget."""
    day_bounds = _day_bounds()
    slices = (
        _RecordedSlice("admitted-a", 2.0, tmp_path, 0.15),
        _RecordedSlice("memory-deferred", 7.0, tmp_path, 0.05),
        _RecordedSlice("admitted-b", 2.0, tmp_path, 0.15),
        _RecordedSlice("admitted-c", 2.0, tmp_path, 0.05),
    )
    budget = day_bounds.ResourceBudget(local_workers=2, max_resident_gib=8.0)

    report = day_bounds.run_bounded_slices(slices=slices, budget=budget, fn=_record_interval)

    intervals = _intervals(slices)
    admitted_completion = min(
        intervals[slice_id][1] for slice_id in ("admitted-a", "admitted-b", "admitted-c")
    )
    assert set(report.results) == {slice_.slice_id for slice_ in slices}
    assert report.deferred_then_run == ("memory-deferred",)
    assert intervals["memory-deferred"][0] > admitted_completion
