from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest

from node_fdm_pipeline.commands._campaign_runner import run_campaign_days
from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan
from node_fdm_pipeline.commands._day_weather import DayGridCache, GridCacheStats
from node_fdm_pipeline.commands._fleet_journal import RunState, replay_journal
from node_fdm_pipeline.commands._trino_lease import acquire_lease

pytestmark = pytest.mark.integration


@dataclass
class _Budget:
    local_workers: int = 1
    max_resident_gib: float = 1.0
    min_free_gib: float = 0.0


def _plan(day: str) -> DayPlan:
    key = DayPartitionKey(cohort="A", meta_selection_day=day)
    return DayPlan(
        meta_selection_day=day,
        partition_keys=(key,),
        selection_ids=frozenset({"selection-A"}),
        selection_ids_by_key={key: frozenset({"selection-A"})},
        source_days=(day,),
    )


class _InterruptingExecutor:
    def __init__(self, interrupt_day: str) -> None:
        self.interrupt_day = interrupt_day

    def estimate_resident_gib(self, plan: DayPlan) -> float:
        return 0.1

    def free_disk_gib(self) -> float:
        return 100.0

    def planned_acquisition_batches(self, plan: DayPlan) -> tuple[str, ...]:
        return ()

    def acquisition_section(
        self,
        plan: DayPlan,
        batch: str,
    ) -> AbstractContextManager[None]:
        return nullcontext()

    def acquire(self, plan: DayPlan, batch: str) -> None:
        raise AssertionError("this executor declares no acquisition batches")

    def run_day(self, plan: DayPlan) -> str:
        if plan.meta_selection_day == self.interrupt_day:
            raise KeyboardInterrupt
        return "cleaned"


def _raise_keyboard_interrupt() -> None:
    raise KeyboardInterrupt


def _run_with_interrupt_handler(  # noqa: PLR0913
    run: Callable[[], object],
    *,
    tmp_path: Path,
    acquisition_key: str,
    grid_cache: DayGridCache[object] | None = None,
    readers: Iterable[object] = (),
    lease: object | None = None,
) -> object:
    campaign_interrupt = importlib.import_module("node_fdm_pipeline.commands._campaign_interrupt")
    return campaign_interrupt.interrupt_campaign(
        run,
        acquisition_key=acquisition_key,
        journal_path=tmp_path / "campaign.jsonl",
        receipt_dir=tmp_path / "receipts",
        grid_cache=grid_cache,
        readers=readers,
        lease=lease,
    )


def test_interruption_journals_the_in_flight_step(tmp_path: Path) -> None:
    """AC1: KeyboardInterrupt durably records the in-flight acquisition as interrupted."""
    plans = (_plan("2024-01-01"), _plan("2024-01-02"))
    executor = _InterruptingExecutor(interrupt_day="2024-01-02")

    with pytest.raises(KeyboardInterrupt):
        _run_with_interrupt_handler(
            lambda: run_campaign_days(plans, _Budget(), executor),
            tmp_path=tmp_path,
            acquisition_key="day:2024-01-02:run",
        )

    snapshot = replay_journal(tmp_path / "campaign.jsonl")
    assert snapshot.states["day:2024-01-02:run"] is RunState.INTERRUPTED


class _Reader:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_interruption_releases_weather_grids_and_readers(tmp_path: Path) -> None:
    """AC2: interruption closes all resident grids and every opened local reader."""
    closed_grids: list[object] = []
    cache = DayGridCache[object](
        opener=lambda source_day, field: (source_day, field),
        closer=closed_grids.append,
    )
    cache.acquire("2024-01-01", "temperature")
    cache.acquire("2024-01-01", "wind")
    readers = (_Reader(), _Reader())

    with pytest.raises(KeyboardInterrupt):
        _run_with_interrupt_handler(
            _raise_keyboard_interrupt,
            tmp_path=tmp_path,
            acquisition_key="day:2024-01-01:weather",
            grid_cache=cache,
            readers=readers,
        )

    stats = cache.stats()
    assert isinstance(stats, GridCacheStats)
    assert stats.resident_entries == 0
    assert set(closed_grids) == {
        ("2024-01-01", "temperature"),
        ("2024-01-01", "wind"),
    }
    assert all(reader.closed for reader in readers)


def test_interruption_leaves_a_foreign_lease_record_untouched(tmp_path: Path) -> None:
    """AC3: a stale handle cannot delete or rewrite the current owner's lease."""
    lease_path = tmp_path / "trino.lease"
    stale_handle = acquire_lease(lease_path, owner="campaign", ttl_s=60.0, now=1.0)
    stale_handle.release()
    foreign_handle = acquire_lease(lease_path, owner="other-process", ttl_s=60.0, now=2.0)
    before = lease_path.read_bytes()

    with pytest.raises(KeyboardInterrupt):
        _run_with_interrupt_handler(
            _raise_keyboard_interrupt,
            tmp_path=tmp_path,
            acquisition_key="day:2024-01-01:acquire",
            lease=stale_handle,
        )

    assert lease_path.read_bytes() == before
    foreign_handle.release()


_TRANSITIONS = ("acquired", "staged", "published", "committed", "cleaned")


class _DurableExecutor:
    def __init__(self, root: Path, interrupt_at: str | None = None) -> None:
        self.root = root
        self.interrupt_at = interrupt_at
        self.root.mkdir(parents=True)
        self.acquisition_journal = root / "acquisitions.jsonl"
        self.transition_journal = root / "transitions.jsonl"
        self.published_root = root / "published"

    def estimate_resident_gib(self, plan: DayPlan) -> float:
        return 0.1

    def free_disk_gib(self) -> float:
        return 100.0

    def planned_acquisition_batches(self, plan: DayPlan) -> tuple[str, ...]:
        key = self._acquisition_key(plan)
        return () if key in self._acquisition_keys() else ("batch-0",)

    def acquisition_section(
        self,
        plan: DayPlan,
        batch: str,
    ) -> AbstractContextManager[None]:
        return nullcontext()

    def acquire(self, plan: DayPlan, batch: str) -> None:
        key = self._acquisition_key(plan)
        with self.acquisition_journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"acquisition_key": key}) + "\n")
        self._interrupt_if_requested("acquired")

    def run_day(self, plan: DayPlan) -> str:
        completed = self._completed_transitions()
        for transition in _TRANSITIONS[1:]:
            if transition in completed:
                continue
            if transition == "published":
                self._publish(plan)
            with self.transition_journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"transition": transition}) + "\n")
            self._interrupt_if_requested(transition)
        return "cleaned"

    def _interrupt_if_requested(self, transition: str) -> None:
        if self.interrupt_at == transition:
            self.interrupt_at = None
            raise KeyboardInterrupt

    def _publish(self, plan: DayPlan) -> None:
        self.published_root.mkdir(parents=True, exist_ok=True)
        for key in plan.partition_keys:
            identity = f"{key.cohort}__{key.meta_selection_day}"
            payload = json.dumps(
                {"cohort": key.cohort, "day": key.meta_selection_day},
                sort_keys=True,
                separators=(",", ":"),
            )
            (self.published_root / f"{identity}.json").write_text(payload, encoding="utf-8")

    def _acquisition_key(self, plan: DayPlan) -> str:
        return f"{plan.meta_selection_day}:batch-0"

    def _acquisition_keys(self) -> tuple[str, ...]:
        if not self.acquisition_journal.exists():
            return ()
        return tuple(
            json.loads(line)["acquisition_key"]
            for line in self.acquisition_journal.read_text(encoding="utf-8").splitlines()
        )

    def _completed_transitions(self) -> frozenset[str]:
        if not self.transition_journal.exists():
            return frozenset()
        return frozenset(
            json.loads(line)["transition"]
            for line in self.transition_journal.read_text(encoding="utf-8").splitlines()
        )


def _publication_digests(root: Path) -> Mapping[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.json"))
    }


def test_interrupt_then_resume_matches_a_continuous_run(tmp_path: Path) -> None:
    """AC4: every durable interruption resumes to identical, exactly-once publications."""
    plan = _plan("2024-01-01")
    continuous = _DurableExecutor(tmp_path / "continuous")
    run_campaign_days((plan,), _Budget(), continuous)
    expected_publications = _publication_digests(continuous.published_root)

    for transition in _TRANSITIONS:
        resumed = _DurableExecutor(tmp_path / f"resumed-{transition}", transition)
        interrupt_root = resumed.root / "interrupt"

        with pytest.raises(KeyboardInterrupt):
            _run_with_interrupt_handler(
                partial(run_campaign_days, (plan,), _Budget(), resumed),
                tmp_path=interrupt_root,
                acquisition_key=f"{plan.meta_selection_day}:{transition}",
            )

        run_campaign_days((plan,), _Budget(), resumed)

        assert _publication_digests(resumed.published_root) == expected_publications
        acquisition_keys = resumed._acquisition_keys()
        assert len(acquisition_keys) == len(set(acquisition_keys)) == 1
