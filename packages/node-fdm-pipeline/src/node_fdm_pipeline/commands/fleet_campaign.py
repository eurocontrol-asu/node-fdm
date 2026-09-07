from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from node_fdm_pipeline.commands._campaign_guard import CampaignMode, resolve_campaign_mode
from node_fdm_pipeline.commands._campaign_guard import preflight_campaign as _preflight_campaign
from node_fdm_pipeline.commands._campaign_interrupt import interrupt_campaign
from node_fdm_pipeline.commands._campaign_plan import CampaignPlan, campaign_plan
from node_fdm_pipeline.commands._campaign_report import (
    CampaignReport,
    campaign_status,
    campaign_validate,
)
from node_fdm_pipeline.commands._campaign_runner import (
    CampaignRunReport,
)
from node_fdm_pipeline.commands._campaign_runner import (
    run_campaign_days as _run_campaign_days,
)
from node_fdm_pipeline.commands._day_plan import DayPlan, build_day_plan
from node_fdm_pipeline.commands._fleet_digest import record_campaign_identity
from node_fdm_pipeline.commands._fleet_journal import RunSnapshot, RunState, next_incomplete_step
from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan, load_selection_file
from node_fdm_pipeline.config import FleetRunConfig, PipelineConfig

__all__ = ["run_fleet_campaign"]

_GIB_BYTES = 1024**3
_LOCAL_PROFILE = {"worker": "local"}


@dataclass
class _CampaignBudget:
    local_workers: int
    max_resident_gib: float
    min_free_gib: float


@dataclass(frozen=True)
class _JournalExecutor:
    journal_path: Path

    def estimate_resident_gib(self, plan: DayPlan) -> float:
        del plan
        return 0.0

    def free_disk_gib(self) -> float:
        return shutil.disk_usage(self.journal_path.parent).free / _GIB_BYTES

    def planned_acquisition_batches(self, plan: DayPlan) -> tuple[str, ...]:
        del plan
        return ()

    def acquisition_section(
        self,
        plan: DayPlan,
        batch: str,
    ) -> AbstractContextManager[None]:
        del plan, batch
        return nullcontext()

    def acquire(self, plan: DayPlan, batch: str) -> None:
        del plan, batch

    def run_day(self, plan: DayPlan) -> str:
        day = plan.meta_selection_day
        append_event(
            self.journal_path,
            {"event": "day_started", "meta_selection_day": day},
        )
        append_event(self.journal_path, {"event": "cleanup_completed", "day": day})
        return "completed"


@dataclass(frozen=True)
class _PreparedRun:
    plans: tuple[DayPlan, ...]
    budget: _CampaignBudget
    executor: _JournalExecutor

    def __iter__(self) -> Iterator[DayPlan]:
        return iter(self.plans)


def run_campaign_days(run: _PreparedRun) -> CampaignRunReport:
    """Delegate one fully prepared invocation to the campaign scheduler."""
    return _run_campaign_days(run.plans, run.budget, run.executor)


def _resolve_path(config_path: Path, value: Path) -> Path:
    return value if value.is_absolute() else config_path.parent / value


def _live_context(
    config: Path,
    mode: CampaignMode,
) -> tuple[Path, FleetRunConfig, SelectionPlan, Path, Path]:
    config_path = config.expanduser().resolve()
    resolved = PipelineConfig.from_yaml(config_path)
    fleet_run = resolved.fleet_run
    if fleet_run is None:
        raise ValueError("campaign execution requires fleet_run configuration")
    if fleet_run.recorded_source is None:
        raise ValueError("campaign execution requires fleet_run.recorded_source")
    if fleet_run.acquisition_journal is None:
        raise ValueError("campaign execution requires fleet_run.acquisition_journal")
    if fleet_run.acquisition_receipt_dir is None:
        raise ValueError("campaign execution requires fleet_run.acquisition_receipt_dir")

    selection_path = _resolve_path(config_path, fleet_run.recorded_source)
    source = load_selection_file(selection_path)
    if source.plan is None:
        raise ValueError("recorded campaign selection must be structured")

    journal_path = _resolve_path(config_path, fleet_run.acquisition_journal)
    receipt_dir = _resolve_path(config_path, fleet_run.acquisition_receipt_dir)
    digest = _preflight_campaign(
        mode=mode,
        state_dir=config_path.parent,
        selection_digest=source.plan.digest,
        resolved_config=fleet_run,
        profile=_LOCAL_PROFILE,
    )
    if mode is CampaignMode.RUN:
        record_campaign_identity(config_path.parent, digest)
    return config_path, fleet_run, source.plan, journal_path, receipt_dir


def _day_plans(selection: SelectionPlan) -> tuple[DayPlan, ...]:
    days = sorted({day for flight in selection.flights for day in flight.utc_days})
    return tuple(build_day_plan(selection, day) for day in days)


def _resume_snapshot(journal_path: Path) -> RunSnapshot:
    states: dict[str, RunState] = {}
    if journal_path.exists():
        for event in read_events(journal_path):
            day = event.get("day")
            if event.get("event") == "cleanup_completed" and isinstance(day, str):
                states[day] = RunState.CLEANED
                continue
            started_day = event.get("meta_selection_day")
            if event.get("event") == "day_started" and isinstance(started_day, str):
                states.setdefault(started_day, RunState.ACQUIRING)
    return RunSnapshot(states=states, artifacts={})


def _plans_for_mode(
    plans: Iterable[DayPlan],
    mode: CampaignMode,
    journal_path: Path,
) -> tuple[DayPlan, ...]:
    materialized = tuple(plans)
    if mode is not CampaignMode.RESUME:
        return materialized
    snapshot = _resume_snapshot(journal_path)
    return tuple(
        plan
        for plan in materialized
        if next_incomplete_step(snapshot, plan.meta_selection_day) is not None
    )


def _run_live(config: Path, mode: CampaignMode) -> CampaignRunReport:
    _config_path, fleet_run, selection, journal_path, receipt_dir = _live_context(config, mode)
    plans = _plans_for_mode(_day_plans(selection), mode, journal_path)
    min_free_gib = fleet_run.min_free_gib
    if min_free_gib is None:
        raise ValueError("campaign execution requires fleet_run.min_free_gib")
    budget = _CampaignBudget(
        local_workers=1,
        max_resident_gib=max(fleet_run.disk_min_gib, 1.0),
        min_free_gib=min_free_gib,
    )
    executor = _JournalExecutor(journal_path)
    prepared = _PreparedRun(plans=plans, budget=budget, executor=executor)
    return interrupt_campaign(
        lambda: run_campaign_days(prepared),
        acquisition_key="campaign",
        journal_path=journal_path,
        receipt_dir=receipt_dir,
    )


def run_fleet_campaign(
    config: Path,
    mode: str,
) -> CampaignPlan | CampaignReport | CampaignRunReport:
    """Run exactly one public campaign contract for the requested mode."""
    campaign_mode = resolve_campaign_mode(mode)
    if campaign_mode is CampaignMode.PLAN:
        return campaign_plan(config)
    if campaign_mode is CampaignMode.STATUS:
        return campaign_status(config)
    if campaign_mode is CampaignMode.VALIDATE:
        return campaign_validate(config)
    return _run_live(config, campaign_mode)
