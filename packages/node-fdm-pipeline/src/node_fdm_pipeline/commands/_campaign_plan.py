from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from node_fdm_pipeline.commands._fleet_journal import (
    RunSnapshot,
    next_incomplete_step,
    replay_journal,
)
from node_fdm_pipeline.commands._fleet_plan import (
    TRINO_BATCH_SIZE,
    FleetPlan,
    build_fleet_plan,
    discover_cohorts,
    plan_shared_acquisitions,
)
from node_fdm_pipeline.commands._fleet_selection import (
    SelectionPlan,
    load_selection_file,
    utc_days_for_interval,
)
from node_fdm_pipeline.config import PipelineConfig

__all__ = [
    "CampaignPlan",
    "PlannedTrinoBatch",
    "campaign_identities",
    "campaign_plan",
    "crossmidnight_dependencies",
    "estimate_disk_footprint",
    "steps_to_resume",
]

_GIB_BYTES = 1024**3


class PlannedTrinoBatch(BaseModel):
    """One offline projection of a shared Trino acquisition."""

    model_config = ConfigDict(frozen=True)

    utc_day: str
    kind: Literal["history", "extended", "flightlist"]
    batch_index: int
    aircraft: frozenset[str]


class CampaignPlan(BaseModel):
    """Complete, offline description of a recorded campaign."""

    model_config = ConfigDict(frozen=True)

    identified_flights: frozenset[str]
    identified_by_cohort: dict[str, frozenset[str]]
    trino_batches: list[PlannedTrinoBatch]
    crossmidnight_dependencies: list[tuple[str, str]]
    estimated_disk_bytes: int
    steps_to_resume: list[tuple[str, str]]


def campaign_identities(
    selection: SelectionPlan,
) -> tuple[frozenset[str], dict[str, frozenset[str]]]:
    """Return unique recorded identities globally and for each cohort."""
    identified = frozenset(flight.selection_id for flight in selection.flights)
    mutable_by_cohort: dict[str, set[str]] = {}
    for flight in selection.flights:
        for cohort in flight.cohorts:
            mutable_by_cohort.setdefault(cohort, set()).add(flight.selection_id)
    return identified, {
        cohort: frozenset(identities) for cohort, identities in sorted(mutable_by_cohort.items())
    }


def crossmidnight_dependencies(selection: SelectionPlan) -> list[tuple[str, str]]:
    """Return ordered adjacent-day dependencies for cross-midnight flights."""
    dependencies: list[tuple[str, str]] = []
    for flight in selection.flights:
        days = utc_days_for_interval(flight.firstseen, flight.lastseen)
        for predecessor, dependent in pairwise(days):
            dependencies.append(
                (
                    datetime.strptime(predecessor, "%Y%m%d").date().isoformat(),
                    datetime.strptime(dependent, "%Y%m%d").date().isoformat(),
                )
            )
    return dependencies


def estimate_disk_footprint(
    plan: FleetPlan,
    *,
    per_partition_bytes: int,
) -> int:
    """Estimate disk bytes from every planned cohort/day partition."""
    if plan.selection is None:
        return 0
    flight_days = tuple(
        (flight, utc_days_for_interval(flight.firstseen, flight.lastseen))
        for flight in plan.selection.flights
    )
    single_day_starts = tuple(days[0] for _flight, days in flight_days if len(days) == 1)
    campaign_start = min(single_day_starts) if single_day_starts else None
    partitions = {
        (
            cohort,
            days[0] if campaign_start is None else max(days[0], campaign_start),
        )
        for flight, days in flight_days
        for cohort in flight.cohorts
    }
    partition_count = len(partitions)
    return partition_count * per_partition_bytes


def steps_to_resume(
    plan: FleetPlan,
    snapshot: RunSnapshot,
) -> list[tuple[str, str]]:
    """Return each acquisition's next durable step in selection-day order."""
    if plan.selection is None:
        return []
    ordered_flights = sorted(
        plan.selection.flights,
        key=lambda flight: (
            utc_days_for_interval(flight.firstseen, flight.lastseen)[0],
            flight.acquisition_key,
        ),
    )
    steps: list[tuple[str, str]] = []
    seen: set[str] = set()
    for flight in ordered_flights:
        if flight.acquisition_key in seen:
            continue
        seen.add(flight.acquisition_key)
        next_step = next_incomplete_step(snapshot, flight.acquisition_key)
        if next_step is not None:
            steps.append((flight.acquisition_key, next_step.value))
    return steps


def _input_path(config_path: Path, value: Path) -> Path:
    expanded = value.expanduser()
    return expanded if expanded.is_absolute() else config_path.parent / expanded


def _build_campaign_fleet_plan(
    config_path: Path,
    config: PipelineConfig,
    selection: SelectionPlan,
    selection_path: Path,
) -> FleetPlan:
    triples = discover_cohorts(config_path.parent)
    if not triples:
        cohort_names = sorted(
            {cohort for flight in selection.flights for cohort in flight.cohorts}
        )
        triples = [(name, config_path, selection_path) for name in cohort_names]
    return build_fleet_plan(
        triples,
        data_root=config.paths.data_dir,
        selection=selection,
    )


def _planned_trino_batches(plan: FleetPlan) -> list[PlannedTrinoBatch]:
    if plan.selection is None:
        return []
    batches: list[PlannedTrinoBatch] = []
    for acquisition in plan_shared_acquisitions(plan.selection):
        aircraft = sorted(plan.dates.get(acquisition.utc_day, ()))
        for batch_index, offset in enumerate(
            range(0, len(aircraft), TRINO_BATCH_SIZE),
            start=1,
        ):
            batches.append(
                PlannedTrinoBatch(
                    utc_day=acquisition.utc_day,
                    kind=acquisition.kind,
                    batch_index=batch_index,
                    aircraft=frozenset(aircraft[offset : offset + TRINO_BATCH_SIZE]),
                )
            )
    return batches


def campaign_plan(config: Path) -> CampaignPlan:
    """Build the complete campaign plan exclusively from recorded local inputs."""
    config_path = config.expanduser().resolve()
    resolved = PipelineConfig.from_yaml(config_path)
    fleet_run = resolved.fleet_run
    if fleet_run is None:
        raise ValueError("campaign planning requires fleet_run configuration")
    if fleet_run.recorded_source is None:
        raise ValueError("campaign planning requires fleet_run.recorded_source")

    selection_path = _input_path(config_path, fleet_run.recorded_source)
    source = load_selection_file(selection_path)
    if source.plan is None:
        raise ValueError("recorded campaign selection must be structured")
    selection = source.plan
    plan = _build_campaign_fleet_plan(config_path, resolved, selection, selection_path)

    if fleet_run.acquisition_journal is None:
        snapshot = RunSnapshot(states={}, artifacts={})
    else:
        journal_path = _input_path(config_path, fleet_run.acquisition_journal)
        snapshot = (
            replay_journal(journal_path)
            if journal_path.exists()
            else RunSnapshot(states={}, artifacts={})
        )

    identified, identified_by_cohort = campaign_identities(selection)
    per_partition_bytes = int(fleet_run.disk_min_gib * _GIB_BYTES)
    return CampaignPlan(
        identified_flights=identified,
        identified_by_cohort=identified_by_cohort,
        trino_batches=_planned_trino_batches(plan),
        crossmidnight_dependencies=crossmidnight_dependencies(selection),
        estimated_disk_bytes=estimate_disk_footprint(
            plan,
            per_partition_bytes=per_partition_bytes,
        ),
        steps_to_resume=steps_to_resume(plan, snapshot),
    )
