"""Selection-day execution boundaries for fleet pipeline work."""

from __future__ import annotations

from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from node_fdm_pipeline.commands._fleet_selection import SelectionPlan, utc_days_for_interval

__all__ = [
    "DayPartitionKey",
    "DayPlan",
    "DayScopeViolation",
    "build_day_plan",
    "require_in_day_scope",
    "require_selection_in_scope",
]


class DayPartitionKey(NamedTuple):
    """Immutable cohort and selection-day partition identity."""

    cohort: str
    meta_selection_day: str


class DayPlan(BaseModel):
    """Immutable boundary for all reads belonging to one selection day."""

    model_config = ConfigDict(frozen=True)

    meta_selection_day: str
    partition_keys: tuple[DayPartitionKey, ...]
    selection_ids: frozenset[str]
    selection_ids_by_key: dict[DayPartitionKey, frozenset[str]]
    source_days: tuple[str, ...]


class DayScopeViolation(ValueError):  # noqa: N818
    """Raised when a requested partition or selection is outside a day plan."""


def build_day_plan(selection: SelectionPlan, meta_selection_day: str) -> DayPlan:
    """Materialise the exact cohort, selection, and source-day scope for one day."""
    flights = tuple(
        flight for flight in selection.flights if meta_selection_day in flight.utc_days
    )
    ids_by_key: dict[DayPartitionKey, set[str]] = {}
    for flight in flights:
        for cohort in flight.cohorts:
            key = DayPartitionKey(cohort, meta_selection_day)
            ids_by_key.setdefault(key, set()).add(flight.selection_id)

    partition_keys = tuple(sorted(ids_by_key))
    source_days = tuple(
        sorted(
            {
                source_day
                for flight in flights
                for source_day in utc_days_for_interval(flight.firstseen, flight.lastseen)
            }
        )
    )
    return DayPlan(
        meta_selection_day=meta_selection_day,
        partition_keys=partition_keys,
        selection_ids=frozenset(flight.selection_id for flight in flights),
        selection_ids_by_key={key: frozenset(ids_by_key[key]) for key in partition_keys},
        source_days=source_days,
    )


def require_in_day_scope(
    plan: DayPlan,
    *,
    cohort: str,
    meta_selection_day: str,
) -> None:
    """Reject a cohort/day partition not declared by the active day plan."""
    key = DayPartitionKey(cohort, meta_selection_day)
    if key not in plan.partition_keys:
        raise DayScopeViolation(
            f"partition ({cohort!r}, {meta_selection_day!r}) is outside the active day plan"
        )


def require_selection_in_scope(plan: DayPlan, selection_id: str) -> None:
    """Reject a selection identifier not declared by the active day plan."""
    if selection_id not in plan.selection_ids:
        raise DayScopeViolation(f"selection {selection_id!r} is outside the active day plan")
