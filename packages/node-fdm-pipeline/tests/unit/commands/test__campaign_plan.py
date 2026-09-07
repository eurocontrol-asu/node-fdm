from __future__ import annotations

from node_fdm_pipeline.commands._fleet_journal import RunSnapshot, RunState
from node_fdm_pipeline.commands._fleet_plan import FleetPlan
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan


def _flight(
    *,
    selection_id: str,
    acquisition_key: str,
    timestamps: tuple[int, int],
    cohorts: frozenset[str],
    utc_days: tuple[str, ...],
) -> SelectedFlight:
    return SelectedFlight(
        icao24=f"icao-{selection_id}",
        callsign=f"CALL-{selection_id}",
        firstseen=timestamps[0],
        lastseen=timestamps[1],
        msn=f"msn-{selection_id}",
        split="train",
        cohorts=cohorts,
        selection_id=selection_id,
        utc_days=utc_days,
        acquisition_key=acquisition_key,
    )


def _fleet_plan(selection: SelectionPlan) -> FleetPlan:
    dates: dict[str, list[str]] = {}
    for flight in selection.flights:
        for day in flight.utc_days:
            dates.setdefault(day, []).append(flight.icao24)
    return FleetPlan(dates=dates, owner={}, cohorts=(), selection=selection)


def test_campaign_identities_deduplicate_overall_and_per_cohort() -> None:
    """AC1: identities are unique globally and within every owning cohort."""
    from node_fdm_pipeline.commands import _campaign_plan

    repeated_first_day = _flight(
        selection_id="flight-a",
        acquisition_key="flight-a-day-1",
        timestamps=(1_704_067_200, 1_704_067_800),
        cohorts=frozenset({"alpha", "beta"}),
        utc_days=("2024-01-01",),
    )
    repeated_second_day = repeated_first_day.model_copy(
        update={
            "acquisition_key": "flight-a-day-2",
            "utc_days": ("2024-01-02",),
        }
    )
    alpha_only = _flight(
        selection_id="flight-b",
        acquisition_key="flight-b-day-2",
        timestamps=(1_704_153_600, 1_704_154_200),
        cohorts=frozenset({"alpha"}),
        utc_days=("2024-01-02",),
    )
    selection = SelectionPlan(
        flights=(repeated_first_day, repeated_second_day, alpha_only),
        digest="identity-fixture",
    )

    identified, by_cohort = _campaign_plan.campaign_identities(selection)

    assert identified == frozenset({"flight-a", "flight-b"})
    assert by_cohort == {
        "alpha": frozenset({"flight-a", "flight-b"}),
        "beta": frozenset({"flight-a"}),
    }


def test_crossmidnight_dependencies_orders_predecessor_before_dependent() -> None:
    """AC3: a flight spanning UTC midnight creates an ordered day dependency."""
    from node_fdm_pipeline.commands import _campaign_plan

    selection = SelectionPlan(
        flights=(
            _flight(
                selection_id="overnight",
                acquisition_key="overnight-key",
                timestamps=(1_704_153_300, 1_704_153_900),
                cohorts=frozenset({"alpha"}),
                utc_days=("2024-01-01", "2024-01-02"),
            ),
        ),
        digest="overnight-fixture",
    )

    assert _campaign_plan.crossmidnight_dependencies(selection) == [("2024-01-01", "2024-01-02")]


def test_crossmidnight_dependencies_omit_same_day_window() -> None:
    """AC3: a flight wholly inside one UTC day creates no dependency."""
    from node_fdm_pipeline.commands import _campaign_plan

    selection = SelectionPlan(
        flights=(
            _flight(
                selection_id="same-day",
                acquisition_key="same-day-key",
                timestamps=(1_704_110_400, 1_704_111_000),
                cohorts=frozenset({"alpha"}),
                utc_days=("2024-01-01",),
            ),
        ),
        digest="same-day-fixture",
    )

    assert _campaign_plan.crossmidnight_dependencies(selection) == []


def test_estimate_disk_footprint_multiplies_planned_partitions() -> None:
    """AC4: footprint equals every planned cohort/day partition times its estimate."""
    from node_fdm_pipeline.commands import _campaign_plan

    selection = SelectionPlan(
        flights=(
            _flight(
                selection_id="first",
                acquisition_key="first-key",
                timestamps=(1_704_110_400, 1_704_111_000),
                cohorts=frozenset({"alpha", "beta"}),
                utc_days=("2024-01-01",),
            ),
            _flight(
                selection_id="second",
                acquisition_key="second-key",
                timestamps=(1_704_196_800, 1_704_197_400),
                cohorts=frozenset({"alpha", "beta"}),
                utc_days=("2024-01-02",),
            ),
        ),
        digest="partition-fixture",
    )

    assert (
        _campaign_plan.estimate_disk_footprint(
            _fleet_plan(selection),
            per_partition_bytes=1_024,
        )
        == 4_096
    )


def test_steps_to_resume_fall_back_to_full_selection_day_order() -> None:
    """AC5: an empty journal schedules every acquisition by ascending selection day."""
    from node_fdm_pipeline.commands import _campaign_plan

    later = _flight(
        selection_id="later",
        acquisition_key="later-key",
        timestamps=(1_704_196_800, 1_704_197_400),
        cohorts=frozenset({"alpha"}),
        utc_days=("2024-01-02",),
    )
    earlier = _flight(
        selection_id="earlier",
        acquisition_key="earlier-key",
        timestamps=(1_704_110_400, 1_704_111_000),
        cohorts=frozenset({"alpha"}),
        utc_days=("2024-01-01",),
    )
    plan = _fleet_plan(SelectionPlan(flights=(later, earlier), digest="resume-order-fixture"))
    snapshot = RunSnapshot(states={}, artifacts={})

    assert _campaign_plan.steps_to_resume(plan, snapshot) == [
        ("earlier-key", RunState.ACQUIRING.value),
        ("later-key", RunState.ACQUIRING.value),
    ]
