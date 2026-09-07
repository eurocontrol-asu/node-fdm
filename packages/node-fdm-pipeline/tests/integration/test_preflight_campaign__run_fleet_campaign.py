from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline import config as config_module
from node_fdm_pipeline.commands._campaign_runner import CampaignRunReport
from node_fdm_pipeline.commands._day_plan import DayPlan
from node_fdm_pipeline.commands._fleet_digest import (
    compute_resume_digest,
    load_campaign_identity,
    record_campaign_identity,
)
from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events
from node_fdm_pipeline.commands._fleet_selection import load_selection_file

pytestmark = pytest.mark.integration

_SELECTION_CSV = """selection_id,icao24,callsign,start,end,cohorts,utc_days
flight-a,abc123,AXM1,1704110400,1704111000,alpha,2024-01-01
flight-b,def456,AXM2,1704196800,1704197400,alpha,2024-01-02
flight-c,ghi789,AXM3,1704283200,1704283800,alpha,2024-01-03
"""


def _fleet_campaign() -> ModuleType:
    """Resolve the ticket's new public module inside each collected test."""
    return importlib.import_module("node_fdm_pipeline.commands.fleet_campaign")


@pytest.fixture
def campaign_files(
    tmp_path: Path,
    make_config: Callable[..., Path],
) -> tuple[Path, Path, Path, tuple[str, ...]]:
    """Create a fully local three-day campaign and its durable state paths."""
    selection_path = tmp_path / "recorded-selection.csv"
    selection_path.write_text(_SELECTION_CSV, encoding="utf-8")
    journal_path = tmp_path / "campaign.journal.jsonl"
    receipt_dir = tmp_path / "receipts"
    config_path = make_config(
        tmp_path / "campaign.yaml",
        tmp_path / "data",
        extra=f"""
fleet_run:
  lease_path: "{tmp_path / "campaign.lease"}"
  lease_ttl_s: 60
  disk_min_gib: 0.000001
  min_free_gib: 0.000001
  recorded_source: "{selection_path}"
  acquisition_journal: "{journal_path}"
  acquisition_receipt_dir: "{receipt_dir}"
""",
    )
    source = load_selection_file(selection_path)
    assert source.plan is not None
    days = tuple(sorted({day for flight in source.plan.flights for day in flight.utc_days}))
    return config_path, journal_path, receipt_dir, days


def _completed_report(plans: Iterable[DayPlan]) -> CampaignRunReport:
    days = tuple(plan.meta_selection_day for plan in plans)
    return CampaignRunReport(
        started_days=days,
        completed_days=days,
        blocked_reasons=(),
        blocking_day=None,
        max_observed_concurrency=1,
    )


def test_run_refuses_incompatible_state_without_journalling(
    campaign_files: tuple[Path, Path, Path, tuple[str, ...]],
) -> None:
    """AC2: an incompatible run identity is refused before the journal changes."""
    campaign = _fleet_campaign()
    config_path, journal_path, _receipt_dir, _days = campaign_files
    resolved = config_module.PipelineConfig.from_yaml(config_path)
    assert resolved.fleet_run is not None
    recorded = compute_resume_digest(
        "different-selection",
        resolved.fleet_run,
        {"worker": "local"},
    )
    record_campaign_identity(config_path.parent, recorded)
    before = journal_path.read_bytes() if journal_path.exists() else None

    with pytest.raises(RuntimeError) as exc_info:
        campaign.run_fleet_campaign(config_path, "run")

    assert type(exc_info.value).__name__ == "CampaignStateIncompatible"
    after = journal_path.read_bytes() if journal_path.exists() else None
    assert after == before


def test_run_records_identity_and_covers_every_selection_day(
    campaign_files: tuple[Path, Path, Path, tuple[str, ...]],
    mocker: MockerFixture,
) -> None:
    """AC3: a clean run records its identity and reports every planned day."""
    campaign = _fleet_campaign()
    config_path, _journal_path, _receipt_dir, days = campaign_files
    runner = mocker.patch.object(campaign, "run_campaign_days", side_effect=_completed_report)

    report = campaign.run_fleet_campaign(config_path, "run")

    load_campaign_identity(config_path.parent)
    assert report.completed_days == days
    plans = tuple(runner.call_args.args[0])
    assert tuple(plan.meta_selection_day for plan in plans) == days


def test_resume_skips_completed_day_and_starts_at_next_incomplete_step(
    campaign_files: tuple[Path, Path, Path, tuple[str, ...]],
    mocker: MockerFixture,
) -> None:
    """AC4: resume does not emit a second start for an already completed day."""
    campaign = _fleet_campaign()
    config_path, journal_path, _receipt_dir, days = campaign_files
    first_day = days[0]
    initial_runner = mocker.patch.object(
        campaign,
        "run_campaign_days",
        side_effect=_completed_report,
    )
    campaign.run_fleet_campaign(config_path, "run")
    initial_runner.reset_mock()
    append_event(journal_path, {"event": "day_started", "meta_selection_day": first_day})
    append_event(journal_path, {"event": "cleanup_completed", "day": first_day})

    def record_resumed_starts(plans: Iterable[DayPlan], *_args: object) -> CampaignRunReport:
        materialized = tuple(plans)
        for plan in materialized:
            append_event(
                journal_path,
                {"event": "day_started", "meta_selection_day": plan.meta_selection_day},
            )
        return _completed_report(materialized)

    initial_runner.side_effect = record_resumed_starts
    report = campaign.run_fleet_campaign(config_path, "resume")

    started = [
        event["meta_selection_day"]
        for event in read_events(journal_path)
        if event.get("event") == "day_started"
    ]
    assert started.count(first_day) == 1
    assert report.started_days == days[1:]
