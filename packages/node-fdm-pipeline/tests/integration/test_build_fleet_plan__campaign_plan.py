from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands._fleet_journal import (
    JournalEvent,
    RunState,
    record_state,
)
from node_fdm_pipeline.commands._fleet_plan import plan_shared_acquisitions
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan, load_selection_file

pytestmark = pytest.mark.integration

_SELECTION_CSV = """selection_id,icao24,callsign,start,end,cohorts,utc_days
flight-a,abc123,AXM1,1704153300,1704153900,alpha|beta,2024-01-01|2024-01-02
flight-b,def456,AXM2,1704196800,1704197400,alpha,2024-01-02
"""


@pytest.fixture
def campaign_fixture(
    tmp_path: Path,
    make_config: Callable[..., Path],
) -> tuple[Path, SelectionPlan, Path, Path]:
    selection_path = tmp_path / "recorded-selection.csv"
    selection_path.write_text(_SELECTION_CSV)
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
  recorded_source: "{selection_path}"
  acquisition_journal: "{journal_path}"
  acquisition_receipt_dir: "{receipt_dir}"
""",
    )
    source = load_selection_file(selection_path)
    assert source.plan is not None
    return config_path, source.plan, journal_path, receipt_dir


def test_campaign_plan_reports_recorded_identities(
    campaign_fixture: tuple[Path, SelectionPlan, Path, Path],
) -> None:
    """AC1: the report exposes deduplicated recorded identities per campaign/cohort."""
    from node_fdm_pipeline.commands import _campaign_plan

    config_path, _selection, _journal_path, _receipt_dir = campaign_fixture

    report = _campaign_plan.campaign_plan(config_path)

    assert report.identified_flights == frozenset({"flight-a", "flight-b"})
    assert report.identified_by_cohort == {
        "alpha": frozenset({"flight-a", "flight-b"}),
        "beta": frozenset({"flight-a"}),
    }


def test_campaign_plan_reports_one_batch_per_shared_acquisition(
    campaign_fixture: tuple[Path, SelectionPlan, Path, Path],
) -> None:
    """AC2: every shared acquisition becomes one dated batch with aircraft membership."""
    from node_fdm_pipeline.commands import _campaign_plan

    config_path, selection, _journal_path, _receipt_dir = campaign_fixture
    expected_acquisitions = plan_shared_acquisitions(selection)

    report = _campaign_plan.campaign_plan(config_path)
    batches = {(batch.utc_day, batch.kind): batch.aircraft for batch in report.trino_batches}
    expected = {
        (acquisition.utc_day, acquisition.kind): frozenset(
            flight.icao24 for flight in selection.flights if acquisition.utc_day in flight.utc_days
        )
        for acquisition in expected_acquisitions
    }

    assert batches == expected
    assert len(report.trino_batches) == len(expected_acquisitions)


def test_campaign_plan_resumes_at_next_incomplete_journal_step(
    campaign_fixture: tuple[Path, SelectionPlan, Path, Path],
) -> None:
    """AC5: the journal advances an acquisition to its next incomplete step."""
    from node_fdm_pipeline.commands import _campaign_plan

    config_path, selection, journal_path, receipt_dir = campaign_fixture
    first = min(selection.flights, key=lambda flight: flight.utc_days[0])
    record_state(
        journal_path,
        receipt_dir,
        JournalEvent(
            acquisition_key=first.acquisition_key,
            state=RunState.ACQUIRING,
            timestamp=datetime(2024, 1, 3, tzinfo=UTC),
            receipt={"selection_id": first.selection_id},
        ),
    )

    report = _campaign_plan.campaign_plan(config_path)

    assert report.steps_to_resume[0] == (
        first.acquisition_key,
        RunState.PROCESSING.value,
    )


def test_campaign_plan_never_constructs_remote_client(
    campaign_fixture: tuple[Path, SelectionPlan, Path, Path],
    mocker: MockerFixture,
) -> None:
    """AC6: the complete plan remains available while every remote factory is faulted."""
    from node_fdm_pipeline.commands import _campaign_plan

    config_path, _selection, _journal_path, _receipt_dir = campaign_fixture
    raising_factory = mocker.patch(
        "node_fdm_pipeline.commands._fleet_fetch._get_opensky",
        side_effect=RuntimeError("remote construction is forbidden in plan mode"),
    )

    report = _campaign_plan.campaign_plan(config_path)

    assert report.identified_flights == frozenset({"flight-a", "flight-b"})
    assert report.trino_batches
    assert report.crossmidnight_dependencies == [("2024-01-01", "2024-01-02")]
    assert report.estimated_disk_bytes > 0
    assert report.steps_to_resume
    raising_factory.assert_not_called()
