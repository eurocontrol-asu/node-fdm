"""Integration coverage for canonical campaign selection loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import pytest

from _config_fixtures import write_config
from node_fdm_pipeline import cli
from node_fdm_pipeline.commands._fleet_selection import SelectionFormatError

pytestmark = pytest.mark.integration

_EXPECTED_DATES = {
    "20191231": ["a00001"],
    "20200101": ["a00001"],
}
_CSV_HEADER = "selection_id,icao24,callsign,firstseen,lastseen,msn,split,cohorts,utc_days"


class _ObservedPlan(Protocol):
    dates: dict[str, list[str]]
    selection: object | None


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def __call__(self, plan: object, *args: object, **kwargs: object) -> list[object]:
        self.calls.append(plan)
        return []


@dataclass(frozen=True)
class _Campaign:
    root: Path
    fleet_dir: Path
    data_root: Path
    resolved_config: Path
    profile: Path
    lease_path: Path
    acquisition_journal: Path
    acquisition_receipt_dir: Path
    recorder: _Recorder

    def selection_file(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def run(self, selection: Path) -> _ObservedPlan:
        cli.download_fleet(
            fleet_dir=self.fleet_dir,
            workers=1,
            data_root=self.data_root,
            force_refresh=False,
            selection=selection,
            resolved_config=self.resolved_config,
            profile=self.profile,
            lease_path=self.lease_path,
            lease_ttl_s=60,
            disk_min_gib=0.001,
            recorded_source=None,
            acquisition_journal=self.acquisition_journal,
            acquisition_receipt_dir=self.acquisition_receipt_dir,
        )
        return cast("_ObservedPlan", self.recorder.calls[-1])


def _selection_rows() -> list[dict[str, object]]:
    row: dict[str, object] = {
        "selection_id": "sel-alpha",
        "icao24": "a00001",
        "callsign": "ALPHA1",
        "firstseen": 1577835000,
        "lastseen": 1577838600,
        "msn": "M1",
        "split": "train",
        "utc_days": list(_EXPECTED_DATES),
    }
    return [{**row, "cohort": cohort} for cohort in ("C1", "C2")]


def _valid_csv() -> str:
    rows = [
        "sel-alpha,a00001,ALPHA1,1577835000,1577838600,M1,train,C1,20191231|20200101",
        "sel-alpha,a00001,ALPHA1,1577835000,1577838600,M1,train,C2,20191231|20200101",
    ]
    return "\n".join([_CSV_HEADER, *rows, ""])


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Campaign:
    fleet_dir = tmp_path / "fleet"
    legacy = (
        ("C1", "b00003", "2020-01-05"),
        ("C2", "c00004", "2020-01-06"),
    )
    for cohort, aircraft, day in legacy:
        cohort_dir = fleet_dir / "types" / cohort
        results_dir = cohort_dir / "results"
        results_dir.mkdir(parents=True)
        write_config(cohort_dir / "config.yaml", tmp_path / "legacy-data" / cohort)
        (results_dir / f"selection_{cohort}.csv").write_text(
            f"icao24,day\n{aircraft},{day}\n",
            encoding="utf-8",
        )

    resolved_config = tmp_path / "resolved-config.json"
    profile = tmp_path / "profile.json"
    resolved_config.write_text('{"workers":1}', encoding="utf-8")
    profile.write_text('{"campaign":"alpha"}', encoding="utf-8")

    shared = tmp_path / "shared"
    receipt_dir = shared / "receipts"
    receipt_dir.mkdir(parents=True)
    recorder = _Recorder()
    monkeypatch.setattr(
        "node_fdm_pipeline.commands._fleet_fetch.download_fleet",
        recorder,
    )
    return _Campaign(
        root=tmp_path,
        fleet_dir=fleet_dir,
        data_root=tmp_path / "staging",
        resolved_config=resolved_config,
        profile=profile,
        lease_path=shared / "download-fleet.lease",
        acquisition_journal=shared / "acquisition.jsonl",
        acquisition_receipt_dir=receipt_dir,
        recorder=recorder,
    )


def test_valid_campaign_csv_drives_fleet_plan_dates(campaign: _Campaign) -> None:
    """AC1: a valid campaign CSV exclusively determines the fleet plan dates."""
    plan = campaign.run(campaign.selection_file("selection.csv", _valid_csv()))

    assert len(campaign.recorder.calls) == 1
    assert plan.selection is not None
    assert plan.dates == _EXPECTED_DATES


def test_equivalent_json_and_csv_yield_equal_fleet_plans(campaign: _Campaign) -> None:
    """AC2: equivalent JSON and CSV selections produce the same fleet plan."""
    csv_plan = campaign.run(campaign.selection_file("selection.csv", _valid_csv()))
    json_plan = campaign.run(
        campaign.selection_file("selection.json", json.dumps(_selection_rows()))
    )

    assert json_plan.dates == _EXPECTED_DATES
    assert json_plan == csv_plan


def test_csv_without_selection_id_raises_before_boundary(campaign: _Campaign) -> None:
    """AC3: a CSV lacking selection_id fails before the download boundary."""
    invalid = _valid_csv().replace("selection_id,", "", 1)
    selection = campaign.selection_file("selection.csv", invalid)

    with pytest.raises(SelectionFormatError) as raised:
        campaign.run(selection)

    assert raised.type is SelectionFormatError
    assert campaign.recorder.calls == []


def test_unterminated_csv_quote_raises_before_boundary(campaign: _Campaign) -> None:
    """AC4: an unterminated CSV quote fails before the download boundary."""
    invalid = (
        f"{_CSV_HEADER}\n"
        'sel-alpha,a00001,ALPHA1,1577835000,1577838600,M1,train,C1,"20191231|20200101\n'
    )
    selection = campaign.selection_file("selection.csv", invalid)

    with pytest.raises(SelectionFormatError) as raised:
        campaign.run(selection)

    assert raised.type is SelectionFormatError
    assert campaign.recorder.calls == []


def test_invalid_json_raises_before_boundary(campaign: _Campaign) -> None:
    """AC5: syntactically invalid JSON fails before the download boundary."""
    selection = campaign.selection_file(
        "selection.json",
        '[{"selection_id": "sel-alpha"',
    )

    with pytest.raises(SelectionFormatError) as raised:
        campaign.run(selection)

    assert raised.type is SelectionFormatError
    assert campaign.recorder.calls == []
