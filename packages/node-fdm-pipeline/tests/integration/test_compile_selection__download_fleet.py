"""Integration coverage for campaign-selection-driven fleet downloads."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from _config_fixtures import write_config
from node_fdm_pipeline.cli import download_fleet
from node_fdm_pipeline.commands._fleet_manifest import read_events

pytestmark = pytest.mark.integration

_DAYS = ("20191231", "20200101")
_KINDS = ("history", "extended", "flightlist")


def _selection_rows(utc_days: tuple[str, ...]) -> list[dict[str, object]]:
    alpha: dict[str, object] = {
        "icao24": "a00001",
        "callsign": "ALPHA1",
        "firstseen": 1577835000,
        "lastseen": 1577838600,
        "msn": "M1",
        "split": "train",
        "selection_id": "sel-alpha",
        "utc_days": list(utc_days),
        "acquisition_key": "acq-alpha",
    }
    return [{**alpha, "cohort": cohort} for cohort in ("C1", "C2")]


def _write_fleet(case_root: Path) -> Path:
    fleet_dir = case_root / "fleet"
    for cohort in ("C1", "C2"):
        cohort_dir = fleet_dir / "types" / cohort
        results_dir = cohort_dir / "results"
        results_dir.mkdir(parents=True)
        write_config(cohort_dir / "config.yaml", case_root / "legacy-data" / cohort)
        (results_dir / f"selection_{cohort}.csv").write_text(
            "icao24,day\nb00003,2020-01-05\n",
            encoding="utf-8",
        )
    return fleet_dir


def _write_recorded_source(case_root: Path) -> Path:
    source = case_root / "recorded-opensky.json"
    rows = [
        {"icao24": "a00001", "timestamp": 1577835000},
        {"icao24": "b00003", "timestamp": 1578182400},
    ]
    source.write_text(
        json.dumps(
            {
                "delay_s": 0.0,
                "responses": {
                    "history": rows,
                    "extended": rows,
                    "flightlist": rows,
                },
            }
        ),
        encoding="utf-8",
    )
    return source


def _run_campaign(
    case_root: Path,
    *,
    utc_days: tuple[str, ...],
) -> tuple[Path, list[dict[str, object]]]:
    case_root.mkdir()
    fleet_dir = _write_fleet(case_root)
    selection = case_root / "selection.json"
    resolved_config = case_root / "resolved-config.json"
    profile = case_root / "profile.json"
    selection.write_text(json.dumps(_selection_rows(utc_days)), encoding="utf-8")
    resolved_config.write_text('{"workers":1}', encoding="utf-8")
    profile.write_text('{"campaign":"alpha"}', encoding="utf-8")
    shared = case_root / "shared"
    shared.mkdir()

    download_fleet(
        fleet_dir=fleet_dir,
        workers=1,
        data_root=case_root / "staging",
        force_refresh=True,
        selection=selection,
        resolved_config=resolved_config,
        profile=profile,
        lease_path=shared / "download-fleet.lease",
        lease_ttl_s=60,
        disk_min_gib=0.001,
        recorded_source=_write_recorded_source(case_root),
        acquisition_journal=shared / "acquisition.jsonl",
        acquisition_receipt_dir=shared / "receipts",
    )

    manifest = next(case_root.rglob("download-fleet.manifest.jsonl"))
    events = read_events(manifest)
    attempts = [event for event in events if event.get("event") == "fetch_attempt"]
    return case_root, attempts


def _artifact_pair(path: Path) -> tuple[str, str]:
    kind = next(part for part in path.parts if part in _KINDS)
    day_part = next(part for part in path.parts if part.startswith("date="))
    return day_part.removeprefix("date=").removesuffix(".parquet"), kind


def test_campaign_selection_is_the_only_source_of_acquisition_identities(
    tmp_path: Path,
) -> None:
    """AC1: acquisition attempts contain only campaign-selected aircraft and UTC days."""
    _root, attempts = _run_campaign(tmp_path / "campaign", utc_days=_DAYS)

    observed = {
        (str(attempt["date"]), tuple(cast("list[str]", attempt["batch"]))) for attempt in attempts
    }

    assert observed == {(day, ("a00001",)) for day in _DAYS}
    assert all("b00003" not in batch for _day, batch in observed)
    assert all(str(attempt["date"]) != "20200105" for attempt in attempts)


def test_campaign_stages_one_owned_artifact_per_day_and_kind(tmp_path: Path) -> None:
    """AC2: kinds stay ordered and each shared staging artifact has both owners."""
    root, attempts = _run_campaign(tmp_path / "campaign", utc_days=_DAYS)

    for day in _DAYS:
        assert [str(attempt["kind"]) for attempt in attempts if attempt["date"] == day] == [
            "history",
            "extended",
            "flightlist",
        ]

    artifact_counts = Counter(_artifact_pair(path) for path in root.rglob("*.parquet"))
    assert artifact_counts == Counter({(day, kind): 1 for day in _DAYS for kind in _KINDS})

    ledgers = list(root.rglob("consumers.json"))
    assert len(ledgers) == len(_DAYS) * len(_KINDS)
    assert all(
        json.loads(path.read_text(encoding="utf-8"))
        == {
            "C1": "pending",
            "C2": "pending",
        }
        for path in ledgers
    )


def test_rewriting_only_utc_days_changes_the_next_run_trace(tmp_path: Path) -> None:
    """AC3: changing only alpha.utc_days changes the next run's requested UTC days."""
    _first_root, first_attempts = _run_campaign(
        tmp_path / "first",
        utc_days=_DAYS,
    )
    _second_root, second_attempts = _run_campaign(
        tmp_path / "second",
        utc_days=("20200102",),
    )

    assert {str(attempt["date"]) for attempt in first_attempts} == set(_DAYS)
    assert {str(attempt["date"]) for attempt in second_attempts} == {"20200102"}
