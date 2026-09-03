"""Integration tests for selection files crossing the filesystem boundary."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest


def _load_fleet_selection() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_selection")


@pytest.mark.integration
def test_load_selection_file_matches_csv_and_json_plans(tmp_path: Path) -> None:
    """AC1: real CSV and JSON files preserve raw text and load equivalent plans."""
    fleet_selection = _load_fleet_selection()
    csv_text = (
        "selection_id,icao24,callsign,firstseen,lastseen,msn,split,cohorts,utc_days\n"
        "sel-alpha,a00001,ALPHA1,1577835000,1577838600,M1,train,C1,20200101\n"
        "sel-zulu,z00026,ZULU9,1577923200,1577926800,M26,test,C9,20200102\n"
    )
    json_text = """[
        {
            "selection_id": "sel-alpha",
            "icao24": "a00001",
            "callsign": "ALPHA1",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
            "msn": "M1",
            "split": "train",
            "cohort": "C1",
            "utc_days": ["20200101"]
        },
        {
            "selection_id": "sel-zulu",
            "icao24": "z00026",
            "callsign": "ZULU9",
            "firstseen": 1577923200,
            "lastseen": 1577926800,
            "msn": "M26",
            "split": "test",
            "cohort": "C9",
            "utc_days": ["20200102"]
        }
    ]"""
    csv_path = tmp_path / "selection.csv"
    json_path = tmp_path / "selection.json"
    csv_path.write_text(csv_text)
    json_path.write_text(json_text)

    csv_source = fleet_selection.load_selection_file(csv_path)
    json_source = fleet_selection.load_selection_file(json_path)

    assert csv_source.raw == csv_text
    assert json_source.raw == json_text
    assert csv_source.plan == json_source.plan
    assert csv_source.plan == fleet_selection.compile_selection_csv(csv_text)


@pytest.mark.integration
def test_load_selection_file_preserves_opaque_legacy_artifact(tmp_path: Path) -> None:
    """AC4: an unknown extension remains byte-for-byte opaque with no compiled plan."""
    fleet_selection = _load_fleet_selection()
    raw = "legacy digest input\nnot,a,selection\n"
    path = tmp_path / "selection.txt"
    path.write_text(raw)

    source = fleet_selection.load_selection_file(path)

    assert source.plan is None
    assert source.raw == raw
    assert source.raw == path.read_text()
