"""Integration tests for selection files crossing the filesystem boundary."""

from __future__ import annotations

import importlib
import re
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


_HISTORICAL_CSV = (
    "selection_id,icao24,callsign,start,end,cohort\n"
    "sel-alpha,a00001,ALPHA1,1577833200,1577840400,C1\n"
    "sel-alpha,a00001,ALPHA1,1577833200,1577840400,C2\n"
)

_CAMPAIGN_CSV = (
    "selection_id,icao24,callsign,start,end,cohorts,utc_days\n"
    "sel-alpha,a00001,ALPHA1,1577833200,1577840400,C1|C2,20191231|20200101\n"
)


@pytest.mark.integration
def test_historical_identification_csv_compiles_its_cohorts_and_derived_days(
    tmp_path: Path,
) -> None:
    """AC1: historical rows mutualise cohorts and derive every interval UTC day."""
    fleet_selection = _load_fleet_selection()
    path = tmp_path / "selection_hist.csv"
    path.write_text(_HISTORICAL_CSV)

    source = fleet_selection.load_selection_file(path)

    assert source.plan is not None
    assert len(source.plan.flights) == 1
    flight = source.plan.flights[0]
    assert flight.cohorts == frozenset({"C1", "C2"})
    assert flight.utc_days == ("20191231", "20200101")


@pytest.mark.integration
def test_both_schemas_of_the_same_flight_yield_one_stable_digest(tmp_path: Path) -> None:
    """AC2: equivalent historical and campaign encodings have stable equal digests."""
    fleet_selection = _load_fleet_selection()
    historical_path = tmp_path / "selection_hist.csv"
    campaign_path = tmp_path / "selection_campaign.csv"
    historical_path.write_text(_HISTORICAL_CSV)
    campaign_path.write_text(_CAMPAIGN_CSV)

    historical_first = fleet_selection.load_selection_file(historical_path).plan
    campaign_first = fleet_selection.load_selection_file(campaign_path).plan
    historical_second = fleet_selection.load_selection_file(historical_path).plan
    campaign_second = fleet_selection.load_selection_file(campaign_path).plan

    assert historical_first is not None
    assert campaign_first is not None
    assert historical_second is not None
    assert campaign_second is not None
    assert historical_first.digest == campaign_first.digest
    assert historical_second.digest == historical_first.digest
    assert campaign_second.digest == campaign_first.digest


@pytest.mark.integration
def test_historical_csv_raw_content_is_preserved_verbatim(tmp_path: Path) -> None:
    """AC3: loading historical CSV preserves its header and rows byte-for-byte."""
    fleet_selection = _load_fleet_selection()
    path = tmp_path / "selection_hist.csv"
    path.write_bytes(_HISTORICAL_CSV.encode())

    source = fleet_selection.load_selection_file(path)

    assert source.raw == _HISTORICAL_CSV
    assert source.raw.encode() == path.read_bytes()


@pytest.mark.integration
def test_header_mixing_cohort_and_cohorts_is_rejected(tmp_path: Path) -> None:
    """AC4: a header mixing singular and plural cohort columns is ambiguous."""
    fleet_selection = _load_fleet_selection()
    raw = (
        "selection_id,icao24,callsign,start,end,cohort,cohorts,utc_days\n"
        "sel-alpha,a00001,ALPHA1,1577833200,1577840400,C1,C1|C2,"
        "20191231|20200101\n"
    )
    path = tmp_path / "selection_ambiguous.csv"
    path.write_text(raw)

    with pytest.raises(fleet_selection.SelectionFormatError) as exc_info:
        fleet_selection.load_selection_file(path)

    message = str(exc_info.value).lower()
    assert "ambig" in message
    assert re.search(r"\bcohort\b", message)
    assert re.search(r"\bcohorts\b", message)


@pytest.mark.integration
def test_historical_csv_without_selection_id_is_rejected_by_name(tmp_path: Path) -> None:
    """AC5: historical CSV validation names its missing selection_id column."""
    fleet_selection = _load_fleet_selection()
    raw = "icao24,callsign,start,end,cohort\na00001,ALPHA1,1577833200,1577840400,C1\n"
    path = tmp_path / "selection_missing_id.csv"
    path.write_text(raw)

    with pytest.raises(fleet_selection.SelectionFormatError) as exc_info:
        fleet_selection.load_selection_file(path)

    message = str(exc_info.value).lower()
    assert "selection_id" in message
    assert "cohorts" not in message
    assert "utc_days" not in message


@pytest.mark.integration
def test_unterminated_quoted_field_is_reported_as_a_csv_syntax_error(
    tmp_path: Path,
) -> None:
    """AC6: malformed historical quoting is reported as a CSV parse error."""
    fleet_selection = _load_fleet_selection()
    raw = (
        "selection_id,icao24,callsign,start,end,cohort\n"
        'sel-alpha,a00001,"ALPHA1,1577833200,1577840400,C1\n'
    )
    path = tmp_path / "selection_bad_quote.csv"
    path.write_text(raw)

    with pytest.raises(fleet_selection.SelectionFormatError) as exc_info:
        fleet_selection.load_selection_file(path)

    message = str(exc_info.value).lower()
    assert "csv" in message
    assert any(term in message for term in ("quot", "syntax", "parse", "unexpected end"))
