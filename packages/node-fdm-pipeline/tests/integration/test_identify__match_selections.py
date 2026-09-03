"""Integration coverage for selection-bounded identify outputs."""

from __future__ import annotations

import csv
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from _config_fixtures import write_config
from node_fdm_pipeline.commands import _raw_cache as raw_cache
from node_fdm_pipeline.commands import data
from node_fdm_pipeline.commands._fleet_selection import (
    SelectionFormatError,
    SelectionPlan,
    compile_selection,
)
from node_fdm_pipeline.commands._selection_match import MatchResult, match_selections
from node_fdm_pipeline.config import PipelineConfig

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
class _IdentifiedCampaign:
    data_dir: Path
    config_path: Path
    result: object
    selection: SelectionPlan


def _selection_rows() -> list[dict[str, object]]:
    alpha: dict[str, object] = {
        "icao24": "a00001",
        "callsign": "ALPHA1",
        "firstseen": 1577835000,
        "lastseen": 1577838600,
        "msn": "M1",
        "split": "train",
        "selection_id": "sel-alpha",
        "utc_days": ["20200101"],
        "acquisition_key": "acq-alpha",
    }
    zulu: dict[str, object] = {
        "icao24": "z00002",
        "callsign": "ZULU2",
        "firstseen": 1577835000,
        "lastseen": 1577838600,
        "msn": "M2",
        "split": "test",
        "selection_id": "sel-zulu",
        "utc_days": ["20200101"],
        "acquisition_key": "acq-zulu",
    }
    return [
        {**alpha, "cohort": "C1"},
        {**alpha, "cohort": "C2"},
        {**zulu, "cohort": "C1"},
    ]


def _write_selection_formats(tmp_path: Path) -> tuple[Path, Path]:
    rows = _selection_rows()
    json_path = tmp_path / "selection.json"
    json_path.write_text(json.dumps(rows), encoding="utf-8")

    csv_path = tmp_path / "selection.csv"
    fieldnames = (
        "selection_id",
        "icao24",
        "callsign",
        "firstseen",
        "lastseen",
        "msn",
        "split",
        "cohorts",
        "utc_days",
    )
    csv_rows: list[dict[str, object]] = []
    for row in rows:
        raw_days = row["utc_days"]
        assert isinstance(raw_days, list)
        csv_rows.append(
            {
                "selection_id": row["selection_id"],
                "icao24": row["icao24"],
                "callsign": row["callsign"],
                "firstseen": row["firstseen"],
                "lastseen": row["lastseen"],
                "msn": row["msn"],
                "split": row["split"],
                "cohorts": row["cohort"],
                "utc_days": "|".join(str(day) for day in raw_days),
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return json_path, csv_path


def _write_campaign(tmp_path: Path) -> tuple[Path, Path, Path]:
    fleet_dir = tmp_path / "fleet"
    data_dir = tmp_path / "identified"
    data_dir.mkdir()
    for cohort in ("C1", "C2"):
        cohort_dir = fleet_dir / "types" / cohort
        cohort_dir.mkdir(parents=True)
        write_config(cohort_dir / "config.yaml", data_dir)

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(_selection_rows()), encoding="utf-8")
    return fleet_dir, data_dir, selection_path


def _write_decoded_source(data_dir: Path) -> None:
    from node_fdm_data.delta import write_columns

    rows = [
        {
            "icao24": "a00001",
            "callsign": "  alpha1  ",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
            "raw_icao24": "a00001",
            "raw_callsign": "  alpha1  ",
            "raw_timestamp": datetime.fromtimestamp(1577835000, tz=UTC),
            "meta_batch_date": "20200101",
        },
        {
            "icao24": "a00001",
            "callsign": "  alpha1  ",
            "firstseen": 1577900000,
            "lastseen": 1577903600,
            "raw_icao24": "a00001",
            "raw_callsign": "  alpha1  ",
            "raw_timestamp": datetime.fromtimestamp(1577900000, tz=UTC),
            "meta_batch_date": "20200101",
        },
    ]
    write_columns(pl.DataFrame(rows), data_dir / "flights.delta")


def _run_identify(config: Path, selection: Path) -> object:
    identify_command: Callable[..., object] = data.identify
    if "selection" in inspect.signature(identify_command).parameters:
        return identify_command(
            config=config,
            selection=selection,
            gap_threshold_s=30,
        )
    return identify_command(config=config, gap_threshold_s=30)


@pytest.fixture
def identified_campaign(tmp_path: Path) -> _IdentifiedCampaign:
    fleet_dir, data_dir, selection_path = _write_campaign(tmp_path)
    config_path = fleet_dir / "types" / "C1" / "config.yaml"
    _write_decoded_source(data_dir)
    selection = compile_selection(_selection_rows())

    result = _run_identify(config_path, selection_path)

    return _IdentifiedCampaign(
        data_dir=data_dir,
        config_path=config_path,
        result=result,
        selection=selection,
    )


def test_identify_writes_alpha_once_with_selection_identity(
    identified_campaign: _IdentifiedCampaign,
) -> None:
    """AC1: identify writes alpha once with every named, normalized selection value."""
    identified = pl.read_delta(str(identified_campaign.data_dir / "flights.delta"))

    assert "selection_id" in identified.columns
    alpha = (
        identified.filter(pl.col("selection_id") == "sel-alpha")
        .select(
            "selection_id",
            "icao24",
            "callsign",
            "msn",
            "split",
            "firstseen",
            "lastseen",
        )
        .unique()
        .to_dicts()
    )
    selected = identified_campaign.selection.flights[0]
    assert alpha == [
        {
            "selection_id": selected.selection_id,
            "icao24": "a00001",
            "callsign": "ALPHA1",
            "msn": "M1",
            "split": "train",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
        }
    ]


def test_identify_excludes_rotation_outside_selected_interval(
    identified_campaign: _IdentifiedCampaign,
) -> None:
    """AC2: identify excludes the same aircraft's wholly out-of-window rotation."""
    identified = pl.read_delta(str(identified_campaign.data_dir / "flights.delta"))
    alpha = identified.filter(pl.col("icao24") == "a00001")

    assert alpha.height == 1
    assert 1577900000 not in alpha["firstseen"].to_list()
    assert 1577903600 not in alpha["lastseen"].to_list()


def test_identify_persists_absent_selection_receipt(
    identified_campaign: _IdentifiedCampaign,
) -> None:
    """AC3: identify leaves a durable sel-zulu receipt carrying an absence reason."""
    assert isinstance(identified_campaign.result, MatchResult)
    rejection = identified_campaign.result.rejections["sel-zulu"]
    assert rejection.selection_id == "sel-zulu"
    assert rejection.kind == "absent"

    cfg = PipelineConfig.from_yaml(identified_campaign.config_path)
    receipt = raw_cache.read_absence_receipt(
        raw_cache.cache_root(cfg, "history"),
        "20200101",
        "history",
    )

    assert receipt is not None
    assert receipt.row_count == 0
    assert "sel-zulu" in receipt.icao24s


def test_identify_loads_equivalent_csv_and_json_selection_plans(tmp_path: Path) -> None:
    """AC1: identify loads equivalent CSV and JSON rows into the same plan."""
    json_path, csv_path = _write_selection_formats(tmp_path)

    json_plan = data._load_identify_selection(tmp_path / "config.yaml", json_path)
    csv_plan = data._load_identify_selection(tmp_path / "config.yaml", csv_path)

    assert json_plan is not None
    assert csv_plan is not None
    assert csv_plan == json_plan
    assert [
        (flight.selection_id, flight.icao24, flight.utc_days, flight.acquisition_key)
        for flight in csv_plan.flights
    ] == [
        (flight.selection_id, flight.icao24, flight.utc_days, flight.acquisition_key)
        for flight in json_plan.flights
    ]


def test_match_selections_is_format_independent(tmp_path: Path) -> None:
    """AC2: alpha is admitted and zulu rejected identically for CSV and JSON."""
    json_path, csv_path = _write_selection_formats(tmp_path)
    rotations = [
        {
            "selection_id": "sel-alpha",
            "icao24": "a00001",
            "callsign": " alpha1 ",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
        }
    ]

    json_plan = data._load_identify_selection(tmp_path / "config.yaml", json_path)
    csv_plan = data._load_identify_selection(tmp_path / "config.yaml", csv_path)
    assert json_plan is not None
    assert csv_plan is not None

    json_result = match_selections(json_plan, rotations)
    csv_result = match_selections(csv_plan, rotations)

    assert tuple(item["selection_id"] for item in json_result.admitted) == ("sel-alpha",)
    assert tuple(item["selection_id"] for item in csv_result.admitted) == ("sel-alpha",)
    assert csv_result.rejections == json_result.rejections
    assert set(csv_result.rejections) == {"sel-zulu"}
    assert csv_result.rejections["sel-zulu"].kind == "absent"


def test_identify_rejects_csv_without_selection_id(tmp_path: Path) -> None:
    """AC3: malformed CSV reports the omitted selection_id at load time."""
    csv_path = tmp_path / "selection.csv"
    csv_path.write_text(
        "icao24,callsign,firstseen,lastseen,msn,split,cohorts,utc_days\n"
        "a00001,ALPHA1,1577835000,1577838600,M1,train,C1,20200101\n",
        encoding="utf-8",
    )

    with pytest.raises(SelectionFormatError) as excinfo:
        data._load_identify_selection(tmp_path / "config.yaml", csv_path)

    assert "selection_id" in str(excinfo.value)
