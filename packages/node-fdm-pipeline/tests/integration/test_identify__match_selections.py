"""Integration coverage for selection-bounded identify outputs."""

from __future__ import annotations

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
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan, compile_selection
from node_fdm_pipeline.commands._selection_match import MatchResult
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
