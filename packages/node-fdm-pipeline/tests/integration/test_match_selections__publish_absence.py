from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _raw_cache as raw_cache
from node_fdm_pipeline.commands import data
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan, compile_selection
from node_fdm_pipeline.commands._selection_match import MatchResult, match_selections
from node_fdm_pipeline.config import PathsConfig, PipelineConfig


@dataclass(frozen=True)
class _CampaignRun:
    identified_path: Path
    result: object
    absence_root: Path
    selection: SelectionPlan
    expected_match: MatchResult


@pytest.fixture
def campaign_run(tmp_path: Path, mocker: MockerFixture) -> _CampaignRun:
    campaign_dir = tmp_path / "campaign"
    type_dir = campaign_dir / "types" / "A20N"
    results_dir = type_dir / "results"
    data_dir = campaign_dir / "identified"
    results_dir.mkdir(parents=True)
    data_dir.mkdir()
    config_path = type_dir / "config.yaml"
    config_path.touch()

    selection_rows: list[dict[str, object]] = [
        {
            "icao24": "a00001",
            "callsign": "ALPHA1",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
            "msn": "M1",
            "split": "train",
            "cohort": "C1",
            "selection_id": "sel-alpha",
        },
        {
            "icao24": "a00001",
            "callsign": "ALPHA1",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
            "msn": "M1",
            "split": "train",
            "cohort": "C2",
            "selection_id": "sel-alpha",
        },
        {
            "icao24": "z00002",
            "callsign": "ZULU2",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
            "msn": "M2",
            "split": "test",
            "cohort": "C3",
            "selection_id": "sel-zulu",
        },
    ]
    pl.DataFrame(selection_rows).write_csv(results_dir / "selection_A20N.csv")
    selection = compile_selection(selection_rows)

    rotations: tuple[dict[str, object], ...] = (
        {
            "icao24": "a00001",
            "callsign": "ALPHA1",
            "firstseen": 1577835000,
            "lastseen": 1577838600,
        },
        {
            "icao24": "a00001",
            "callsign": "ALPHA1",
            "firstseen": 1577900000,
            "lastseen": 1577903600,
        },
    )
    expected_match = match_selections(selection, rotations)
    staged_path = campaign_dir / "staged.parquet"
    pl.DataFrame(
        [
            {
                **rotations[0],
                "raw_icao24": "a00001",
                "raw_callsign": "ALPHA1",
                "raw_timestamp": datetime(2019, 12, 31, 23, 30, tzinfo=UTC),
                "meta_batch_date": "20200101",
            },
            {
                **rotations[1],
                "raw_icao24": "a00001",
                "raw_callsign": "ALPHA1",
                "raw_timestamp": datetime(2020, 1, 1, 17, 33, 20, tzinfo=UTC),
                "meta_batch_date": "20200101",
            },
        ]
    ).write_parquet(staged_path)

    cfg = cast("PipelineConfig", SimpleNamespace(paths=PathsConfig(data_dir=data_dir)))
    mocker.patch.object(PipelineConfig, "from_yaml", return_value=cfg)
    mocker.patch(
        "node_fdm_data.delta.read_delta_table",
        side_effect=lambda _path: pl.read_parquet(staged_path),
    )
    identified_path = data_dir / "campaign.parquet"

    def write_identified(frame: pl.DataFrame, _path: Path) -> None:
        frame.write_parquet(identified_path)

    mocker.patch("node_fdm_data.delta.write_columns", side_effect=write_identified)

    result = data.identify(config=config_path, gap_threshold_s=30)
    return _CampaignRun(
        identified_path=identified_path,
        result=result,
        absence_root=raw_cache.cache_root(cfg, "history"),
        selection=selection,
        expected_match=expected_match,
    )


@pytest.mark.integration
def test_alpha_is_identified_once_with_its_full_identity(campaign_run: _CampaignRun) -> None:
    """AC1: the campaign keeps alpha once and projects its complete selection identity."""
    alpha_rows = (
        pl.read_parquet(campaign_run.identified_path)
        .filter(pl.col("icao24") == "a00001")
        .to_dicts()
    )
    selected_alpha = next(
        flight for flight in campaign_run.selection.flights if flight.selection_id == "sel-alpha"
    )

    assert len(alpha_rows) == 1
    assert alpha_rows[0]["selection_id"] == selected_alpha.selection_id
    assert alpha_rows[0]["callsign"] == selected_alpha.callsign
    assert alpha_rows[0]["firstseen"] == selected_alpha.firstseen
    assert alpha_rows[0]["lastseen"] == selected_alpha.lastseen
    assert alpha_rows[0]["msn"] == selected_alpha.msn
    assert alpha_rows[0]["split"] == selected_alpha.split
    assert set(alpha_rows[0]["cohorts"]) == {"C1", "C2"}
    assert tuple(alpha_rows[0]["utc_days"]) == ("20191231", "20200101")


@pytest.mark.integration
def test_out_of_interval_rotation_is_not_admitted(campaign_run: _CampaignRun) -> None:
    """AC2: the campaign excludes a second rotation outside alpha's interval."""
    identified = pl.read_parquet(campaign_run.identified_path)
    expected_firstseen = {
        cast("dict[str, object]", rotation)["firstseen"]
        for rotation in campaign_run.expected_match.admitted
    }

    assert set(identified["firstseen"].to_list()) == expected_firstseen
    assert identified.filter(pl.col("firstseen") == 1577900000).is_empty()


@pytest.mark.integration
def test_absent_selection_publishes_an_explicit_receipt(campaign_run: _CampaignRun) -> None:
    """AC3: a missing zulu rotation is rejected and durably receipted by selection id."""
    assert isinstance(campaign_run.result, MatchResult)
    assert campaign_run.result.rejections["sel-zulu"].kind == "absent"
    assert campaign_run.result.rejections == campaign_run.expected_match.rejections

    receipt = raw_cache.read_absence_receipt(
        campaign_run.absence_root,
        "20200101",
        "history",
    )
    assert receipt is not None
    assert receipt.row_count == 0
    assert "sel-zulu" in receipt.icao24s
