from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
import pytest
import xarray as xr

from _daily_corpus import build_raw_opensky_day
from _local_era5_grid import write_local_era5_grid
from node_fdm_pipeline.commands import _day_assemble, _day_runner
from node_fdm_pipeline.commands._day_plan import build_day_plan
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan
from node_fdm_pipeline.commands._science_profile import resolve_science_profile

pytestmark = pytest.mark.integration

DAYS = ("20200101", "20200102")
ERA5_FIELDS = (
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
)
ERA5_SENTINELS = {
    "20200101": (273.15, 5.0, -2.0),
    "20200102": (283.15, 15.0, 3.0),
}
PROFILE_ID = "opensky26-exp03-v1"
VERSIONS = {"node-fdm-pipeline": "test-code-v1"}
_CAMPAIGN_RUNS = 0


@dataclass(frozen=True)
class _Reader:
    root: Path

    def __call__(self, source_day: str, selection_ids: frozenset[str]) -> pl.DataFrame:
        return pl.read_parquet(self.root / f"{source_day}.parquet").filter(
            pl.col("selection_id").is_in(selection_ids)
        )


@dataclass(frozen=True)
class _GridOpener:
    stores: dict[str, Path]

    def __call__(self, source_day: str, _field_name: str) -> object:
        return cast("xr.Dataset", xr.open_zarr(str(self.stores[source_day]), chunks=None))


class _Lease:
    def acquire(self) -> None:
        pass

    def heartbeat(self) -> None:
        pass

    def assert_owned(self) -> None:
        pass

    def release(self) -> None:
        pass


@dataclass(frozen=True)
class _Scenario:
    root: Path
    selection: SelectionPlan
    stores: dict[str, Path]


@dataclass(frozen=True)
class _CampaignResult:
    expected_midnight_rows: pl.DataFrame
    partitions: dict[tuple[str, str], pl.DataFrame]


def _flight(  # noqa: PLR0913
    selection_id: str,
    cohort: str,
    icao24: str,
    firstseen: int,
    lastseen: int,
    utc_days: tuple[str, ...],
) -> SelectedFlight:
    return SelectedFlight(
        icao24=icao24,
        callsign=cohort,
        firstseen=firstseen,
        lastseen=lastseen,
        msn=f"{cohort}-001",
        split="train",
        cohorts=frozenset({cohort}),
        selection_id=selection_id,
        utc_days=utc_days,
        acquisition_key=selection_id,
    )


def _selection() -> SelectionPlan:
    return SelectionPlan(
        flights=(
            _flight("sel-a20n", "A20N", "39a20a", 1_577_872_800, 1_577_872_860, ("20200101",)),
            _flight(
                "sel-a20n-midnight",
                "A20N",
                "39a20f",
                1_577_923_080,
                1_577_923_320,
                DAYS,
            ),
            _flight("sel-b738", "B738", "4bb738", 1_577_880_000, 1_577_880_060, ("20200101",)),
            _flight("sel-e190", "E190", "3ce190", 1_577_952_000, 1_577_952_060, ("20200102",)),
            _flight("sel-crj9", "CRJ9", "4cc9a0", 1_577_955_600, 1_577_955_660, ("20200102",)),
            _flight("sel-cl35", "CL35", "43c135", 1_577_962_800, 1_577_962_860, ("20200102",)),
        ),
        digest="two-day-midnight-campaign",
    )


def _close_grid(handle: object) -> None:
    assert isinstance(handle, xr.Dataset)
    handle.close()


def _run_day(root: Path, selection: SelectionPlan, day: str, stores: dict[str, Path]) -> None:
    _day_runner.run_selection_day(
        day,
        selection=selection,
        opensky_reader=_Reader(root / "raw"),
        grid_opener=_GridOpener(stores),
        grid_closer=_close_grid,
        era5_fields=ERA5_FIELDS,
        lease=_Lease(),
        staging_root=root / "staging",
        published_root=root / "published",
        journal_path=root / "journal.jsonl",
        counters_path=root / "counters.json",
        cleanup_artifacts={},
        cleanup_decrements=(),
        science_profile=resolve_science_profile(PROFILE_ID),
        versions=VERSIONS,
        local_workers=2,
        max_resident_gib=8.0,
    )


def _run_campaign(scenario: _Scenario) -> None:
    global _CAMPAIGN_RUNS
    _CAMPAIGN_RUNS += 1
    assert _CAMPAIGN_RUNS == 1
    for day in DAYS:
        _run_day(scenario.root, scenario.selection, day, scenario.stores)


@pytest.fixture(scope="module")
def scenario(tmp_path_factory: pytest.TempPathFactory) -> _Scenario:
    tmp_path = tmp_path_factory.mktemp("selection-day-campaign")
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    for day in DAYS:
        build_raw_opensky_day(day).with_columns(pl.lit("train").alias("meta_split")).write_parquet(
            raw_root / f"{day}.parquet"
        )
    stores = write_local_era5_grid(tmp_path / "era5")
    (tmp_path / "counters.json").write_text(json.dumps({}) + "\n", encoding="utf-8")
    return _Scenario(root=tmp_path, selection=_selection(), stores=stores)


def _published_partitions(root: Path) -> dict[tuple[str, str], pl.DataFrame]:
    partitions: dict[tuple[str, str], pl.DataFrame] = {}
    for path in sorted(root.rglob("*.parquet")):
        frame = pl.read_parquet(path)
        key = (str(frame["cohort"][0]), str(frame["meta_selection_day"][0]))
        assert key not in partitions
        partitions[key] = frame
    return partitions


def _expected_midnight_rows(scenario: _Scenario) -> pl.DataFrame:
    plan = build_day_plan(scenario.selection, "20200101")
    profile = resolve_science_profile(PROFILE_ID)
    handles = {
        day: cast(
            "xr.Dataset",
            xr.open_zarr(str(scenario.stores[day]), chunks=None),
        )
        for day in DAYS
    }
    try:
        assembled_days = []
        for day in DAYS:
            frame = pl.read_parquet(scenario.root / "raw" / f"{day}.parquet").filter(
                pl.col("selection_id") == "sel-a20n-midnight"
            )
            assembled_days.append(
                _day_assemble.assemble_partition(
                    frame,
                    plan,
                    ("A20N", "20200101"),
                    profile,
                    VERSIONS,
                    grid_handles=handles,
                )
            )
        return (
            pl.concat(assembled_days, how="vertical_relaxed")
            .unique(subset=["selection_id", "raw_timestamp"], keep="first")
            .sort("raw_timestamp")
        )
    finally:
        for handle in handles.values():
            handle.close()


@pytest.fixture(scope="module")
def campaign(scenario: _Scenario) -> _CampaignResult:
    expected_midnight_rows = _expected_midnight_rows(scenario)
    _run_campaign(scenario)
    return _CampaignResult(
        expected_midnight_rows=expected_midnight_rows,
        partitions=_published_partitions(scenario.root / "published"),
    )


def _all_midnight_rows(partitions: dict[tuple[str, str], pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(
        frame.filter(pl.col("selection_id") == "sel-a20n-midnight")
        for frame in partitions.values()
    ).sort("raw_timestamp")


def test_published_midnight_selection_is_one_ordered_source_day_union(
    campaign: _CampaignResult,
) -> None:
    """AC1: the 20200101 A20N partition holds one ordered, duplicate-free day union."""
    expected = campaign.expected_midnight_rows
    partitions = campaign.partitions
    midnight = partitions[("A20N", "20200101")].filter(
        pl.col("selection_id") == "sel-a20n-midnight"
    )
    actual = list(
        midnight.sort("raw_timestamp").select("meta_source_day", "raw_timestamp").iter_rows()
    )
    expected_pairs = list(expected.select("meta_source_day", "raw_timestamp").iter_rows())

    assert actual == expected_pairs
    assert midnight["raw_timestamp"].n_unique() == midnight.height
    assert _all_midnight_rows(partitions).height == expected.height


def test_midnight_rows_use_their_own_source_day_era5_values(
    campaign: _CampaignResult,
) -> None:
    """AC2: each midnight row carries the ERA5 sentinels of its own source day."""
    expected = campaign.expected_midnight_rows
    partitions = campaign.partitions
    midnight = _all_midnight_rows(partitions)

    assert midnight.height == expected.height
    for source_day, sentinels in ERA5_SENTINELS.items():
        rows = midnight.filter(pl.col("meta_source_day") == source_day)
        expected_count = expected.filter(pl.col("meta_source_day") == source_day).height
        assert rows.height == expected_count
        for column, sentinel in zip(
            ("era_temp_K", "era_u_wind_ms", "era_v_wind_ms"),
            sentinels,
            strict=True,
        ):
            assert rows[column].to_list() == pytest.approx([sentinel] * rows.height)


def test_two_daily_calls_publish_only_selection_day_anchored_partitions(
    campaign: _CampaignResult,
) -> None:
    """AC3: independent daily calls emit exactly the durable campaign identities."""
    partitions = campaign.partitions
    selection_ids = {
        str(selection_id)
        for frame in partitions.values()
        for selection_id in frame["selection_id"].unique()
    }
    midnight_keys = {
        key
        for key, frame in partitions.items()
        if "sel-a20n-midnight" in frame["selection_id"].to_list()
    }

    assert selection_ids == {
        "sel-a20n",
        "sel-a20n-midnight",
        "sel-b738",
        "sel-e190",
        "sel-crj9",
        "sel-cl35",
    }
    assert set(partitions) == {
        ("A20N", "20200101"),
        ("B738", "20200101"),
        ("E190", "20200102"),
        ("CRJ9", "20200102"),
        ("CL35", "20200102"),
    }
    assert ("A20N", "20200102") not in partitions
    assert midnight_keys == {("A20N", "20200101")}
