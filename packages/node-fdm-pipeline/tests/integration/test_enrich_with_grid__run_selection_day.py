from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import polars as pl
import pytest
import xarray as xr

from _daily_corpus import build_raw_opensky_day
from _local_era5_grid import write_local_era5_grid
from node_fdm_pipeline.commands import _day_runner
from node_fdm_pipeline.commands._day_plan import DayScopeViolation
from node_fdm_pipeline.commands._day_weather import GridCacheStats
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan
from node_fdm_pipeline.commands._science_profile import resolve_science_profile

pytestmark = pytest.mark.integration

DAY = "20200101"
PROFILE_ID = "opensky26-exp03-v1"
VERSIONS = {"node-fdm-pipeline": "test-code-v1"}
ERA5_FIELDS = (
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
)
GRID_SENTINELS = {
    "era_temp_K": 251.25,
    "era_u_wind_ms": 17.5,
    "era_v_wind_ms": -8.5,
}
_GRID_INPUTS = ("raw_timestamp", "raw_lat_deg", "raw_lon_deg", "raw_alt_ft")


@dataclass
class _Reader:
    root: Path
    source_day_override: str | None = None

    def __call__(self, source_day: str, selection_ids: frozenset[str]) -> pl.DataFrame:
        frame = pl.read_parquet(self.root / f"{source_day}.parquet").filter(
            pl.col("selection_id").is_in(selection_ids)
        )
        if self.source_day_override is not None:
            rogue = frame.head(1).with_columns(
                pl.lit(self.source_day_override).alias("meta_source_day")
            )
            frame = pl.concat((frame, rogue))
        return frame


@dataclass
class _GridOpener:
    stores: dict[str, Path]
    opened_days: list[str] = field(default_factory=list)

    def __call__(self, source_day: str, _field_name: str) -> object:
        self.opened_days.append(source_day)
        return cast("xr.Dataset", xr.open_zarr(str(self.stores[source_day]), chunks=None))


@dataclass
class _Lease:
    def acquire(self) -> None:
        pass

    def heartbeat(self) -> None:
        pass

    def assert_owned(self) -> None:
        pass

    def release(self) -> None:
        pass


@dataclass
class _Scenario:
    selection: SelectionPlan
    reader: _Reader
    opener: _GridOpener
    root: Path
    artifact: Path
    counters: Path


def _selection() -> SelectionPlan:
    return SelectionPlan(
        flights=(
            SelectedFlight(
                icao24="39a20a",
                callsign="A20N1",
                firstseen=1_577_872_800,
                lastseen=1_577_873_100,
                msn="A20N-001",
                split="train",
                cohorts=frozenset({"A20N"}),
                selection_id="sel-a20n",
                utc_days=(DAY,),
                acquisition_key="a20n-20200101",
            ),
            SelectedFlight(
                icao24="4bb738",
                callsign="B7381",
                firstseen=1_577_880_000,
                lastseen=1_577_880_300,
                msn="B738-001",
                split="train",
                cohorts=frozenset({"B738"}),
                selection_id="sel-b738",
                utc_days=(DAY,),
                acquisition_key="b738-20200101",
            ),
        ),
        digest="shared-grid-enrichment",
    )


def _write_distinct_grid(stores: dict[str, Path]) -> None:
    dataset = xr.open_zarr(str(stores[DAY]), chunks=None).load()
    dataset.close()
    source_values = {
        "temperature": GRID_SENTINELS["era_temp_K"],
        "u_component_of_wind": GRID_SENTINELS["era_u_wind_ms"],
        "v_component_of_wind": GRID_SENTINELS["era_v_wind_ms"],
    }
    for field_name, value in source_values.items():
        dataset[field_name] = xr.full_like(dataset[field_name], value)
    dataset.to_zarr(str(stores[DAY]), mode="w", consolidated=True)
    dataset.close()


@pytest.fixture
def scenario(tmp_path: Path) -> _Scenario:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    build_raw_opensky_day(DAY).with_columns(pl.lit("train").alias("meta_split")).write_parquet(
        raw_root / f"{DAY}.parquet"
    )
    stores = write_local_era5_grid(tmp_path / "era5")
    _write_distinct_grid(stores)
    artifact = tmp_path / "cleanup-token"
    artifact.write_text("input", encoding="utf-8")
    counters = tmp_path / "counters.json"
    counters.write_text(json.dumps({"cleanup-token": 0}) + "\n", encoding="utf-8")
    return _Scenario(
        selection=_selection(),
        reader=_Reader(raw_root),
        opener=_GridOpener(stores),
        root=tmp_path,
        artifact=artifact,
        counters=counters,
    )


def _close_grid(handle: object) -> None:
    assert isinstance(handle, xr.Dataset)
    handle.close()


def _run_day(scenario: _Scenario, *, reader: _Reader | None = None) -> _day_runner.DayRunReport:
    return _day_runner.run_selection_day(
        DAY,
        selection=scenario.selection,
        opensky_reader=reader or scenario.reader,
        grid_opener=scenario.opener,
        grid_closer=_close_grid,
        era5_fields=ERA5_FIELDS,
        lease=_Lease(),
        staging_root=scenario.root / "staging",
        published_root=scenario.root / "published",
        journal_path=scenario.root / "journal.jsonl",
        counters_path=scenario.counters,
        cleanup_artifacts={"cleanup-token": scenario.artifact},
        cleanup_decrements=(),
        science_profile=resolve_science_profile(PROFILE_ID),
        versions=VERSIONS,
        local_workers=2,
        max_resident_gib=8.0,
    )


def _partitions(root: Path) -> dict[str, pl.DataFrame]:
    frames = (pl.read_parquet(path) for path in sorted(root.rglob("*.parquet")))
    return {str(frame["cohort"][0]): frame for frame in frames}


def _assert_grid_sentinels(frame: pl.DataFrame) -> None:
    eligible = frame.filter(
        pl.all_horizontal([pl.col(column).is_not_null() for column in _GRID_INPUTS])
    )
    assert eligible.height > 0
    for column, expected in GRID_SENTINELS.items():
        assert eligible[column].null_count() == 0
        assert eligible[column].to_list() == pytest.approx([expected] * eligible.height)


def test_a20n_partition_uses_local_grid_sentinels(scenario: _Scenario) -> None:
    """AC1: A20N valid rows carry the values read from the 20200101 local grid."""
    _run_day(scenario)

    partitions = _partitions(scenario.root / "published")
    _assert_grid_sentinels(partitions["A20N"])


def test_b738_partition_reuses_single_source_day_grid(scenario: _Scenario) -> None:
    """AC2: B738 consumes the A20N shared handle and 20200101 is opened once."""
    report = _run_day(scenario)

    partitions = _partitions(scenario.root / "published")
    _assert_grid_sentinels(partitions["B738"])
    for column in GRID_SENTINELS:
        assert partitions["B738"][column][0] == pytest.approx(partitions["A20N"][column][0])
    stats = GridCacheStats(
        opens=report.grid_opens,
        total_opens=sum(report.grid_opens.values()),
    )
    assert (
        sum(count for (source_day, _field), count in stats.opens.items() if source_day == DAY) == 1
    )
    assert scenario.opener.opened_days == [DAY]


def test_out_of_plan_source_day_is_rejected_before_grid_open(
    scenario: _Scenario,
) -> None:
    """AC3: a 20200103 slice is rejected without opening a grid outside the DayPlan."""
    rogue_reader = _Reader(scenario.reader.root, source_day_override="20200103")

    with pytest.raises(DayScopeViolation):
        _run_day(scenario, reader=rogue_reader)

    assert scenario.opener.opened_days == [DAY]
