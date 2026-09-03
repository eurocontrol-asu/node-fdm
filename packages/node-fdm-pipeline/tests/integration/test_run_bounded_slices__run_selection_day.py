from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import polars as pl
import pytest
import xarray as xr
from pytest_mock import MockerFixture

from _daily_corpus import build_raw_opensky_day
from _local_era5_grid import write_local_era5_grid
from node_fdm_pipeline.commands import _day_bounds, _day_runner
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
DERIVED_COLUMNS = {
    "fdm_flag_valid",
    "fdm_tas_from_cas_kt",
    "fdm_gamma_rad",
    "fdm_distance_cum_m",
}


@dataclass
class _Reader:
    root: Path

    def __call__(self, source_day: str, selection_ids: frozenset[str]) -> pl.DataFrame:
        return pl.read_parquet(self.root / f"{source_day}.parquet").filter(
            pl.col("selection_id").is_in(selection_ids)
        )


@dataclass
class _GridOpener:
    stores: dict[str, Path]

    def __call__(self, source_day: str, field_name: str) -> object:
        return cast("xr.Dataset", xr.open_zarr(str(self.stores[source_day]), chunks=None))


def _close_grid(handle: object) -> None:
    assert isinstance(handle, xr.Dataset)
    handle.close()


class _BoundedSlice(Protocol):
    @property
    def slice_id(self) -> str: ...

    @property
    def slice_gib(self) -> float: ...


@dataclass
class _Lease:
    calls: list[str] = field(default_factory=list)

    def acquire(self) -> None:
        self.calls.append("acquire")

    def heartbeat(self) -> None:
        self.calls.append("heartbeat")

    def assert_owned(self) -> None:
        self.calls.append("assert_owned")

    def release(self) -> None:
        self.calls.append("release")


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
        digest="two-cohort-science-chain-test",
    )


def test_run_selection_day_executes_science_in_bounded_slices(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC3: two cohort partitions complete scientifically under the two-worker bound."""
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    build_raw_opensky_day(DAY).with_columns(pl.lit("train").alias("meta_split")).write_parquet(
        raw_root / f"{DAY}.parquet"
    )
    stores = write_local_era5_grid(tmp_path / "era5")
    artifact = tmp_path / "cleanup-token"
    artifact.write_text("input", encoding="utf-8")
    counters = tmp_path / "counters.json"
    counters.write_text(json.dumps({"cleanup-token": 0}) + "\n", encoding="utf-8")
    captured: list[_day_bounds.RunReport] = []
    real_run_bounded_slices = _day_bounds.run_bounded_slices

    def record_run_report[SliceT: _BoundedSlice, ResultT](
        slices: Iterable[SliceT],
        budget: _day_bounds.ResourceBudget,
        fn: Callable[[SliceT], ResultT],
    ) -> _day_bounds.RunReport:
        report = real_run_bounded_slices(slices, budget, fn)
        captured.append(report)
        return report

    mocker.patch.object(_day_runner, "run_bounded_slices", side_effect=record_run_report)
    report = _day_runner.run_selection_day(
        DAY,
        selection=_selection(),
        opensky_reader=_Reader(raw_root),
        grid_opener=_GridOpener(stores),
        grid_closer=_close_grid,
        era5_fields=ERA5_FIELDS,
        lease=_Lease(),
        staging_root=tmp_path / "staging",
        published_root=tmp_path / "published",
        journal_path=tmp_path / "journal.jsonl",
        counters_path=counters,
        cleanup_artifacts={"cleanup-token": artifact},
        cleanup_decrements=(),
        science_profile=resolve_science_profile(PROFILE_ID),
        versions=VERSIONS,
        local_workers=2,
        max_resident_gib=8.0,
    )

    assert captured
    (bounded_report,) = captured
    partitions = tuple(
        pl.read_parquet(path) for path in sorted((tmp_path / "published").rglob("*.parquet"))
    )
    assert len(bounded_report.results) == len(partitions) == 2
    assert all(DERIVED_COLUMNS <= set(partition.columns) for partition in partitions)
    assert bounded_report.max_observed_concurrency <= 2
    assert len(report.worker_pids) == len(partitions)
