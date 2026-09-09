from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from node_fdm_data.meteo import enrich_era5
from node_fdm_data.preprocessing.clean_speeds import clean_bds_speeds
from node_fdm_data.preprocessing.derive import derive_columns
from polars.testing import assert_frame_equal

from _config_fixtures import write_config
from _daily_corpus import build_raw_opensky_day
from _local_era5_grid import write_local_era5_grid
from node_fdm_pipeline.commands import data
from node_fdm_pipeline.commands._day_assemble import assemble_partition
from node_fdm_pipeline.commands._day_plan import build_day_plan
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan
from node_fdm_pipeline.commands._science_profile import resolve_science_profile
from node_fdm_pipeline.config import PipelineConfig

pytestmark = pytest.mark.integration

DAY = "20200101"
PROFILE_ID = "opensky26-exp03-v1"
VERSIONS = {"node-fdm-pipeline": "test-code-v1"}
SCIENTIFIC_COLUMNS = (
    "fdm_flag_valid",
    "fdm_tas_from_cas_kt",
    "fdm_gamma_rad",
    "fdm_distance_cum_m",
)


class _LocalGrid:
    def __init__(self, store: Path) -> None:
        self.store = store

    def interpolate(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        with xr.open_zarr(str(self.store), chunks=None) as dataset:
            for variable in (
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ):
                result[variable] = float(dataset[variable].values.reshape(-1)[0])
        return result


def _raw_b738_trajectory() -> pl.DataFrame:
    endpoints = build_raw_opensky_day(DAY).filter(pl.col("cohort") == "B738").sort("raw_timestamp")
    start = endpoints.row(0, named=True)
    end = endpoints.row(-1, named=True)
    samples = 301
    numeric = ("raw_lat_deg", "raw_lon_deg", "raw_alt_ft", "raw_gs_kt", "raw_track_deg")
    columns: dict[str, object] = {
        name: [start[name]] * samples
        for name in (
            "selection_id",
            "cohort",
            "meta_selection_day",
            "meta_source_day",
            "meta_aircraft_type",
            "raw_icao24",
        )
    }
    columns["raw_timestamp"] = [
        start["raw_timestamp"] + timedelta(seconds=offset) for offset in range(samples)
    ]
    for name in numeric:
        delta = float(end[name]) - float(start[name])
        columns[name] = np.linspace(float(start[name]), float(start[name]) + 5.0 * delta, samples)
    columns["raw_callsign"] = ["B7381"] * samples
    columns["meta_split"] = ["train"] * samples
    columns["raw_vz_ftmin"] = np.full(samples, -2_000.0)
    columns["bds_mach"] = np.full(samples, np.nan)
    columns["bds_ias_kt"] = np.full(samples, np.nan)
    columns["bds_tas_kt"] = np.full(samples, np.nan)
    columns["bds_mcp_alt_sel_ft"] = columns["raw_alt_ft"]
    return pl.DataFrame(columns)


def _selection(frame: pl.DataFrame) -> SelectionPlan:
    first = frame["raw_timestamp"].min()
    last = frame["raw_timestamp"].max()
    assert isinstance(first, datetime)
    assert isinstance(last, datetime)
    return SelectionPlan(
        flights=(
            SelectedFlight(
                icao24="4bb738",
                callsign="B7381",
                firstseen=int(first.timestamp()),
                lastseen=int(last.timestamp()),
                msn="B738-001",
                split="train",
                cohorts=frozenset({"B738"}),
                selection_id="sel-b738",
                utc_days=(DAY,),
                acquisition_key="b738-20200101",
            ),
        ),
        digest="b738-science-chain-test",
    )


def _persisted_inputs(tmp_path: Path) -> tuple[pl.DataFrame, dict[str, Path]]:
    corpus_path = tmp_path / "raw" / f"{DAY}.parquet"
    corpus_path.parent.mkdir(parents=True)
    _raw_b738_trajectory().write_parquet(corpus_path)
    stores = write_local_era5_grid(tmp_path / "era5")
    return pl.read_parquet(corpus_path), stores


def _assemble(frame: pl.DataFrame) -> pl.DataFrame:
    return assemble_partition(
        frame,
        build_day_plan(_selection(frame), DAY),
        ("B738", DAY),
        resolve_science_profile(PROFILE_ID),
        VERSIONS,
    )


def _historical_oracle(tmp_path: Path, frame: pl.DataFrame, store: Path) -> pl.DataFrame:
    data_dir = tmp_path / "historical"
    data_dir.mkdir()
    frame.write_delta(str(data_dir / "flights.delta"), mode="overwrite")
    config_path = write_config(
        tmp_path / "historical.yaml",
        data_dir,
        typecodes=["B738"],
        extra="""era5_features:
  - temperature
  - u_component_of_wind
  - v_component_of_wind
computing:
  default_cpu_count: 1
""",
    )
    data.identify(config=config_path)
    data.preprocess(config=config_path)
    data.flag(config=config_path)

    config = PipelineConfig.from_yaml(config_path)
    oracle = pl.read_delta(str(data_dir / "flights.delta"))
    oracle = enrich_era5(oracle, _LocalGrid(store))
    oracle = clean_bds_speeds(oracle, **config.clean_speeds.model_dump())
    oracle = derive_columns(oracle, lateral_cfg=config.lateral_detection.to_params())
    oracle = data._build_selected_params_with_valid_filter(
        oracle,
        None,
        profile=PROFILE_ID,
    )
    return oracle.sort("raw_timestamp")


def test_assemble_partition_returns_scientific_columns_for_raw_b738(
    tmp_path: Path,
) -> None:
    """AC1: raw B738 rows become non-null scientific rows after daily assembly."""
    frame, _stores = _persisted_inputs(tmp_path)

    result = _assemble(frame)

    assert set(SCIENTIFIC_COLUMNS) <= set(result.columns)
    valid = result.filter(pl.col("fdm_flag_valid"))
    assert not valid.is_empty()
    assert valid.select(SCIENTIFIC_COLUMNS).null_count().row(0) == (0,) * len(SCIENTIFIC_COLUMNS)


def test_daily_output_equals_historical_scientific_oracle_row_by_row(
    tmp_path: Path,
) -> None:
    """AC2: daily assembly has ordered row parity with the historical chain."""
    frame, stores = _persisted_inputs(tmp_path)
    historical = _historical_oracle(tmp_path, frame, stores[DAY])
    daily = _assemble(frame).sort("raw_timestamp")
    common_columns = [column for column in historical.columns if column in daily.columns]

    assert [column for column in daily.columns if column in historical.columns] == common_columns
    assert set(SCIENTIFIC_COLUMNS) <= set(common_columns)
    assert_frame_equal(
        daily.select(common_columns),
        historical.select(common_columns),
        check_row_order=True,
        check_column_order=True,
    )
