"""Integration contract for the per-day local ARCO-ERA5 test stores."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr

from _daily_corpus import build_raw_opensky_day

pytestmark = pytest.mark.integration

type GridWriter = Callable[[Path], dict[str, Path]]

_DAYS = ("20200101", "20200102")
_ERA5_VARIABLES = {
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
}


def _writer() -> GridWriter:
    module = import_module("_local_era5_grid")
    return cast(GridWriter, module.write_local_era5_grid)


def _sentinel(dataset: xr.Dataset, variable: str) -> float:
    return float(dataset[variable].values.reshape(-1)[0])


def test_one_store_is_written_per_utc_source_day(tmp_path: Path) -> None:
    """AC1: write and return exactly one on-disk store per UTC source day."""
    stores = _writer()(tmp_path)

    assert stores == {day: tmp_path / f"{day}.zarr" for day in _DAYS}
    assert all(path.exists() for path in stores.values())
    assert all(path.parent == tmp_path for path in stores.values())


def test_each_store_reopens_with_era5_variables_and_daily_sentinels(tmp_path: Path) -> None:
    """AC2: expose ARCO-ERA5 fields with distinct temperature and u-wind by day."""
    stores = _writer()(tmp_path)
    sentinels: dict[str, tuple[float, float]] = {}

    for day in _DAYS:
        with xr.open_zarr(str(stores[day]), chunks=None) as dataset:
            assert _ERA5_VARIABLES <= set(dataset.data_vars)
            sentinels[day] = (
                _sentinel(dataset, "temperature"),
                _sentinel(dataset, "u_component_of_wind"),
            )

    assert sentinels["20200101"][0] != sentinels["20200102"][0]
    assert sentinels["20200101"][1] != sentinels["20200102"][1]


def test_each_grid_covers_its_source_day_corpus_footprint(tmp_path: Path) -> None:
    """AC3: keep times in-day and bracket that day's corpus latitude/longitude."""
    stores = _writer()(tmp_path)

    for day in _DAYS:
        frame = build_raw_opensky_day(day)
        raw_latitudes: list[float] = frame.get_column("raw_lat_deg").to_list()
        raw_longitudes: list[float] = frame.get_column("raw_lon_deg").to_list()
        iso_day = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        day_start = np.datetime64(iso_day, "ns")
        day_end = day_start + np.timedelta64(1, "D")

        with xr.open_zarr(str(stores[day]), chunks=None) as dataset:
            times = np.asarray(dataset.coords["time"].values, dtype="datetime64[ns]")
            latitudes = np.asarray(dataset.coords["latitude"].values, dtype=float)
            longitudes = np.asarray(dataset.coords["longitude"].values, dtype=float)

            assert bool(np.all(times >= day_start))
            assert bool(np.all(times < day_end))
            assert float(latitudes.min()) <= min(raw_latitudes)
            assert float(latitudes.max()) >= max(raw_latitudes)
            assert float(longitudes.min()) <= min(raw_longitudes)
            assert float(longitudes.max()) >= max(raw_longitudes)
