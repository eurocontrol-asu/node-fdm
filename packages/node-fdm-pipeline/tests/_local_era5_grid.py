"""Create compact per-day ARCO-ERA5 stores for offline integration tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr

from _daily_corpus import build_raw_opensky_day

__all__ = ["write_local_era5_grid"]

_DAYS = ("20200101", "20200102")
_GRID_PADDING_DEG = 0.25
_PRESSURE_LEVEL_HPA = 500
_SENTINELS: dict[str, tuple[float, float, float]] = {
    "20200101": (273.15, 5.0, -2.0),
    "20200102": (283.15, 15.0, 3.0),
}
_GRID_DIMS = ("time", "level", "latitude", "longitude")


def _coordinates(day: str) -> tuple[list[datetime], list[float], list[float]]:
    frame = build_raw_opensky_day(day)
    timestamps = cast(list[datetime], frame.get_column("raw_timestamp").to_list())
    latitudes = [float(value) for value in frame.get_column("raw_lat_deg").to_list()]
    longitudes = [float(value) for value in frame.get_column("raw_lon_deg").to_list()]

    times = [
        min(timestamps).replace(tzinfo=None),
        max(timestamps).replace(tzinfo=None),
    ]
    latitude_bounds = [
        max(latitudes) + _GRID_PADDING_DEG,
        min(latitudes) - _GRID_PADDING_DEG,
    ]
    longitude_bounds = [
        min(longitudes) - _GRID_PADDING_DEG,
        max(longitudes) + _GRID_PADDING_DEG,
    ]
    return times, latitude_bounds, longitude_bounds


def _dataset(day: str) -> xr.Dataset:
    times, latitudes, longitudes = _coordinates(day)
    temperature, u_wind, v_wind = _SENTINELS[day]
    shape = (len(times), 1, len(latitudes), len(longitudes))

    return xr.Dataset(
        data_vars={
            "temperature": (_GRID_DIMS, np.full(shape, temperature, dtype=np.float32)),
            "u_component_of_wind": (_GRID_DIMS, np.full(shape, u_wind, dtype=np.float32)),
            "v_component_of_wind": (_GRID_DIMS, np.full(shape, v_wind, dtype=np.float32)),
        },
        coords={
            "time": times,
            "level": [_PRESSURE_LEVEL_HPA],
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )


def write_local_era5_grid(root: Path) -> dict[str, Path]:
    """Write one compact, consolidated ARCO-ERA5-compatible store per corpus day."""
    root.mkdir(parents=True, exist_ok=True)
    stores: dict[str, Path] = {}

    for day in _DAYS:
        store = root / f"{day}.zarr"
        dataset = _dataset(day)
        dataset.to_zarr(str(store), mode="w", consolidated=True)
        dataset.close()
        stores[day] = store

    return stores
