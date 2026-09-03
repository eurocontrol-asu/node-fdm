from __future__ import annotations

import importlib
from pathlib import Path

import polars as pl
import pytest


@pytest.mark.integration
def test_zero_consumer_evicts_fields_and_reopens_grid(tmp_path: Path) -> None:
    """AC4: fully released fields are evicted, closed, and opened again on demand."""
    weather = importlib.import_module("node_fdm_pipeline.commands._day_weather")
    source_day = "20200101"
    for field, value in (("t2m", 273.15), ("u10", 12.5)):
        pl.DataFrame({"value": [value]}).write_parquet(tmp_path / f"{source_day}_{field}.parquet")

    def opener(day: str, field: str) -> pl.DataFrame:
        return pl.read_parquet(tmp_path / f"{day}_{field}.parquet")

    closed: list[pl.DataFrame] = []

    def closer(handle: pl.DataFrame) -> None:
        closed.append(handle)

    cache = weather.DayGridCache(opener=opener, closer=closer)
    first_t2m = cache.acquire(source_day, "t2m")
    assert cache.acquire(source_day, "t2m") is first_t2m
    first_u10 = cache.acquire(source_day, "u10")
    assert cache.acquire(source_day, "u10") is first_u10

    cache.release(source_day, "t2m")
    cache.release(source_day, "t2m")
    cache.release(source_day, "u10")
    cache.release(source_day, "u10")

    assert cache.evict(source_day) == ("t2m", "u10")
    assert len(closed) == 2
    assert any(handle is first_t2m for handle in closed)
    assert any(handle is first_u10 for handle in closed)

    reopened_t2m = cache.acquire(source_day, "t2m")

    assert reopened_t2m is not first_t2m
    assert cache.stats().opens[(source_day, "t2m")] == 2
