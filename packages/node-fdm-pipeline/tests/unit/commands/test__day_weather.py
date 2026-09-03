from __future__ import annotations

import importlib

import pytest


def test_three_cohorts_share_one_open_grid() -> None:
    """AC1: three cohorts share one opened handle for one source-day field."""
    weather = importlib.import_module("node_fdm_pipeline.commands._day_weather")
    opened: list[tuple[str, str, object]] = []

    def opener(source_day: str, field: str) -> object:
        handle = object()
        opened.append((source_day, field, handle))
        return handle

    cache = weather.DayGridCache(opener=opener, closer=lambda _handle: None)

    handle_a = cache.acquire("20200101", "t2m")
    handle_b = cache.acquire("20200101", "t2m")
    handle_c = cache.acquire("20200101", "t2m")

    assert handle_a is handle_b is handle_c
    assert cache.stats().opens[("20200101", "t2m")] == 1
    assert len(opened) == 1


def test_distinct_utc_source_day_opens_own_grid() -> None:
    """AC2: a distinct UTC source day owns a distinct cache entry and open."""
    weather = importlib.import_module("node_fdm_pipeline.commands._day_weather")

    def opener(_source_day: str, _field: str) -> object:
        return object()

    cache = weather.DayGridCache(opener=opener, closer=lambda _handle: None)

    cache.acquire("20200101", "t2m")
    cache.acquire("20200102", "t2m")
    stats = cache.stats()

    assert stats.total_opens == 2
    assert set(stats.opens) == {
        ("20200101", "t2m"),
        ("20200102", "t2m"),
    }


def test_evict_refuses_grid_with_remaining_consumer() -> None:
    """AC3: eviction reports the day and count while a consumer remains."""
    weather = importlib.import_module("node_fdm_pipeline.commands._day_weather")

    def opener(_source_day: str, _field: str) -> object:
        return object()

    cache = weather.DayGridCache(opener=opener, closer=lambda _handle: None)
    cache.acquire("20200101", "t2m")
    cache.acquire("20200101", "t2m")
    cache.release("20200101", "t2m")

    with pytest.raises(weather.GridStillReferenced) as exc_info:
        cache.evict("20200101")

    message = str(exc_info.value)
    assert "20200101" in message
    assert "1" in message.replace("20200101", "")
