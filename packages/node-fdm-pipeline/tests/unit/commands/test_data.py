"""Unit tests for download cache helpers in commands.data.

These tests exercise the in-memory contract of ``_ensure_window_cached``
without real OpenSky I/O: ``_raw_cache.cache_misses`` is patched to control
the miss set and the per-kind fetcher is patched to observe calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from node_fdm_pipeline.config import PipelineConfig


@pytest.fixture
def cfg(tmp_path: Path) -> PipelineConfig:
    from node_fdm_pipeline.config import PipelineConfig

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""\
paths:
  data_dir: "{tmp_path / "data"}"

typecodes:
  - A320
"""
    )
    return PipelineConfig.from_yaml(cfg_path)


def test_ensure_window_cached_skips_when_all_cached(
    cfg: PipelineConfig, mocker: MockerFixture
) -> None:
    """AC1: full cache hit -> no fetch call."""
    from node_fdm_pipeline.commands import data as data_mod

    mocker.patch("node_fdm_pipeline.commands._raw_cache.cache_misses", return_value=[])
    fetch = mocker.patch.object(data_mod, "_fetch_and_cache_window")

    data_mod._ensure_window_cached(cfg, date_str="20240101", icao24_list=["abc123", "def456"])

    assert fetch.call_count == 0


def test_ensure_window_cached_force_ignores_cache(
    cfg: PipelineConfig, mocker: MockerFixture
) -> None:
    """AC3: force=True -> fetch every icao24 regardless of cache."""
    from node_fdm_pipeline.commands import data as data_mod

    cm = mocker.patch("node_fdm_pipeline.commands._raw_cache.cache_misses", return_value=[])
    fetch = mocker.patch.object(data_mod, "_fetch_and_cache_window")

    icao = ["abc123", "def456"]
    data_mod._ensure_window_cached(cfg, date_str="20240101", icao24_list=icao, force=True)

    assert cm.call_count == 0
    assert fetch.call_count >= 1
    for call in fetch.call_args_list:
        kwargs = call.kwargs
        passed = kwargs.get("icao24_misses")
        if passed is None and len(call.args) > 2:
            passed = call.args[2]
        assert passed is not None
        assert list(passed) == icao
