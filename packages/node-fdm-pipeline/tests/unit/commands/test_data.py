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


def _fake_history_flight() -> object:
    """Build a minimal ADS-B-only Flight (no BDS columns) for decoder tests."""
    import pandas as pd
    from traffic.core import Flight

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-09-01", periods=3, freq="s", tz="UTC"),
            "icao24": ["abc123"] * 3,
            "callsign": ["TEST123"] * 3,
            "latitude": [48.85, 48.86, 48.87],
            "longitude": [2.35, 2.36, 2.37],
            "altitude": [30000.0, 30100.0, 30200.0],
            "groundspeed": [450.0, 451.0, 452.0],
            "track": [90.0, 91.0, 92.0],
            "vertical_rate": [0.0, 100.0, -50.0],
        }
    )
    return Flight(df)


def test_raw_ehs_decoder_query_ehs_exception_yields_full_bds_schema(
    mocker: MockerFixture,
) -> None:
    """When `query_ehs` raises, decoder still emits every `_BDS_SOURCE_KEYS` column."""
    from node_fdm_pipeline.commands import data as data_mod

    flight = _fake_history_flight()
    mocker.patch.object(type(flight), "query_ehs", side_effect=RuntimeError("rs1090 boom"))

    decoder = data_mod._RawEHSDecoder(rawdata=None)
    out = decoder(flight)

    assert out is not None
    cols = set(out.data.columns)
    for key in data_mod._BDS_SOURCE_KEYS:
        assert key in cols, f"missing BDS source key: {key}"


def test_raw_ehs_decoder_missing_bds_subframe_yields_full_bds_schema(
    mocker: MockerFixture,
) -> None:
    """When decoded.data lacks bds40/50/60, decoder still emits every BDS source key."""
    import pandas as pd

    from node_fdm_pipeline.commands import data as data_mod

    flight = _fake_history_flight()
    fake_decoded = mocker.Mock()
    fake_decoded.data = pd.DataFrame({"timestamp": flight.data["timestamp"]})
    mocker.patch.object(type(flight), "query_ehs", return_value=fake_decoded)

    decoder = data_mod._RawEHSDecoder(rawdata=None)
    out = decoder(flight)

    assert out is not None
    cols = set(out.data.columns)
    for key in data_mod._BDS_SOURCE_KEYS:
        assert key in cols, f"missing BDS source key: {key}"


def test_flight_with_empty_bds_keys_preserves_existing_columns() -> None:
    """The helper must not overwrite columns the flight already carries."""
    import pandas as pd
    from traffic.core import Flight

    from node_fdm_pipeline.commands import data as data_mod

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-09-01", periods=2, freq="s", tz="UTC"),
            "icao24": ["abc123", "abc123"],
            "TAS": [450.0, 451.0],
        }
    )
    out = data_mod._flight_with_empty_bds_keys(Flight(df))
    assert list(out.data["TAS"]) == [450.0, 451.0]
    for key in data_mod._BDS_SOURCE_KEYS:
        assert key in out.data.columns
