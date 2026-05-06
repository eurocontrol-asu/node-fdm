"""Integration tests for the Phase-3 ``fdm decode`` command (AXM-1681).

Real filesystem (``tmp_path``) for the parquet cache and Delta Table.
``traffic.data.opensky`` is mocked to a tripwire that records any access.
``_RawEHSDecoder`` is patched to a passthrough so the test does not depend
on the rs1090 binary decoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import polars as pl
import pytest

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from node_fdm_pipeline.commands._raw_cache import Kind
    from node_fdm_pipeline.config import PipelineConfig

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


ICAO24_LIST = ["abc123", "def456"]


@pytest.fixture
def cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "aircraft_db.csv").write_text(
        "icao24,registration,typecode,age,airline\n"
        "abc123,F-WXYZ,A320,5,AFR\n"
        "def456,F-WXYY,A320,5,AFR\n"
    )
    p.write_text(
        f"""\
paths:
  data_dir: "{data_dir}"

typecodes:
  - A320
"""
    )
    return p


@pytest.fixture
def cfg(cfg_path: Path) -> PipelineConfig:
    from node_fdm_pipeline.config import PipelineConfig

    return PipelineConfig.from_yaml(cfg_path)


@pytest.fixture
def tripwire_opensky(mocker: MockerFixture) -> MagicMock:
    """Patch ``opensky`` with a MagicMock; AC2 asserts it stays untouched."""
    fake = MagicMock(name="opensky_tripwire")
    mocker.patch("node_fdm_pipeline.commands.data.opensky", fake, create=True)
    mocker.patch(
        "node_fdm_pipeline.commands.data._get_opensky",
        side_effect=AssertionError("_get_opensky must not be called from decode"),
    )
    return fake


@pytest.fixture
def passthrough_decoder(mocker: MockerFixture) -> Any:
    """Replace _RawEHSDecoder with a passthrough so query_ehs is never invoked."""
    from node_fdm_pipeline.commands import data as data_mod

    class _Passthrough:
        def __init__(self, rawdata: object = None) -> None:
            self.rawdata = rawdata

        def __call__(self, flight: Any) -> Any:
            return flight

    return mocker.patch.object(data_mod, "_RawEHSDecoder", _Passthrough)


def _make_history_df(icao24: str, n: int = 4) -> pl.DataFrame:
    """OpenSky-shape history rows for one icao24, pre-rename."""
    from datetime import datetime

    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, 12, 0, i) for i in range(n)],
            "icao24": [icao24] * n,
            "callsign": ["TEST01"] * n,
            "latitude": [48.0 + i * 0.001 for i in range(n)],
            "longitude": [2.0 + i * 0.001 for i in range(n)],
            "altitude": [35000.0] * n,
            "groundspeed": [440.0] * n,
            "track": [90.0] * n,
            "vertical_rate": [100.0] * n,
        }
    )


def _make_flightlist_df(icao24_list: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "icao24": icao24_list,
            "callsign": ["TEST01"] * len(icao24_list),
            "departure": ["LFPG"] * len(icao24_list),
            "arrival": ["EGLL"] * len(icao24_list),
            "typecode": ["A320"] * len(icao24_list),
        }
    )


def _prepopulate_cache(
    cfg: PipelineConfig,
    date_str: str,
    icao24_list: list[str],
    kinds: tuple[Kind, ...] = ("history", "extended", "flightlist"),
) -> None:
    from node_fdm_pipeline.commands import _raw_cache

    for kind in kinds:
        if kind == "flightlist":
            _raw_cache.write_atomic(
                _raw_cache.cache_path(cfg, kind, date_str, "_"),
                _make_flightlist_df(icao24_list),
            )
        else:
            for icao24 in icao24_list:
                _raw_cache.write_atomic(
                    _raw_cache.cache_path(cfg, kind, date_str, icao24),
                    _make_history_df(icao24),
                )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_decode_writes_delta_with_expected_columns(
    cfg_path: Path,
    cfg: PipelineConfig,
    tripwire_opensky: MagicMock,
    passthrough_decoder: Any,
) -> None:
    """AC3: decode reads the raw cache and writes flights.delta with raw_*+meta_*."""
    from node_fdm_pipeline.commands.data import decode

    _prepopulate_cache(cfg, "20240101", ICAO24_LIST)

    decode(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    delta_path = Path(cfg.paths.data_dir) / "flights.delta"
    assert delta_path.exists()
    result = pl.read_delta(str(delta_path))
    assert "raw_icao24" in result.columns
    assert "raw_timestamp" in result.columns
    assert "meta_batch_date" in result.columns
    assert "meta_aircraft_type" in result.columns


def test_decode_skips_uncached_icao24_with_warning(
    cfg_path: Path,
    cfg: PipelineConfig,
    tripwire_opensky: MagicMock,
    passthrough_decoder: Any,
    mocker: MockerFixture,
) -> None:
    """AC4: missing (date, icao24) cache entries are skipped, count logged."""
    from node_fdm_pipeline.commands import data as data_mod
    from node_fdm_pipeline.commands.data import decode

    # Only abc123 is cached; def456 is missing.
    _prepopulate_cache(cfg, "20240101", ["abc123"], kinds=("history", "extended"))
    # flightlist cached for both so the join has data
    from node_fdm_pipeline.commands import _raw_cache

    _raw_cache.write_atomic(
        _raw_cache.cache_path(cfg, "flightlist", "20240101", "_"),
        _make_flightlist_df(ICAO24_LIST),
    )

    warn_spy = mocker.spy(data_mod.log, "warning")

    decode(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    delta_path = Path(cfg.paths.data_dir) / "flights.delta"
    result = pl.read_delta(str(delta_path))
    assert set(result["raw_icao24"].unique().to_list()) == {"abc123"}

    skip_calls = [c for c in warn_spy.call_args_list if c.args and "skipped" in c.args[0]]
    assert len(skip_calls) == 1
    assert skip_calls[0].kwargs.get("count") == 1


def test_decode_icao24_filter_subsets(
    cfg_path: Path,
    cfg: PipelineConfig,
    tripwire_opensky: MagicMock,
    passthrough_decoder: Any,
    tmp_path: Path,
) -> None:
    """AC5: --icao24-filter restricts decoding to the intersection."""
    from node_fdm_pipeline.commands.data import decode

    _prepopulate_cache(cfg, "20240101", ICAO24_LIST)

    filter_path = tmp_path / "filter.txt"
    filter_path.write_text("abc123\n")

    decode(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
        icao24_filter=filter_path,
    )

    delta_path = Path(cfg.paths.data_dir) / "flights.delta"
    result = pl.read_delta(str(delta_path))
    assert set(result["raw_icao24"].unique().to_list()) == {"abc123"}


def test_decode_zero_network(
    cfg_path: Path,
    cfg: PipelineConfig,
    tripwire_opensky: MagicMock,
    passthrough_decoder: Any,
) -> None:
    """AC2: decode never accesses ``opensky`` (history/extended/flightlist)."""
    from node_fdm_pipeline.commands.data import decode

    _prepopulate_cache(cfg, "20240101", ICAO24_LIST)

    decode(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert tripwire_opensky.mock_calls == []
    assert tripwire_opensky.history.call_count == 0
    assert tripwire_opensky.extended.call_count == 0
    assert tripwire_opensky.flightlist.call_count == 0


def test_decode_idempotent(
    cfg_path: Path,
    cfg: PipelineConfig,
    tripwire_opensky: MagicMock,
    passthrough_decoder: Any,
) -> None:
    """AC6: decode twice over identical fixture -> same row count + columns."""
    from node_fdm_pipeline.commands.data import decode

    _prepopulate_cache(cfg, "20240101", ICAO24_LIST)
    delta_path = Path(cfg.paths.data_dir) / "flights.delta"

    decode(config=cfg_path, start_date="2024-01-01", end_date="2024-01-02")
    first = pl.read_delta(str(delta_path))

    decode(config=cfg_path, start_date="2024-01-01", end_date="2024-01-02")
    second = pl.read_delta(str(delta_path))

    assert len(first) == len(second)
    assert set(first.columns) == set(second.columns)


def test_decode_matches_download_chain_columns(
    cfg_path: Path,
    cfg: PipelineConfig,
    passthrough_decoder: Any,
    mocker: MockerFixture,
) -> None:
    """AC7: download (auto-chain) and standalone decode produce the same column set."""
    import pandas as pd

    from node_fdm_pipeline.commands.data import decode, download

    # Mock opensky for the ``download`` run only — it writes to the raw cache,
    # then chains decode() which performs the actual Delta build.
    fake = MagicMock(name="opensky")

    def make_traffic(icao24: list[str]) -> MagicMock:
        t = MagicMock()
        rows = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 12:00:00"] * max(len(icao24), 1)),
                "icao24": icao24 or [ICAO24_LIST[0]],
                "callsign": ["TEST01"] * max(len(icao24), 1),
                "latitude": [48.0] * max(len(icao24), 1),
                "longitude": [2.0] * max(len(icao24), 1),
                "altitude": [35000.0] * max(len(icao24), 1),
                "groundspeed": [440.0] * max(len(icao24), 1),
                "track": [90.0] * max(len(icao24), 1),
                "vertical_rate": [100.0] * max(len(icao24), 1),
            }
        )
        t.data = rows
        return t

    fake.history.side_effect = lambda *a, **kw: make_traffic(kw.get("icao24", []))
    fake.extended.side_effect = lambda *a, **kw: make_traffic(kw.get("icao24", []))
    fake.flightlist.side_effect = lambda *a, **kw: pd.DataFrame(
        {
            "icao24": kw.get("icao24", ICAO24_LIST),
            "callsign": ["TEST01"] * len(kw.get("icao24", ICAO24_LIST)),
            "departure": ["LFPG"] * len(kw.get("icao24", ICAO24_LIST)),
            "arrival": ["EGLL"] * len(kw.get("icao24", ICAO24_LIST)),
            "typecode": ["A320"] * len(kw.get("icao24", ICAO24_LIST)),
        }
    )
    mocker.patch("node_fdm_pipeline.commands.data.opensky", fake, create=True)
    mocker.patch("node_fdm_pipeline.commands.data._require_traffic")

    download(config=cfg_path, start_date="2024-01-01", end_date="2024-01-02")
    delta_path = Path(cfg.paths.data_dir) / "flights.delta"
    download_cols = set(pl.read_delta(str(delta_path)).columns)

    # Now wipe Delta and run decode standalone over the same cache.
    import shutil

    shutil.rmtree(delta_path)
    decode(config=cfg_path, start_date="2024-01-01", end_date="2024-01-02")
    decode_cols = set(pl.read_delta(str(delta_path)).columns)

    assert download_cols == decode_cols
