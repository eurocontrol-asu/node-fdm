"""Integration tests for the diff-based ``fdm download`` command.

Real filesystem (``tmp_path``) for the parquet cache; ``traffic.data.opensky``
is mocked at the import boundary used inside ``_fetch_and_cache_window``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
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
    p.write_text(
        f"""\
paths:
  data_dir: "{tmp_path / "data"}"

typecodes:
  - A320
selected_params:
  mach:
    sigma_s: 8.0
    sigma_r: 0.01          # opensky26 tbl.3
    n_passes: 2
    slope_tol: 3.0e-4      # opensky26 tbl.3
    flat_tol: 5.0e-2
    min_len: 15
  cas:
    cutoff_s: 180.0
    sigma_s: 8.0
    sigma_r: 3.0           # opensky26 tbl.3
    n_passes: 2
    slope_tol: 0.09        # opensky26 tbl.3
    flat_tol: 20.0
    min_len: 5
  vz:
    sigma_s: 6.0
    sigma_r: 100.0         # opensky26 tbl.3
    slope_tol: 50.0        # opensky26 tbl.3
    flat_tol: 100.0
    min_len: 10
  alt:
    sigma_s: 6.0
    sigma_r: 20.0          # opensky26 tbl.3
    n_passes: 2
    tol_ftmin: 150.0
    min_len: 6
  gamma:
    sigma_s: 6.0
    sigma_r: 0.002         # opensky26 tbl.3
    slope_tol: 1.2e-3      # opensky26 tbl.3
    flat_tol: 2.0e-3
    abs_min: 5.0e-3
    min_len: 10
"""
    )
    return p


@pytest.fixture
def cfg(cfg_path: Path) -> PipelineConfig:
    from node_fdm_pipeline.config import PipelineConfig

    return PipelineConfig.from_yaml(cfg_path)


@pytest.fixture
def patch_aircraft_db(mocker: MockerFixture) -> Any:
    """Bypass aircraft_db loading and force a known icao24 list."""
    from node_fdm_pipeline.commands import data as data_mod

    aircraft_db = pl.DataFrame({"icao24": ICAO24_LIST, "typecode": ["A320"] * len(ICAO24_LIST)})
    return mocker.patch.object(
        data_mod, "_load_aircraft_db", return_value=(aircraft_db, list(ICAO24_LIST))
    )


@pytest.fixture
def mock_opensky(mocker: MockerFixture) -> MagicMock:
    """Mock the module-level ``opensky`` indirection on commands.data.

    The post-refactor module exposes ``opensky`` (or fetches via a thin wrapper);
    we monkeypatch ``node_fdm_pipeline.commands.data.opensky`` so each kind
    returns a non-empty traffic-like object whose ``.data`` is a pandas DataFrame.
    """
    import pandas as pd

    fake = MagicMock(name="opensky")

    def make_traffic(_icao24: list[str]) -> MagicMock:
        t = MagicMock()
        t.data = pd.DataFrame(
            {
                "icao24": _icao24 or ICAO24_LIST,
                "timestamp": pd.to_datetime(["2024-01-01"] * max(len(_icao24 or ICAO24_LIST), 1)),
            }
        )
        return t

    fake.history.side_effect = lambda *a, **kw: make_traffic(kw.get("icao24", []))
    fake.extended.side_effect = lambda *a, **kw: make_traffic(kw.get("icao24", []))
    fake.flightlist.side_effect = lambda *a, **kw: pd.DataFrame(
        {"icao24": kw.get("icao24", ICAO24_LIST)}
    )

    mocker.patch("node_fdm_pipeline.commands.data.opensky", fake, create=True)
    return fake


@pytest.fixture
def stub_decode(mocker: MockerFixture) -> MagicMock:
    """Stub the Phase-3 ``decode`` chain to a no-op observable."""
    from node_fdm_pipeline.commands import data as data_mod

    return cast(MagicMock, mocker.patch.object(data_mod, "decode", create=True))


def _prepopulate_cache(
    cfg: PipelineConfig, date_str: str, icao24_list: list[str], kinds: list[Kind]
) -> None:
    from node_fdm_pipeline.commands import _raw_cache

    df = pl.DataFrame({"icao24": ["x"], "ts": [0]})
    for kind in kinds:
        if kind == "flightlist":
            _raw_cache.write_atomic(_raw_cache.cache_path(cfg, kind, date_str, "_"), df)
        else:
            for icao24 in icao24_list:
                _raw_cache.write_atomic(_raw_cache.cache_path(cfg, kind, date_str, icao24), df)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_download_cold_calls_trino_per_date_per_kind(
    cfg_path: Path,
    cfg: PipelineConfig,
    patch_aircraft_db: Any,
    mock_opensky: MagicMock,
    stub_decode: MagicMock,
) -> None:
    """AC5: cold cache calls each kind once per date."""
    from node_fdm_pipeline.commands.data import download

    download(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert mock_opensky.history.call_count == 1
    assert mock_opensky.extended.call_count == 1
    assert mock_opensky.flightlist.call_count == 1


def test_download_warm_zero_trino_calls(
    cfg_path: Path,
    cfg: PipelineConfig,
    patch_aircraft_db: Any,
    mock_opensky: MagicMock,
    stub_decode: MagicMock,
) -> None:
    """AC1: full cache -> zero trino calls."""
    from node_fdm_pipeline.commands.data import download

    _prepopulate_cache(cfg, "20240101", ICAO24_LIST, ["history", "extended", "flightlist"])

    download(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert mock_opensky.history.call_count == 0
    assert mock_opensky.extended.call_count == 0
    assert mock_opensky.flightlist.call_count == 0


def test_download_partial_cache_only_misses_fetched(
    cfg_path: Path,
    cfg: PipelineConfig,
    patch_aircraft_db: Any,
    mock_opensky: MagicMock,
    stub_decode: MagicMock,
) -> None:
    """AC2: partial cache -> only missing icao24 are fetched."""
    from node_fdm_pipeline.commands.data import download

    _prepopulate_cache(cfg, "20240101", [ICAO24_LIST[0]], ["history", "extended"])

    download(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert mock_opensky.history.call_count == 1
    passed = mock_opensky.history.call_args.kwargs["icao24"]
    assert list(passed) == [ICAO24_LIST[1]]


def test_download_force_refresh_ignores_cache(
    cfg_path: Path,
    cfg: PipelineConfig,
    patch_aircraft_db: Any,
    mock_opensky: MagicMock,
    stub_decode: MagicMock,
) -> None:
    """AC3: --force-refresh fetches all icao24 even when cached."""
    from node_fdm_pipeline.commands.data import download

    _prepopulate_cache(cfg, "20240101", ICAO24_LIST, ["history", "extended", "flightlist"])

    download(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
        force_refresh=True,
    )

    assert mock_opensky.history.call_count == 1
    passed = mock_opensky.history.call_args.kwargs["icao24"]
    assert list(passed) == ICAO24_LIST


def test_download_no_decode_skips_delta_write(
    cfg_path: Path,
    cfg: PipelineConfig,
    patch_aircraft_db: Any,
    mock_opensky: MagicMock,
    stub_decode: MagicMock,
) -> None:
    """AC4: --no-decode populates raw cache, never produces flights.delta."""
    from node_fdm_pipeline.commands.data import download

    download(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
        no_decode=True,
    )

    assert stub_decode.call_count == 0
    delta_dir = Path(cfg.paths.data_dir) / "flights.delta"
    assert not delta_dir.exists()

    raw_root = Path(cfg.paths.data_dir) / "raw"
    assert raw_root.exists()


def test_download_tmp_treated_as_miss(
    cfg_path: Path,
    cfg: PipelineConfig,
    patch_aircraft_db: Any,
    mock_opensky: MagicMock,
    stub_decode: MagicMock,
) -> None:
    """AC6: a leftover .tmp file is not a cache hit."""
    from node_fdm_pipeline.commands import _raw_cache
    from node_fdm_pipeline.commands.data import download

    _prepopulate_cache(cfg, "20240101", ICAO24_LIST, ["history", "extended", "flightlist"])

    target = _raw_cache.cache_path(cfg, "history", "20240101", ICAO24_LIST[0])
    target.unlink()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(b"junk")

    download(
        config=cfg_path,
        start_date="2024-01-01",
        end_date="2024-01-02",
    )

    assert mock_opensky.history.call_count == 1
    passed = mock_opensky.history.call_args.kwargs["icao24"]
    assert ICAO24_LIST[0] in list(passed)
