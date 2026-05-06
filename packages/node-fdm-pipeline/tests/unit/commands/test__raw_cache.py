from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import pytest

from node_fdm_pipeline.commands._raw_cache import (
    cache_misses,
    cache_path,
    is_cached,
    read_partition,
    write_atomic,
)

if TYPE_CHECKING:
    from node_fdm_pipeline.config import PipelineConfig


@pytest.fixture
def cfg(tmp_path: Path) -> PipelineConfig:
    return cast("PipelineConfig", SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)))


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame({"icao24": ["4d22ad"], "ts": [1]})


def test_cache_path_history_layout(cfg: PipelineConfig) -> None:
    p = cache_path(cfg, "history", "20250901", "4d22ad")
    assert (
        p
        == cfg.paths.data_dir
        / "raw"
        / "history"
        / "date=20250901"
        / "icao24=4d22ad"
        / "data.parquet"
    )


def test_cache_path_extended_layout(cfg: PipelineConfig) -> None:
    p = cache_path(cfg, "extended", "20250901", "4d22ad")
    assert (
        p
        == cfg.paths.data_dir
        / "raw"
        / "extended"
        / "date=20250901"
        / "icao24=4d22ad"
        / "data.parquet"
    )


def test_cache_path_flightlist_layout(cfg: PipelineConfig) -> None:
    p = cache_path(cfg, "flightlist", "20250901", "4d22ad")
    assert p == cfg.paths.data_dir / "raw" / "flightlist" / "date=20250901.parquet"


def test_is_cached_hit(cfg: PipelineConfig, df: pl.DataFrame) -> None:
    p = cache_path(cfg, "history", "20250901", "4d22ad")
    write_atomic(p, df)
    assert is_cached(cfg, "history", "20250901", "4d22ad") is True


def test_is_cached_miss_empty_dir(cfg: PipelineConfig) -> None:
    assert is_cached(cfg, "history", "20250901", "4d22ad") is False


def test_is_cached_tmp_treated_as_miss(cfg: PipelineConfig) -> None:
    p = cache_path(cfg, "history", "20250901", "4d22ad")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(b"partial")
    assert is_cached(cfg, "history", "20250901", "4d22ad") is False


def test_cache_misses_empty_returns_all(cfg: PipelineConfig) -> None:
    icaos = ["a1", "a2", "a3", "a4", "a5"]
    assert list(cache_misses(cfg, "history", "20250901", icaos)) == icaos


def test_cache_misses_partial(cfg: PipelineConfig, df: pl.DataFrame) -> None:
    icaos = ["a1", "a2", "a3", "a4", "a5"]
    for i in ("a1", "a3"):
        write_atomic(cache_path(cfg, "history", "20250901", i), df)
    missing = list(cache_misses(cfg, "history", "20250901", icaos))
    assert sorted(missing) == ["a2", "a4", "a5"]


def test_write_atomic_no_tmp_remains_on_success(cfg: PipelineConfig, df: pl.DataFrame) -> None:
    p = cache_path(cfg, "history", "20250901", "4d22ad")
    write_atomic(p, df)
    assert p.exists()
    assert not p.with_suffix(p.suffix + ".tmp").exists()


def test_write_atomic_no_partial_on_crash(
    cfg: PipelineConfig, df: pl.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = cache_path(cfg, "history", "20250901", "4d22ad")

    def boom(self: Any, *a: Any, **kw: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", boom)
    with pytest.raises(RuntimeError):
        write_atomic(p, df)
    assert not p.exists()


def test_read_partition_concats_only_hits(cfg: PipelineConfig, df: pl.DataFrame) -> None:
    icaos = ["a1", "a2", "a3", "a4", "a5"]
    written = ["a1", "a2", "a4"]
    for i in written:
        write_atomic(cache_path(cfg, "history", "20250901", i), df)
    result = read_partition(cfg, "history", "20250901", icaos)
    assert result.height == len(written) * df.height


def test_read_partition_empty_when_all_miss(cfg: PipelineConfig) -> None:
    result = read_partition(cfg, "history", "20250901", ["a1", "a2"])
    assert isinstance(result, pl.DataFrame)
    assert result.height == 0
