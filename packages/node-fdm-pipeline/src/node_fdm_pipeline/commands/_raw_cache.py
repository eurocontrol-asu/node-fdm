"""Raw cache module for OpenSky downloads.

Pure I/O against the on-disk parquet cache layout. No dependency on
``traffic`` / ``opensky``; only ``polars`` + ``pathlib`` + ``os``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from node_fdm_pipeline.config import PipelineConfig

__all__ = [
    "cache_misses",
    "cache_path",
    "cache_root",
    "is_cached",
    "read_parquet",
    "read_partition",
    "write_atomic",
]

Kind = Literal["history", "extended", "flightlist"]


def cache_root(cfg: PipelineConfig, kind: Kind) -> Path:
    return Path(cfg.paths.data_dir) / "raw" / kind


def cache_path(cfg: PipelineConfig, kind: Kind, date_str: str, icao24: str) -> Path:
    root = cache_root(cfg, kind)
    if kind == "flightlist":
        return root / f"date={date_str}.parquet"
    return root / f"date={date_str}" / f"icao24={icao24}" / "data.parquet"


def is_cached(cfg: PipelineConfig, kind: Kind, date_str: str, icao24: str) -> bool:
    return cache_path(cfg, kind, date_str, icao24).exists()


def cache_misses(
    cfg: PipelineConfig, kind: Kind, date_str: str, icao24_list: Iterable[str]
) -> list[str]:
    return [i for i in icao24_list if not is_cached(cfg, kind, date_str, i)]


def write_atomic(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.rename(tmp, path)


def read_parquet(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def read_partition(
    cfg: PipelineConfig, kind: Kind, date_str: str, icao24_list: Iterable[str]
) -> pl.DataFrame:
    frames = [
        read_parquet(p) for i in icao24_list if (p := cache_path(cfg, kind, date_str, i)).exists()
    ]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")
