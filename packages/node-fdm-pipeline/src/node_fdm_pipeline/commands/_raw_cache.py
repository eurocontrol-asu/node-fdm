"""Raw cache module for OpenSky downloads.

Pure I/O against the on-disk parquet cache layout. No dependency on
``traffic`` / ``opensky``; only ``polars`` + ``pathlib`` + ``os``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from node_fdm_pipeline.config import PipelineConfig

__all__ = [
    "AbsenceReceipt",
    "absence_digest",
    "cache_misses",
    "cache_path",
    "cache_root",
    "is_cached",
    "publish_absence",
    "read_absence_receipt",
    "read_parquet",
    "read_partition",
    "write_atomic",
]

Kind = Literal["history", "extended", "flightlist"]


class AbsenceReceipt(BaseModel):
    """Durable proof that a batch produced no raw rows."""

    model_config = ConfigDict(frozen=True)

    day: str
    kind: Kind
    digest: str
    row_count: Literal[0] = 0
    icao24s: tuple[str, ...]


def absence_digest(day: str, kind: Kind, icao24s: Iterable[str]) -> str:
    """Return a stable digest for a day, kind, and covered aircraft set."""
    canonical_icao24s = sorted(set(icao24s))
    payload = json.dumps(
        {"day": day, "kind": kind, "icao24s": canonical_icao24s},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _absence_receipt_path(root: Path, day: str, kind: Kind) -> Path:
    return root / ".absences" / f"{day}.{kind}.json"


def _absence_artifact_path(root: Path, day: str, digest: str) -> Path:
    return root / f"date={day}" / ".absences" / f"{digest}.parquet"


def _write_atomic_receipt(path: Path, receipt: AbsenceReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(receipt.model_dump_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def publish_absence(root: Path, day: str, kind: Kind, icao24s: Iterable[str]) -> AbsenceReceipt:
    """Atomically publish a zero-row artifact and its durable receipt."""
    canonical_icao24s = tuple(sorted(set(icao24s)))
    digest = absence_digest(day, kind, canonical_icao24s)
    receipt = AbsenceReceipt(
        day=day,
        kind=kind,
        digest=digest,
        icao24s=canonical_icao24s,
    )
    artifact_path = _absence_artifact_path(root, day, digest)
    if not artifact_path.exists():
        write_atomic(artifact_path, pl.DataFrame(schema={"icao24": pl.String}))
    _write_atomic_receipt(_absence_receipt_path(root, day, kind), receipt)
    return receipt


def read_absence_receipt(root: Path, day: str, kind: Kind) -> AbsenceReceipt | None:
    """Reload an absence receipt from disk, if one was published."""
    path = _absence_receipt_path(root, day, kind)
    if not path.exists():
        return None
    return AbsenceReceipt.model_validate_json(path.read_text(encoding="utf-8"))


def cache_root(cfg: PipelineConfig, kind: Kind) -> Path:
    """Return the on-disk raw-cache root for one acquisition kind."""
    data_dir = Path(cfg.paths.data_dir)
    try:
        campaign_cache = Path(cfg.paths.era5_cache_dir)
    except AttributeError:
        staging_root = data_dir.parent
    else:
        staging_root = campaign_cache.parent if campaign_cache.is_absolute() else data_dir
    return staging_root / "raw" / kind


def _consumer_ledger_path(cfg: PipelineConfig, kind: Kind, date_str: str) -> Path:
    return cache_root(cfg, kind) / f"date={date_str}" / "consumers.json"


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_consumer_ledger(
    cfg: PipelineConfig,
    kind: Kind,
    date_str: str,
    consumers: Iterable[str],
) -> None:
    """Atomically stage the complete pending-consumer ledger for one UTC day."""
    path = _consumer_ledger_path(cfg, kind, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(raw_path)
    payload = dict.fromkeys(sorted(set(consumers)), "pending")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def cache_path(cfg: PipelineConfig, kind: Kind, date_str: str, icao24: str) -> Path:
    """Return the canonical parquet path for one raw-cache partition."""
    root = cache_root(cfg, kind)
    if kind == "flightlist":
        return root / f"date={date_str}.parquet"
    return root / f"date={date_str}" / f"icao24={icao24}" / "data.parquet"


def is_cached(cfg: PipelineConfig, kind: Kind, date_str: str, icao24: str) -> bool:
    """Return whether data or an explicit absence covers the requested aircraft."""
    root = cache_root(cfg, kind)
    try:
        receipt = read_absence_receipt(root, date_str, kind)
    except (OSError, ValueError):
        return False
    return (
        receipt is not None
        and receipt.day == date_str
        and receipt.kind == kind
        and receipt.digest == absence_digest(date_str, kind, receipt.icao24s)
        and icao24 in receipt.icao24s
        and _absence_artifact_path(root, date_str, receipt.digest).is_file()
    )


def cache_misses(
    cfg: PipelineConfig, kind: Kind, date_str: str, icao24_list: Iterable[str]
) -> list[str]:
    """Return aircraft not covered by data or a persisted absence receipt."""
    return [i for i in icao24_list if not is_cached(cfg, kind, date_str, i)]


def write_atomic(path: Path, df: pl.DataFrame) -> None:
    """Publish a parquet frame by atomically renaming a completed temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(raw_path)
    try:
        df.write_parquet(temporary_path)
        descriptor = os.open(temporary_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def read_parquet(path: Path) -> pl.DataFrame:
    """Read a parquet frame from the raw cache."""
    return pl.read_parquet(path)


def read_partition(
    cfg: PipelineConfig,
    kind: Kind,
    date_str: str,
    icao24_list: Iterable[str],
    *,
    drop_columns: Iterable[str] = (),
) -> pl.DataFrame:
    """Read every cached parquet for the given (kind, date, icao24*) set.

    *drop_columns* is dropped from each frame **before** concat — caller-supplied
    blacklist for columns whose dtype is known to vary across files (e.g. the
    OpenSky ``serials`` column, sometimes ``List[Int64]`` and sometimes ``String``).
    """
    drop_set = set(drop_columns)
    frames: list[pl.DataFrame] = []
    for icao24 in icao24_list:
        path = cache_path(cfg, kind, date_str, icao24)
        if not path.exists():
            continue
        df = read_parquet(path)
        if drop_set:
            df = df.drop([c for c in drop_set if c in df.columns])
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")
