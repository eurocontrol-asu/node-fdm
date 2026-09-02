from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import node_fdm_pipeline.commands._raw_cache as raw_cache

if TYPE_CHECKING:
    from node_fdm_pipeline.config import PipelineConfig


@pytest.fixture
def cfg(tmp_path: Path) -> PipelineConfig:
    return cast("PipelineConfig", SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path)))


def _receipt_field(receipt: object, name: str) -> object:
    if isinstance(receipt, Mapping):
        return receipt[name]
    return getattr(receipt, name)


def _publish_batch(cfg: PipelineConfig) -> tuple[Path, str]:
    root = raw_cache.cache_root(cfg, "history")
    digest = raw_cache.absence_digest("2024-03-01", "history", ["z001", "z002"])
    raw_cache.publish_absence(root, "2024-03-01", "history", ["z001", "z002"])
    return root, digest


@pytest.mark.integration
def test_zero_rows_publish_a_complete_explicit_absence(cfg: PipelineConfig) -> None:
    """AC2: a zero-row publication persists its digest and exact covered set."""
    root, expected_digest = _publish_batch(cfg)

    receipt = raw_cache.read_absence_receipt(root, "2024-03-01", "history")

    assert _receipt_field(receipt, "row_count") == 0
    assert _receipt_field(receipt, "digest") == expected_digest
    assert set(cast("list[str]", _receipt_field(receipt, "icao24s"))) == {
        "z001",
        "z002",
    }


@pytest.mark.integration
def test_later_consumer_reads_receipt_from_reopened_cache_root(cfg: PipelineConfig) -> None:
    """AC2: a later consumer reloads the absence receipt solely from disk."""
    root, expected_digest = _publish_batch(cfg)
    reopened_root = Path(str(root))

    receipt = raw_cache.read_absence_receipt(reopened_root, "2024-03-01", "history")

    assert _receipt_field(receipt, "row_count") == 0
    assert _receipt_field(receipt, "digest") == expected_digest
    assert set(cast("list[str]", _receipt_field(receipt, "icao24s"))) == {
        "z001",
        "z002",
    }


@pytest.mark.integration
def test_published_absence_covers_the_requested_batch(cfg: PipelineConfig) -> None:
    """AC3: every aircraft in a published absence is treated as cached."""
    _publish_batch(cfg)

    assert all(
        raw_cache.is_cached(cfg, "history", "2024-03-01", icao24) for icao24 in ("z001", "z002")
    )
