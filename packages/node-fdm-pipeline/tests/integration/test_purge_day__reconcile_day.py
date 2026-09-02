"""Integration contracts for reconciliation and guarded raw-cache purge."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Literal

import polars as pl
import pytest

from node_fdm_pipeline.commands import _raw_cache
from node_fdm_pipeline.config import PathsConfig, PipelineConfig

DAY = "2026-09-01"
KIND: Literal["history"] = "history"
AIRCRAFT = ("abc123", "def456")


def _retention() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._cache_retention")


def _config(data_dir: Path) -> PipelineConfig:
    return PipelineConfig.model_construct(
        paths=PathsConfig(data_dir=data_dir),
        typecodes=["A320"],
    )


def _published_artifact(root: Path) -> tuple[_raw_cache.AbsenceReceipt, Path]:
    receipt = _raw_cache.publish_absence(root, DAY, KIND, AIRCRAFT)
    payload = root / f"date={DAY}" / ".absences" / f"{receipt.digest}.parquet"
    return receipt, payload


@pytest.mark.integration
def test_payload_without_receipt_is_invalid_and_preserved(tmp_path: Path) -> None:
    """AC1: a payload without its receipt is invalid, uncached, and preserved."""
    cfg = _config(tmp_path)
    root = _raw_cache.cache_root(cfg, KIND)
    _, payload = _published_artifact(root)
    receipt_path = root / ".absences" / f"{DAY}.{KIND}.json"
    receipt_path.unlink()

    reconciliation = _retention().reconcile_day(root, DAY, KIND)

    assert reconciliation.status == "invalid"
    assert payload.exists()
    assert _raw_cache.is_cached(cfg, KIND, DAY, AIRCRAFT[0]) is False


@pytest.mark.integration
def test_payload_digest_mismatch_is_invalid_and_preserved(tmp_path: Path) -> None:
    """AC2: a changed payload with a mismatching digest is invalid but preserved."""
    root = _raw_cache.cache_root(_config(tmp_path), KIND)
    _, payload = _published_artifact(root)
    pl.DataFrame({"icao24": ["tampered"]}).write_parquet(payload)

    reconciliation = _retention().reconcile_day(root, DAY, KIND)

    assert reconciliation.status == "invalid"
    assert payload.exists()


@pytest.mark.integration
def test_complete_artifact_is_visible_with_receipt_facts(tmp_path: Path) -> None:
    """AC3: a verified artifact exposes digest, row count, and covered aircraft."""
    root = _raw_cache.cache_root(_config(tmp_path), KIND)
    receipt, _ = _published_artifact(root)

    reconciliation = _retention().reconcile_day(root, DAY, KIND)

    assert reconciliation.status == "visible"
    assert reconciliation.digest == receipt.digest
    assert reconciliation.row_count == receipt.row_count
    assert set(reconciliation.icao24s) == set(receipt.icao24s)


@pytest.mark.integration
def test_purge_requires_visible_artifact_and_committed_consumers(tmp_path: Path) -> None:
    """AC5: purge removes a verified day only after every consumer is committed."""
    root = _raw_cache.cache_root(_config(tmp_path), KIND)
    _, payload = _published_artifact(root)
    day_directory = payload.parents[1]
    retention = _retention()

    blocked = retention.purge_day(root, DAY, KIND, {"warehouse": "pending"})

    assert blocked.allowed is False
    assert day_directory.exists()

    allowed = retention.purge_day(root, DAY, KIND, {"warehouse": "committed"})

    assert allowed.allowed is True
    assert not day_directory.exists()
