"""Integration contracts for campaign staging retention."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

from node_fdm_pipeline.commands import _raw_cache
from node_fdm_pipeline.config import PathsConfig, PipelineConfig

KIND: Literal["history"] = "history"
AIRCRAFT = ("abc123", "def456")
RELEASED_DAY = "20191231"
PENDING_DAY = "20200101"


def _retention() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._cache_retention")


def _config(data_dir: Path) -> PipelineConfig:
    return PipelineConfig.model_construct(
        paths=PathsConfig(data_dir=data_dir),
        typecodes=["A320"],
    )


def _stage(
    cfg: PipelineConfig,
    day: str,
    consumers: dict[str, str],
) -> tuple[Path, Path, Path]:
    root = _raw_cache.cache_root(cfg, KIND)
    receipt = _raw_cache.publish_absence(root, day, KIND, AIRCRAFT)
    payload_path = root / f"date={day}" / ".absences" / f"{receipt.digest}.parquet"
    receipt_path = root / ".absences" / f"{day}.{KIND}.json"
    ledger_path = root / f"date={day}" / "consumers.json"
    ledger_path.write_text(
        json.dumps(consumers, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return payload_path, receipt_path, ledger_path


@pytest.mark.integration
def test_pending_consumer_blocks_purge(tmp_path: Path) -> None:
    """AC1: a pending C2 blocks purge and remains visible in the decision."""
    cfg = _config(tmp_path)
    root = _raw_cache.cache_root(cfg, KIND)
    payload, receipt, ledger = _stage(
        cfg,
        RELEASED_DAY,
        {"C1": "committed", "C2": "pending"},
    )

    decision = _retention().purge_day(root, RELEASED_DAY, KIND)

    assert decision.allowed is False
    assert decision.reason == "consumer_pending"
    assert "C2" in decision.model_dump_json()
    assert payload.exists()
    assert receipt.exists()
    assert ledger.exists()


@pytest.mark.integration
def test_invalid_or_missing_receipt_blocks_purge(tmp_path: Path) -> None:
    """AC2: missing and digest-mismatched receipts block committed consumers."""
    for case in ("missing", "digest_mismatch"):
        cfg = _config(tmp_path / case)
        root = _raw_cache.cache_root(cfg, KIND)
        payload, receipt_path, ledger = _stage(
            cfg,
            RELEASED_DAY,
            {"C1": "committed", "C2": "committed"},
        )
        if case == "missing":
            receipt_path.unlink()
        else:
            receipt = _raw_cache.read_absence_receipt(root, RELEASED_DAY, KIND)
            assert receipt is not None
            receipt_path.write_text(
                receipt.model_copy(
                    update={"digest": f"tampered-{receipt.digest}"}
                ).model_dump_json(),
                encoding="utf-8",
            )

        decision = _retention().purge_day(root, RELEASED_DAY, KIND)

        assert decision.allowed is False
        assert decision.reason == "receipt_invalid"
        assert payload.exists()
        assert ledger.exists()
        if case == "digest_mismatch":
            assert receipt_path.exists()


@pytest.mark.integration
def test_fully_released_verified_day_is_only_partition_purged(tmp_path: Path) -> None:
    """AC3: purge removes one released day without touching a pending day."""
    cfg = _config(tmp_path)
    root = _raw_cache.cache_root(cfg, KIND)
    released_payload, released_receipt, _ = _stage(
        cfg,
        RELEASED_DAY,
        {"C1": "committed", "C2": "committed"},
    )
    pending_payload, pending_receipt, pending_ledger = _stage(
        cfg,
        PENDING_DAY,
        {"C1": "committed", "C2": "pending"},
    )

    decision = _retention().purge_day(root, RELEASED_DAY, KIND)

    assert decision.allowed is True
    assert not released_payload.exists()
    assert not released_receipt.exists()
    assert pending_payload.exists()
    assert pending_receipt.exists()
    assert pending_ledger.exists()
