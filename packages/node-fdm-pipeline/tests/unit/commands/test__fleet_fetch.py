"""Unit tests for the strictly sequential fleet downloader."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _fleet_fetch
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigestMismatch,
    compute_resume_digest,
)
from node_fdm_pipeline.config import FleetRunConfig

_SELECTION_DIGEST = "selection-v1"
_RESOLVED_CONFIG: DigestInput = {"workers": 1, "mode": "fleet"}
_PROFILE: DigestInput = {"aircraft": "A320", "version": 1}


def _download_kwargs(lease_path: Path, run_dir: Path) -> dict[str, Any]:
    return {
        "fleet_config": FleetRunConfig(
            lease_path=lease_path,
            lease_ttl_s=60,
            disk_min_gib=1.0,
        ),
        "recorded_digest": compute_resume_digest(
            _SELECTION_DIGEST,
            _RESOLVED_CONFIG,
            _PROFILE,
        ),
        "selection_digest": _SELECTION_DIGEST,
        "resolved_config": _RESOLVED_CONFIG,
        "profile": _PROFILE,
        "journal_path": run_dir / "download-fleet.journal.jsonl",
        "receipt_dir": run_dir / "receipts",
    }


def _plan() -> Any:
    return SimpleNamespace(
        cohorts=(SimpleNamespace(name="A"),),
        dates={"20240101": ["abc123"], "20240102": ["def456"]},
        aircraft_days=2,
        requests_saved=lambda: (2, 2),
    )


def test_fetch_disables_pyopensky_cache() -> None:
    calls: list[dict[str, Any]] = []

    def fetcher(*args: Any, **kwargs: Any) -> str:
        calls.append(kwargs)
        return "ok"

    result = _fleet_fetch._fetch_with_retry(
        fetcher,
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        ["abc123"],
        date_str="20240101",
    )

    assert result == "ok"
    assert calls == [{"icao24": ["abc123"], "cached": False}]


def test_download_fleet_rejects_parallel_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sequential"):
        _fleet_fetch.download_fleet(
            _plan(),
            workers=2,
            **_download_kwargs(tmp_path / "fleet.lease", tmp_path),
        )


def test_download_fleet_is_ordered_and_appends_manifest(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    seen: list[str] = []

    def fetch_one_date(plan: Any, date: str, *, force: bool = False) -> Any:
        seen.append(date)
        return SimpleNamespace(
            date=date,
            requested=1,
            written=3,
            requests=3,
            empty_kinds=(),
            error=None,
        )

    mocker.patch.object(_fleet_fetch, "_get_opensky")
    mocker.patch.object(_fleet_fetch, "fetch_one_date", side_effect=fetch_one_date)
    manifest = tmp_path / "download-fleet.manifest.jsonl"

    outcomes = _fleet_fetch.download_fleet(
        _plan(),
        manifest_path=manifest,
        **_download_kwargs(tmp_path / "fleet.lease", tmp_path),
    )

    assert seen == ["20240101", "20240102"]
    assert [outcome.date for outcome in outcomes] == seen
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "run_started",
        "date_finished",
        "date_finished",
        "run_finished",
    ]
    assert len(records[0]["plan_sha256"]) == 64
    assert records[-1]["failed"] == 0


def test_download_fleet_rejects_mismatching_recorded_digest() -> None:
    """AC1: a recorded resume digest that differs from current inputs is rejected."""
    recorded = compute_resume_digest(
        "stale-selection",
        _RESOLVED_CONFIG,
        _PROFILE,
    )
    fleet_config = FleetRunConfig(
        lease_path=Path("/shared/download-fleet.lease"),
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )

    with pytest.raises(ResumeDigestMismatch, match=r"\bselection\b"):
        _fleet_fetch.download_fleet(
            _plan(),
            fleet_config=fleet_config,
            recorded_digest=recorded,
            selection_digest=_SELECTION_DIGEST,
            resolved_config=_RESOLVED_CONFIG,
            profile=_PROFILE,
        )
