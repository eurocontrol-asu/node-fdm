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


def test_download_fleet_rejects_parallel_workers() -> None:
    with pytest.raises(ValueError, match="sequential"):
        _fleet_fetch.download_fleet(_plan(), workers=2)


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

    outcomes = _fleet_fetch.download_fleet(_plan(), manifest_path=manifest)

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
