"""Integration coverage for typed Trino failure routing during fleet acquisition."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _fleet_fetch
from node_fdm_pipeline.commands._fleet_plan import FleetPlan
from node_fdm_pipeline.config import FleetRunConfig


class _ScriptedOpenSky:
    def __init__(self, failures: dict[tuple[str, ...], str]) -> None:
        self.failures = failures
        self.calls: list[tuple[str, ...]] = []

    def history(self, *args: Any, icao24: list[str], cached: bool) -> None:
        del args
        assert cached is False
        batch = tuple(icao24)
        self.calls.append(batch)
        error = self.failures.get(batch)
        if error is not None:
            raise RuntimeError(error)


def _aircraft(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index:03d}" for index in range(1, count + 1)]


def _fleet_config(tmp_path: Path, *, max_retries: int = 3) -> FleetRunConfig:
    return FleetRunConfig(
        lease_path=tmp_path / "fleet.lease",
        lease_ttl_s=60,
        disk_min_gib=1.0,
        retry_min_delay_s=0.0,
        retry_max_delay_s=0.01,
        retry_base_s=0.01,
        retry_max_retries=max_retries,
        bisection_floor=5,
    )


def _run(
    tmp_path: Path,
    mocker: MockerFixture,
    aircraft: list[str],
    failures: dict[tuple[str, ...], str],
    *,
    max_retries: int = 3,
) -> tuple[_ScriptedOpenSky, list[dict[str, Any]]]:
    source = _ScriptedOpenSky(failures)
    plan = FleetPlan(
        dates={"20240101": aircraft},
        owner={},
        cohorts=(),
    )
    mocker.patch.object(
        _fleet_fetch,
        "_missing_by_owner",
        return_value={item: object() for item in aircraft},
    )
    journal = tmp_path / "acquisition.jsonl"
    mocker.patch.object(_fleet_fetch, "_KINDS", ("history",))
    mocker.patch.object(_fleet_fetch, "_get_opensky", return_value=source)
    mocker.patch.object(time, "sleep")

    _fleet_fetch.fetch_one_date(
        plan,
        "20240101",
        force=True,
        manifest_path=journal,
        fleet_config=_fleet_config(tmp_path, max_retries=max_retries),
    )

    records = [json.loads(line) for line in journal.read_text().splitlines()]
    return source, records


def _events(records: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [record for record in records if record["event"] == event]


@pytest.mark.integration
def test_time_limit_bisects_parent_once(tmp_path: Path, mocker: MockerFixture) -> None:
    """AC2: a time-limit fault journals two halves and submits the parent once."""
    parent = _aircraft("a", 100)

    source, records = _run(
        tmp_path,
        mocker,
        parent,
        {tuple(parent): "EXCEEDED_TIME_LIMIT: query too large"},
    )

    branches = _events(records, "fetch_branch")
    assert [record["batch"] for record in branches] == [parent[:50], parent[50:]]
    assert all(record["parent_batch"] == parent for record in branches)
    assert [record["batch"] for record in _events(records, "fetch_attempt")].count(parent) == 1
    assert source.calls.count(tuple(parent)) == 1


@pytest.mark.integration
def test_memory_limit_bisects_parent_once(tmp_path: Path, mocker: MockerFixture) -> None:
    """AC3: a memory-limit fault journals two five-aircraft branches."""
    parent = _aircraft("m", 10)

    source, records = _run(
        tmp_path,
        mocker,
        parent,
        {tuple(parent): "EXCEEDED_MEMORY_LIMIT: allocation refused"},
    )

    branches = _events(records, "fetch_branch")
    assert [record["batch"] for record in branches] == [parent[:5], parent[5:]]
    assert all(record["parent_batch"] == parent for record in branches)
    assert [record["batch"] for record in _events(records, "fetch_attempt")].count(parent) == 1
    assert source.calls.count(tuple(parent)) == 1


@pytest.mark.integration
def test_time_limit_at_floor_is_terminal_without_resubmission(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """AC4: a fault at the split floor is terminal and the child is not resubmitted."""
    parent = _aircraft("b", 10)
    first_child = parent[:5]

    source, records = _run(
        tmp_path,
        mocker,
        parent,
        {
            tuple(parent): "EXCEEDED_TIME_LIMIT: parent too large",
            tuple(first_child): "EXCEEDED_TIME_LIMIT: child too large",
        },
    )

    floor_records = [
        record
        for record in _events(records, "fetch_attempt")
        if record["batch"] == first_child and record["outcome"] == "terminal_floor"
    ]
    assert len(floor_records) == 1
    assert all(len(record["batch"]) >= 5 for record in _events(records, "fetch_branch"))
    assert source.calls.count(tuple(first_child)) == 1


@pytest.mark.integration
def test_queue_full_retries_same_batch_with_bounded_delays(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """AC5: queue saturation retries only the same batch within configured bounds."""
    parent = _aircraft("q", 10)

    source, records = _run(
        tmp_path,
        mocker,
        parent,
        {tuple(parent): "QUERY_QUEUE_FULL: retry later"},
        max_retries=3,
    )

    attempts = _events(records, "fetch_attempt")
    assert len(attempts) == 3
    assert [record["batch"] for record in attempts] == [parent, parent, parent]
    assert all(0.0 <= record["observed_delay_s"] <= 0.01 for record in attempts)
    assert _events(records, "fetch_branch") == []
    assert source.calls == [tuple(parent)] * 3


@pytest.mark.integration
def test_unknown_failure_is_terminal_without_retry_or_split(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """AC6: an unrecognized failure journals one terminal attempt only."""
    parent = _aircraft("u", 10)

    source, records = _run(
        tmp_path,
        mocker,
        parent,
        {tuple(parent): "SYNTAX_ERROR: malformed query"},
    )

    attempts = _events(records, "fetch_attempt")
    assert len(attempts) == 1
    assert attempts[0]["batch"] == parent
    assert attempts[0]["outcome"] == "terminal"
    assert _events(records, "fetch_branch") == []
    assert source.calls == [tuple(parent)]
