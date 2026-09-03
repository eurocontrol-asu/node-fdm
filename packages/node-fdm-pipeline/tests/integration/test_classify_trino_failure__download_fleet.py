"""Campaign coverage for classified Trino failures and attempt attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd
import pytest

from _config_fixtures import write_config
from node_fdm_pipeline.commands import _fleet_fetch, _raw_cache
from node_fdm_pipeline.commands._fleet_digest import DigestInput, compute_resume_digest
from node_fdm_pipeline.commands._fleet_fetch import DateOutcome, download_fleet
from node_fdm_pipeline.commands._fleet_journal import replay_attempts
from node_fdm_pipeline.commands._fleet_plan import FleetPlan, build_fleet_plan
from node_fdm_pipeline.commands._fleet_selection import compile_selection
from node_fdm_pipeline.config import FleetRunConfig

_DATE = "20240101"


class _ScriptedSource:
    def __init__(self, failures: dict[tuple[str, ...], list[str]]) -> None:
        self.failures = failures
        self.calls: list[tuple[str, ...]] = []

    def history(
        self,
        *_args: object,
        icao24: list[str],
        cached: bool,
    ) -> pd.DataFrame:
        assert cached is False
        batch = tuple(icao24)
        self.calls.append(batch)
        scripted = self.failures.get(batch, [])
        if scripted:
            raise RuntimeError(scripted.pop(0))
        return pd.DataFrame(
            {
                "icao24": icao24,
                "timestamp": [1704067200 + index for index in range(len(icao24))],
            }
        )


def _aircraft(prefix: str) -> list[str]:
    return [f"{prefix}{index:03d}" for index in range(1, 11)]


def _campaign(tmp_path: Path, aircraft: list[str]) -> FleetPlan:
    config_path = tmp_path / "C1.yaml"
    write_config(config_path, tmp_path / "C1")
    rows = [
        {
            "icao24": icao24,
            "callsign": icao24.upper(),
            "firstseen": 1704067200 + index,
            "lastseen": 1704070800 + index,
            "msn": icao24,
            "split": "train",
            "selection_id": f"selection-{icao24}",
            "cohort": "C1",
        }
        for index, icao24 in enumerate(aircraft)
    ]
    selection = compile_selection(rows)
    return build_fleet_plan(
        [("C1", config_path, tmp_path / "C1-selection.csv")],
        data_root=tmp_path / "campaign-staging",
        selection=selection,
    )


def _run_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aircraft: list[str],
    failures: dict[tuple[str, ...], list[str]],
) -> tuple[list[DateOutcome], FleetPlan, Path]:
    plan = _campaign(tmp_path, aircraft)
    source = _ScriptedSource(failures)
    manifest = tmp_path / "download-fleet.manifest.jsonl"
    selection_digest = plan.selection.digest if plan.selection is not None else "missing"
    resolved_config: DigestInput = {"workers": 1, "mode": "fleet"}
    profile: DigestInput = {"aircraft": "A320", "version": 1}
    fleet_config = FleetRunConfig(
        lease_path=tmp_path / "fleet.lease",
        lease_ttl_s=60,
        disk_min_gib=0.000001,
        retry_min_delay_s=0.01,
        retry_max_delay_s=0.02,
        retry_base_s=0.01,
        retry_max_retries=2,
        bisection_floor=5,
    )
    seeded_random = Random(7)  # noqa: S311 - deterministic retry jitter

    monkeypatch.setattr(_fleet_fetch, "_KINDS", ("history",))
    monkeypatch.setattr(_fleet_fetch, "_get_opensky", lambda: source)
    monkeypatch.setattr(
        "node_fdm_pipeline.commands._fleet_fetch.random.Random",
        lambda: seeded_random,
    )
    monkeypatch.setattr(
        "node_fdm_pipeline.commands._fleet_fetch.time.sleep",
        lambda _delay: None,
    )

    outcomes = download_fleet(
        plan,
        force=True,
        manifest_path=manifest,
        fleet_config=fleet_config,
        recorded_digest=compute_resume_digest(selection_digest, resolved_config, profile),
        selection_digest=selection_digest,
        resolved_config=resolved_config,
        profile=profile,
    )
    return list(outcomes), plan, manifest


def _manifest_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _staged_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.mark.integration
def test_succeeded_half_survives_terminal_floor_day(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2: a successful sibling stays staged and attested after floor exhaustion."""
    aircraft = _aircraft("s")
    succeeded = aircraft[:5]
    failed = aircraft[5:]

    outcomes, plan, manifest = _run_campaign(
        tmp_path,
        monkeypatch,
        aircraft,
        {
            tuple(aircraft): ["EXCEEDED_TIME_LIMIT: parent too large"],
            tuple(failed): ["EXCEEDED_TIME_LIMIT: child still too large"],
        },
    )

    payloads = [
        _raw_cache.cache_path(plan.cohorts[0].cfg, "history", _DATE, icao24)
        for icao24 in succeeded
    ]
    receipt = next(
        event for event in _manifest_events(manifest) if event["event"] == "date_finished"
    )
    assert outcomes[0].written == len(succeeded)
    assert all(path.is_file() for path in payloads)
    assert receipt["staged_icao24s"] == succeeded
    assert receipt["staged_digest"] == _staged_digest(payloads)


@pytest.mark.integration
def test_attempts_carry_their_selection_acquisition_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3: replayed attempts identify every selection served by their batch."""
    aircraft = _aircraft("r")

    outcomes, plan, manifest = _run_campaign(
        tmp_path,
        monkeypatch,
        aircraft,
        {tuple(aircraft): ["QUERY_QUEUE_FULL: retry later"]},
    )

    events = _manifest_events(manifest)
    attempts = replay_attempts(events)
    expected_keys = tuple(
        flight.acquisition_key
        for icao24 in aircraft
        if (flight := plan.resolve(icao24, _DATE)) is not None
    )
    assert outcomes[0].error is None
    assert len(attempts) == 2
    assert attempts[0].attempt == 1
    assert attempts[0].outcome == "retry_later"
    assert 0.01 <= attempts[0].observed_delay_s <= 0.02
    assert attempts[0].acquisition_keys == expected_keys
    assert attempts[1].attempt == 2
    assert attempts[1].outcome == "success"
    assert attempts[1].batch == tuple(aircraft)
    assert attempts[1].acquisition_keys == expected_keys
    assert not any(event["event"] == "fetch_branch" for event in events)
