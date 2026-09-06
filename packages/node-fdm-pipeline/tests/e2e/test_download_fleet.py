"""End-to-end coverage for the download-fleet command."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import pytest

from _config_fixtures import write_config


def _recorded_fleet(tmp_path: Path) -> Path:
    fleet_dir = tmp_path / "fleet"
    cohort_dir = fleet_dir / "types" / "A320"
    results_dir = cohort_dir / "results"
    data_dir = tmp_path / "data"
    results_dir.mkdir(parents=True)
    data_dir.mkdir()
    write_config(cohort_dir / "config.yaml", data_dir)
    (results_dir / "selection_A320.csv").write_text(
        "icao24,day\nabc123,2024-01-01\ndef456,2024-01-02\n",
        encoding="utf-8",
    )
    return fleet_dir


@pytest.mark.e2e
def test_download_fleet_historical_dry_run_lists_recorded_dates(tmp_path: Path) -> None:
    """AC1: a historical dry-run exits zero and prints every recorded planned date."""
    fleet_dir = _recorded_fleet(tmp_path)

    completed = subprocess.run(
        ["fdm", "download-fleet", "--fleet-dir", str(fleet_dir), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "20240101" in completed.stdout
    assert "20240102" in completed.stdout


def _single_date_recorded_fleet(tmp_path: Path) -> Path:
    fleet_dir = _recorded_fleet(tmp_path)
    selection = fleet_dir / "types" / "A320" / "results" / "selection_A320.csv"
    selection.write_text("icao24,day\nabc123,2024-01-01\n", encoding="utf-8")
    return fleet_dir


def _provider_sitecustomize(tmp_path: Path) -> tuple[Path, Path]:
    patch_dir = tmp_path / "provider-double"
    patch_dir.mkdir()
    counter_path = tmp_path / "provider-calls.txt"
    (patch_dir / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "import os\n"
        "from node_fdm_pipeline.commands import _fleet_fetch\n"
        "_original_download_fleet = _fleet_fetch.download_fleet\n"
        "def _counting_fetch(plan, date, *, force=False):\n"
        "    path = Path(os.environ['FDM_PROVIDER_COUNT'])\n"
        "    count = int(path.read_text(encoding='utf-8')) if path.exists() else 0\n"
        "    path.write_text(str(count + 1), encoding='utf-8')\n"
        "    return SimpleNamespace(date=date, requested=1, written=1, requests=1, "
        "empty_kinds=(), error=None)\n"
        "def _download_with_double(*args, **kwargs):\n"
        "    kwargs.setdefault('fetch_boundary', _counting_fetch)\n"
        "    return _original_download_fleet(*args, **kwargs)\n"
        "_fleet_fetch.download_fleet = _download_with_double\n",
        encoding="utf-8",
    )
    return patch_dir, counter_path


@pytest.mark.e2e
def test_download_fleet_cli_reports_mutated_campaign_identity_without_provider_call(
    tmp_path: Path,
) -> None:
    """AC5: the CLI reports a resume mismatch without a second provider call."""
    fleet_dir = _single_date_recorded_fleet(tmp_path)
    patch_dir, counter_path = _provider_sitecustomize(tmp_path)
    selection = tmp_path / "selection.txt"
    resolved_config = tmp_path / "resolved-config.json"
    profile = tmp_path / "profile.json"
    selection.write_text("recorded-offline-selection", encoding="utf-8")
    resolved_config.write_text('{"workers": 1}', encoding="utf-8")
    profile.write_text('{"aircraft": "A320"}', encoding="utf-8")
    lease_path = tmp_path / "shared" / "download-fleet.lease"
    lease_path.parent.mkdir()
    env = os.environ.copy()
    env["FDM_PROVIDER_COUNT"] = str(counter_path)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(patch_dir), env.get("PYTHONPATH", ""))))
    command = [
        "fdm",
        "download-fleet",
        "--fleet-dir",
        str(fleet_dir),
        "--workers",
        "1",
        "--selection",
        str(selection),
        "--resolved-config",
        str(resolved_config),
        "--profile",
        str(profile),
        "--lease-path",
        str(lease_path),
        "--lease-ttl-s",
        "60",
        "--disk-min-gib",
        "1",
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    assert first.returncode == 0, first.stderr
    assert counter_path.read_text(encoding="utf-8") == "1"
    profile.write_text('{"aircraft": "A321"}', encoding="utf-8")

    rejected = subprocess.run(command, check=False, capture_output=True, text=True, env=env)

    assert rejected.returncode != 0
    assert "mismatch" in f"{rejected.stdout}\n{rejected.stderr}".lower()
    assert counter_path.read_text(encoding="utf-8") == "1"


def _recorded_source(tmp_path: Path, *, delay_s: float) -> Path:
    source = tmp_path / "recorded-opensky.json"
    source.write_text(
        json.dumps(
            {
                "delay_s": delay_s,
                "responses": {
                    "history": None,
                    "extended": None,
                    "flightlist": None,
                },
            }
        ),
        encoding="utf-8",
    )
    return source


@dataclass(frozen=True)
class _CampaignOptions:
    name: str
    lease_path: Path
    recorded_source: Path
    journal_path: Path
    receipt_dir: Path
    wait_budget_s: float
    poll_interval_s: float


@dataclass(frozen=True, slots=True)
class _ConcurrentCampaignResult:
    first: subprocess.CompletedProcess[str]
    second: subprocess.CompletedProcess[str]
    journal_path: Path


_CONCURRENT_CAMPAIGN_RUNS = 0


def _campaign_command(tmp_path: Path, options: _CampaignOptions) -> list[str]:
    campaign_dir = tmp_path / options.name
    fleet_dir = _single_date_recorded_fleet(campaign_dir)
    selection = campaign_dir / "selection.txt"
    resolved_config = campaign_dir / "resolved-config.json"
    profile = campaign_dir / "profile.json"
    selection.write_text(f"recorded-offline-selection-{options.name}", encoding="utf-8")
    resolved_config.write_text('{"workers": 1}', encoding="utf-8")
    profile.write_text(json.dumps({"campaign": options.name}), encoding="utf-8")
    return [
        "fdm",
        "download-fleet",
        "--fleet-dir",
        str(fleet_dir),
        "--workers",
        "1",
        "--selection",
        str(selection),
        "--resolved-config",
        str(resolved_config),
        "--profile",
        str(profile),
        "--lease-path",
        str(options.lease_path),
        "--lease-ttl-s",
        "5",
        "--lease-wait-budget-s",
        str(options.wait_budget_s),
        "--lease-poll-interval-s",
        str(options.poll_interval_s),
        "--disk-min-gib",
        "0.001",
        "--recorded-source",
        str(options.recorded_source),
        "--acquisition-journal",
        str(options.journal_path),
        "--acquisition-receipt-dir",
        str(options.receipt_dir),
    ]


class _JournalEvent(TypedDict):
    state: str
    timestamp: str
    receipt: dict[str, object]


def _journal_events(journal_path: Path) -> list[_JournalEvent]:
    events: list[_JournalEvent] = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        event = cast("dict[str, object]", json.loads(line))
        receipt_path = Path(str(event["receipt"]))
        receipt = cast(
            "dict[str, object]",
            json.loads(receipt_path.read_text(encoding="utf-8")),
        )
        events.append(
            {
                "state": str(event["state"]),
                "timestamp": str(event["timestamp"]),
                "receipt": receipt,
            }
        )
    return events


def _wait_for_state(journal_path: Path, state: str, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if journal_path.exists() and any(
            event["state"] == state for event in _journal_events(journal_path)
        ):
            return
        time.sleep(0.01)
    raise AssertionError(f"journal never reached {state!r}")


def _run_two_campaigns(tmp_path: Path) -> _ConcurrentCampaignResult:
    lease_path = tmp_path / "shared" / "download-fleet.lease"
    lease_path.parent.mkdir()
    journal_path = tmp_path / "shared" / "acquisition.jsonl"
    receipt_dir = tmp_path / "shared" / "receipts"
    recorded_source = _recorded_source(tmp_path, delay_s=0.4)
    first_command = _campaign_command(
        tmp_path,
        _CampaignOptions(
            name="campaign-a",
            lease_path=lease_path,
            recorded_source=recorded_source,
            journal_path=journal_path,
            receipt_dir=receipt_dir,
            wait_budget_s=2.0,
            poll_interval_s=0.02,
        ),
    )
    second_command = _campaign_command(
        tmp_path,
        _CampaignOptions(
            name="campaign-b",
            lease_path=lease_path,
            recorded_source=recorded_source,
            journal_path=journal_path,
            receipt_dir=receipt_dir,
            wait_budget_s=2.0,
            poll_interval_s=0.02,
        ),
    )

    first = subprocess.Popen(
        first_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _wait_for_state(journal_path, "boundary_entered")
    second = subprocess.Popen(
        second_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    first_stdout, first_stderr = first.communicate(timeout=10)
    second_stdout, second_stderr = second.communicate(timeout=10)
    return _ConcurrentCampaignResult(
        first=subprocess.CompletedProcess(
            first_command,
            first.returncode,
            first_stdout,
            first_stderr,
        ),
        second=subprocess.CompletedProcess(
            second_command,
            second.returncode,
            second_stdout,
            second_stderr,
        ),
        journal_path=journal_path,
    )


@pytest.fixture(scope="module")
def concurrent_campaign(
    tmp_path_factory: pytest.TempPathFactory,
) -> _ConcurrentCampaignResult:
    global _CONCURRENT_CAMPAIGN_RUNS
    _CONCURRENT_CAMPAIGN_RUNS += 1
    assert _CONCURRENT_CAMPAIGN_RUNS == 1
    return _run_two_campaigns(tmp_path_factory.mktemp("concurrent-campaign"))


@pytest.mark.e2e
def test_two_concurrent_campaign_processes_both_complete(
    concurrent_campaign: _ConcurrentCampaignResult,
) -> None:
    """AC1: a live shared lease serialises two campaign CLI processes without rejecting either."""

    first, second = concurrent_campaign.first, concurrent_campaign.second

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr


@pytest.mark.e2e
def test_concurrent_campaign_boundary_sections_never_overlap(
    concurrent_campaign: _ConcurrentCampaignResult,
) -> None:
    """AC2: the shared journal proves that concurrent CLI boundaries never overlap."""

    first, second = concurrent_campaign.first, concurrent_campaign.second
    assert first.returncode == second.returncode == 0
    events = sorted(
        _journal_events(concurrent_campaign.journal_path),
        key=lambda event: str(event["timestamp"]),
    )
    boundary_events = [
        event for event in events if event["state"] in {"boundary_entered", "lease_released"}
    ]

    active = 0
    peak = 0
    for event in boundary_events:
        active += 1 if event["state"] == "boundary_entered" else -1
        peak = max(peak, active)

    assert [event["state"] for event in boundary_events] == [
        "boundary_entered",
        "lease_released",
        "boundary_entered",
        "lease_released",
    ]
    assert peak == 1
    assert active == 0
    assert str(boundary_events[1]["timestamp"]) < str(boundary_events[2]["timestamp"])


@pytest.mark.e2e
def test_foreign_live_lease_waits_for_budget_before_giving_up(tmp_path: Path) -> None:
    """AC3: a foreign live holder is journalled and rejected only after the wait budget."""

    lease_path = tmp_path / "shared" / "download-fleet.lease"
    lease_path.parent.mkdir()
    now = time.time()
    lease_path.write_text(
        json.dumps(
            {
                "owner": "foreign-owner",
                "acquired_at": now,
                "heartbeat_at": now,
                "expires_at": now + 60.0,
            }
        ),
        encoding="utf-8",
    )
    journal_path = tmp_path / "shared" / "acquisition.jsonl"
    receipt_dir = tmp_path / "shared" / "receipts"
    command = _campaign_command(
        tmp_path,
        _CampaignOptions(
            name="waiting-campaign",
            lease_path=lease_path,
            recorded_source=_recorded_source(tmp_path, delay_s=0.0),
            journal_path=journal_path,
            receipt_dir=receipt_dir,
            wait_budget_s=0.25,
            poll_interval_s=0.05,
        ),
    )

    started = time.monotonic()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    elapsed_s = time.monotonic() - started

    events = _journal_events(journal_path)
    waits = [event for event in events if event["state"] == "waiting"]
    assert elapsed_s >= 0.25
    assert completed.returncode != 0
    assert "LeaseUnavailable" in completed.stderr
    assert "foreign-owner" in completed.stderr
    assert waits
    assert all(event["receipt"]["holder"] == "foreign-owner" for event in waits)
