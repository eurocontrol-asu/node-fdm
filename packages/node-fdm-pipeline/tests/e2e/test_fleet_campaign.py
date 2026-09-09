"""Black-box tests for the fleet-campaign operator CLI."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

from _config_fixtures import SELECTED_PARAMS_YAML
from node_fdm_pipeline import cli, run_fleet_campaign

pytestmark = pytest.mark.e2e

_EXPECTED_MODES = {"plan", "resume", "run", "status", "validate"}
_OPERATOR_FIGURES = (
    "duration",
    "bytes",
    "rows",
    "flight_identities",
    "rejects",
    "split_batches",
    "cache_hits",
    "cache_misses",
    "throughput",
    "eta",
    "errors",
    "caches",
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _offline_env() -> dict[str, str]:
    env = os.environ.copy()
    dead_proxy = "http://127.0.0.1:9"
    env.update(
        {
            "ALL_PROXY": dead_proxy,
            "HTTP_PROXY": dead_proxy,
            "HTTPS_PROXY": dead_proxy,
            "NO_PROXY": "",
        }
    )
    return env


def _write_campaign(tmp_path: Path, *, utc_days: list[str] | None = None) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            [
                {
                    "icao24": "abc123",
                    "callsign": "AXM001",
                    "firstseen": 1735689600,
                    "lastseen": 1735693200,
                    "msn": "MSN-1",
                    "split": "train",
                    "cohort": "A320",
                    "selection_id": "selection-1",
                    "utc_days": utc_days or ["2025-01-01"],
                }
            ]
        ),
        encoding="utf-8",
    )
    recorded_opensky = tmp_path / "recorded-opensky.json"
    recorded_opensky.write_text(
        json.dumps(
            {
                "delay_s": 0.05,
                "responses": {"history": None, "extended": None, "flightlist": None},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "campaign.yaml"
    config.write_text(
        (
            "paths:\n"
            f"  data_dir: {json.dumps(str(data_dir))}\n"
            "typecodes:\n"
            "  - A320\n"
            "fleet_run:\n"
            f"  lease_path: {json.dumps(str(tmp_path / 'campaign.lease'))}\n"
            "  lease_ttl_s: 60\n"
            "  disk_min_gib: 0.001\n"
            "  min_free_gib: 0.001\n"
            "  recorded_source: selection.json\n"
            "  recorded_opensky_source: recorded-opensky.json\n"
            "  acquisition_journal: journal.jsonl\n"
            "  acquisition_receipt_dir: receipts\n"
        )
        + SELECTED_PARAMS_YAML,
        encoding="utf-8",
    )
    return config


def _run_cli(config: Path, mode: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "node_fdm_pipeline",
        "fleet-campaign",
        mode,
        "--config",
        str(config),
    ]
    if mode in {"run", "resume"}:
        command.extend(["--only-step", "download"])
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        env=_offline_env(),
    )


def _listed_modes(output: str) -> set[str]:
    clean = _ANSI_ESCAPE.sub("", output)
    candidates = {
        token
        for line in clean.splitlines()
        if not line.lstrip().startswith("-")
        for token in re.findall(
            r"(?<![a-z-])([a-z]+(?:-[a-z]+)*)(?=\s{2,}|\s*│)",
            line,
        )
    }
    return candidates & (_EXPECTED_MODES | {"follow", "start", "stop"})


def _write_journalled_steps(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    for index, step in enumerate(("step-a", "step-b"), start=1):
        receipt = tmp_path / f"{step}.json"
        receipt.write_text(
            json.dumps(
                {
                    "duration": float(index),
                    "bytes": index * 100,
                    "rows": index * 10,
                    "flight_identities": [f"flight-{index}"],
                    "rejects": index,
                    "split_batches": index + 1,
                    "cache_hits": index + 2,
                    "cache_misses": index + 3,
                    "errors": [f"error-{index}"],
                    "caches": [f"cache-{index}"],
                    "day_journals": [],
                }
            ),
            encoding="utf-8",
        )
        events.append(
            {
                "acquisition_key": step,
                "state": "committed",
                "timestamp": f"2025-01-0{index}T00:00:00+00:00",
                "receipt": str(receipt),
            }
        )
    (tmp_path / "journal.jsonl").write_text(
        "".join(f"{json.dumps(event)}\n" for event in events),
        encoding="utf-8",
    )


def test_fleet_campaign_help_lists_exactly_five_modes() -> None:
    """AC1: group help exposes exactly the five public campaign modes."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "node_fdm_pipeline",
            "fleet-campaign",
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=_offline_env(),
    )

    assert result.returncode == 0
    assert _listed_modes(result.stdout + result.stderr) == _EXPECTED_MODES


def test_plan_output_matches_direct_contract_rendering(tmp_path: Path) -> None:
    """AC2: plan stdout is the direct contract object's rendering."""
    config = _write_campaign(tmp_path)
    expected = f"{cli._render_campaign_result(run_fleet_campaign(config, 'plan'))}\n"

    result = _run_cli(config, "plan")

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout == expected


def test_status_and_validate_render_every_operator_figure(tmp_path: Path) -> None:
    """AC3: status and validate emit one complete operator row per journalled step."""
    config = _write_campaign(tmp_path)
    _write_journalled_steps(tmp_path)

    for mode in ("status", "validate"):
        result = _run_cli(config, mode)

        assert result.returncode == 0
        rows = [
            line
            for line in result.stdout.splitlines()
            if any(step in line for step in ("step-a", "step-b"))
        ]
        assert len(rows) == 2
        for step in ("step-a", "step-b"):
            matching = [line for line in rows if step in line]
            assert len(matching) == 1
            assert all(figure in matching[0] for figure in _OPERATOR_FIGURES)


def test_resume_without_state_is_a_typed_refusal(tmp_path: Path) -> None:
    """AC4: a stateless resume refuses on stderr without appending a journal event."""
    config = _write_campaign(tmp_path)
    journal = tmp_path / "journal.jsonl"
    before = journal.read_bytes() if journal.exists() else None

    result = _run_cli(config, "resume")

    after = journal.read_bytes() if journal.exists() else None
    assert result.returncode != 0
    assert "no campaign identity recorded" in result.stderr.lower()
    assert after == before


def test_sigint_marks_the_in_flight_campaign_interrupted(tmp_path: Path) -> None:
    """AC5: SIGINT exits non-zero after durably marking the in-flight run interrupted."""
    many_days = [
        (date(2025, 1, 1) + timedelta(days=offset)).isoformat() for offset in range(2_000)
    ]
    config = _write_campaign(tmp_path, utc_days=many_days)
    journal = tmp_path / "journal.jsonl"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "node_fdm_pipeline",
            "fleet-campaign",
            "run",
            "--config",
            str(config),
            "--only-step",
            "download",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_offline_env(),
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if journal.exists() and '"state":"processing"' in journal.read_text(encoding="utf-8"):
                break
            if process.poll() is not None:
                break
            time.sleep(0.01)
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"campaign exited before SIGINT: stdout={stdout!r} stderr={stderr!r}")
        assert journal.exists()
        process.send_signal(signal.SIGINT)
        _stdout, _stderr = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)

    events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    assert process.returncode != 0
    assert events[-1]["acquisition_key"]
    assert any(event.get("state") == "interrupted" for event in events)
    assert events[-1]["state"] == "lease_released"
