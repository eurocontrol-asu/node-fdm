"""End-to-end coverage for the download-fleet command."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
