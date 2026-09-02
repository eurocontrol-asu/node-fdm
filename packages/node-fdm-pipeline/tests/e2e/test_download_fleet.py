"""End-to-end coverage for the download-fleet command."""

from __future__ import annotations

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
