"""Black-box tests for the decode-fleet command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_decode_fleet_prints_one_deprecation_notice(tmp_path: Path) -> None:
    """AC2: decode-fleet emits one notice naming fleet-campaign run."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    fdm = Path(sys.executable).with_name("fdm")

    completed = subprocess.run(
        [str(fdm), "decode-fleet", "--fleet-dir", str(fleet_dir), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    stderr = completed.stderr.lower()
    assert stderr.count("deprecated") == 1
    assert stderr.count("fdm fleet-campaign run") == 1


@pytest.mark.e2e
def test_decode_fleet_partial_campaign_options_create_no_lock(tmp_path: Path) -> None:
    """AC2: partial campaign options name every omission before creating any lock."""
    data_dir = tmp_path / "data"
    fleet_dir = data_dir / "fleet"
    fleet_dir.mkdir(parents=True)
    selection = tmp_path / "selection.txt"
    resolved_config = tmp_path / "resolved-config.json"
    lease_path = tmp_path / "shared" / "decode-fleet.lease"
    selection.write_text("selection-v1", encoding="utf-8")
    resolved_config.write_text('{"workers": 1}', encoding="utf-8")
    fdm = Path(sys.executable).with_name("fdm")

    result = subprocess.run(
        [
            str(fdm),
            "decode-fleet",
            "--fleet-dir",
            str(fleet_dir),
            "--selection",
            str(selection),
            "--resolved-config",
            str(resolved_config),
            "--lease-path",
            str(lease_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert all(name in result.stderr for name in ("profile", "lease_ttl_s", "disk_min_gib"))
    assert not lease_path.exists()
    assert not list(data_dir.rglob("*.lock"))
