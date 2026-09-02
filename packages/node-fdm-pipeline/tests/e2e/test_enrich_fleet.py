"""Black-box tests for the enrich-fleet command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_enrich_fleet_partial_campaign_options_create_no_lock(tmp_path: Path) -> None:
    """AC2: partial campaign options name every omission before creating any lock."""
    data_dir = tmp_path / "data"
    fleet_dir = data_dir / "fleet"
    fleet_dir.mkdir(parents=True)
    selection = tmp_path / "selection.txt"
    selection.write_text("selection-v1", encoding="utf-8")
    lease_path = tmp_path / "shared" / "enrich-fleet.lease"
    lease_path.parent.mkdir()
    fdm = Path(sys.executable).with_name("fdm")

    result = subprocess.run(
        [
            str(fdm),
            "enrich-fleet",
            "--fleet-dir",
            str(fleet_dir),
            "--data-root",
            str(data_dir),
            "--selection",
            str(selection),
            "--lease-path",
            str(lease_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    diagnostic = result.stderr.lower().replace("_", "-")
    assert result.returncode != 0
    for missing in ("resolved-config", "profile", "lease-ttl-s", "disk-min-gib"):
        assert missing in diagnostic
    assert not lease_path.exists()
    assert list(data_dir.rglob("*.lock")) == []
