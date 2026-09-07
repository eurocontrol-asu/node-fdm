"""Black-box tests for the enrich-fleet command."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_enrich_fleet_prints_one_deprecation_notice(tmp_path: Path) -> None:
    """AC2: enrich-fleet emits one notice naming fleet-campaign run."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    fdm = Path(sys.executable).with_name("fdm")

    completed = subprocess.run(
        [str(fdm), "enrich-fleet", "--fleet-dir", str(fleet_dir), "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    stderr = completed.stderr.lower()
    assert stderr.count("deprecated") == 1
    assert stderr.count("fdm fleet-campaign run") == 1


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


@pytest.mark.e2e
def test_enrich_fleet_cli_rejects_mutated_campaign_identity(tmp_path: Path) -> None:
    """AC5: the CLI reports a resume mismatch before a second weather-provider call."""
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    shared = campaign_root / "shared"
    shared.mkdir()
    selection = tmp_path / "selection.txt"
    resolved_config = tmp_path / "resolved-config.json"
    profile = tmp_path / "profile.json"
    selection.write_text("selection-a", encoding="utf-8")
    resolved_config.write_text('{"workers": 1}', encoding="utf-8")
    profile.write_text('{"aircraft": "A320"}', encoding="utf-8")
    counter_path = tmp_path / "weather-calls.txt"
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
                from __future__ import annotations

                import os
                from pathlib import Path
                from types import SimpleNamespace

                from node_fdm_pipeline.commands import _fleet_enrich
                from node_fdm_pipeline.commands.data import EnrichOutcome

                _complete = False

                def _plan(*_args, **_kwargs):
                    root = Path(os.environ["FDM_TEST_CAMPAIGN_ROOT"])
                    cohort = _fleet_enrich.EnrichmentCohort(
                        name="local",
                        cfg=SimpleNamespace(),
                    )
                    return _fleet_enrich.EnrichmentPlan(
                        days={"2025-01-01": (cohort,)},
                        cache_root=root / "era5-cache",
                        features=("temperature",),
                    )

                def _provider(*_args, **_kwargs):
                    global _complete
                    counter = Path(os.environ["FDM_TEST_WEATHER_COUNTER"])
                    calls = int(counter.read_text()) if counter.exists() else 0
                    counter.write_text(str(calls + 1))
                    _complete = True
                    return SimpleNamespace(close=lambda: None)

                def _status(*_args, **_kwargs):
                    return _complete, int(_complete), 0.0

                def _enrich(*_args, **_kwargs):
                    return EnrichOutcome(
                        rows=1,
                        era_columns=("temperature",),
                        max_null_fraction=0.0,
                    )

                _fleet_enrich.build_enrichment_plan = _plan
                _fleet_enrich._default_provider = _provider
                _fleet_enrich.weather_status = _status
                _fleet_enrich.enrich_with_grid = _enrich
                """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(hooks), env.get("PYTHONPATH", "")) if part
    )
    env["FDM_TEST_CAMPAIGN_ROOT"] = str(campaign_root)
    env["FDM_TEST_WEATHER_COUNTER"] = str(counter_path)
    fdm = Path(sys.executable).with_name("fdm")
    command = [
        str(fdm),
        "enrich-fleet",
        "--fleet-dir",
        str(campaign_root),
        "--data-root",
        str(campaign_root),
        "--selection",
        str(selection),
        "--resolved-config",
        str(resolved_config),
        "--profile",
        str(profile),
        "--lease-path",
        str(shared / "enrich-fleet.lease"),
        "--lease-ttl-s",
        "60",
        "--disk-min-gib",
        "1",
    ]

    first = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert first.returncode == 0
    assert counter_path.read_text(encoding="utf-8") == "1"

    profile.write_text('{"aircraft": "A321"}', encoding="utf-8")
    second = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert second.returncode != 0
    assert "resume digest mismatch" in second.stderr.lower()
    assert counter_path.read_text(encoding="utf-8") == "1"
