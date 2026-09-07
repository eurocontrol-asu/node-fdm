"""Integration coverage for restricted fleet-campaign execution and legacy delegation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline import cli, run_fleet_campaign
from node_fdm_pipeline.commands import fleet_campaign

pytestmark = pytest.mark.integration

_SELECTION_CSV = """selection_id,icao24,callsign,start,end,cohorts,utc_days
flight-a,abc123,AXM1,1704153300,1704153900,alpha,2024-01-01
"""


@pytest.fixture
def campaign_config(
    tmp_path: Path,
    make_config: Callable[..., Path],
) -> tuple[Path, Path]:
    selection_path = tmp_path / "recorded-selection.csv"
    selection_path.write_text(_SELECTION_CSV, encoding="utf-8")
    journal_path = tmp_path / "campaign.journal.jsonl"
    config_path = make_config(
        tmp_path / "campaign.yaml",
        tmp_path / "data",
        extra=f"""
fleet_run:
  lease_path: "{tmp_path / "campaign.lease"}"
  lease_ttl_s: 60
  disk_min_gib: 0.000001
  min_free_gib: 0.0
  recorded_source: "{selection_path}"
  acquisition_journal: "{journal_path}"
  acquisition_receipt_dir: "{tmp_path / "receipts"}"
""",
    )
    return config_path, journal_path


def _named_steps(value: object) -> set[str]:
    if isinstance(value, dict):
        named = {child for key, child in value.items() if key == "step" and isinstance(child, str)}
        return named.union(*(_named_steps(child) for child in value.values()))
    if isinstance(value, list | tuple):
        return set().union(*(_named_steps(child) for child in value))
    return set()


def test_only_steps_restricts_run_to_named_steps(
    campaign_config: tuple[Path, Path],
) -> None:
    """AC1: only_steps runs decode rows and records no download journal step."""
    config_path, journal_path = campaign_config

    report = run_fleet_campaign(config_path, "run", only_steps=["decode"])

    report_steps = _named_steps(report.model_dump(mode="json"))
    events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert report_steps == {"decode"}
    assert _named_steps(events) == {"decode"}


def test_unknown_step_name_is_refused(
    campaign_config: tuple[Path, Path],
) -> None:
    """AC1: UnknownCampaignStep names every accepted campaign step."""
    config_path, _journal_path = campaign_config

    with pytest.raises(fleet_campaign.UnknownCampaignStep) as caught:
        fleet_campaign.run_fleet_campaign(
            config_path,
            "run",
            only_steps=["segments"],
        )

    diagnostic = str(caught.value)
    assert all(step in diagnostic for step in ("download", "decode", "enrich"))


def _write_legacy_fleet(
    root: Path,
    make_config: Callable[..., Path],
) -> Path:
    cohort = root / "types" / "A320"
    results = cohort / "results"
    results.mkdir(parents=True)
    (results / "selection_A320.csv").write_text(
        "icao24,day\nabc123,2024-01-01\n",
        encoding="utf-8",
    )
    (cohort / "recorded-source.json").write_text(
        '{"rows": [{"icao24": "abc123"}]}',
        encoding="utf-8",
    )
    return make_config(cohort / "config.yaml", root / "data")


def test_legacy_download_matches_direct_campaign_contract(
    tmp_path: Path,
    make_config: Callable[..., Path],
    mocker: MockerFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC3: legacy download renders and journals exactly like only_steps download."""
    direct_root = tmp_path / "direct"
    legacy_root = tmp_path / "legacy"
    direct_config = _write_legacy_fleet(direct_root, make_config)
    _write_legacy_fleet(legacy_root, make_config)

    def recorded_contract(
        config: Path,
        mode: str,
        *,
        only_steps: Sequence[str] | None = None,
    ) -> str:
        source = (config.parent / "recorded-source.json").read_text(encoding="utf-8")
        steps = tuple(only_steps or ())
        event = {"mode": mode, "source": json.loads(source), "step": steps[0]}
        (config.parent / "campaign.journal.jsonl").write_text(
            json.dumps(event, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return f"mode={mode} step={steps[0]} rows=1"

    mocker.patch.object(fleet_campaign, "run_fleet_campaign", side_effect=recorded_contract)
    mocker.patch.object(cli, "run_fleet_campaign", side_effect=recorded_contract)

    direct_result = fleet_campaign.run_fleet_campaign(
        direct_config,
        "run",
        only_steps=["download"],
    )
    direct_rendering = f"{direct_result}\n"
    capsys.readouterr()

    cli.download_fleet(
        fleet_dir=legacy_root,
        workers=1,
        data_root=None,
        dry_run=True,
        force_refresh=False,
    )

    legacy_output = capsys.readouterr()
    direct_events = (direct_config.parent / "campaign.journal.jsonl").read_text(encoding="utf-8")
    legacy_events = (legacy_root / "types" / "A320" / "campaign.journal.jsonl").read_text(
        encoding="utf-8"
    )
    assert legacy_output.out == direct_rendering
    assert legacy_events == direct_events
