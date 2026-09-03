"""Integration coverage for shared campaign staging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from _config_fixtures import write_config
from node_fdm_pipeline.commands import _fleet_fetch, _raw_cache
from node_fdm_pipeline.commands._fleet_fetch import download_fleet
from node_fdm_pipeline.commands._fleet_plan import FleetPlan, build_fleet_plan
from node_fdm_pipeline.commands._fleet_selection import compile_selection


class RecordedOpenSky:
    """In-process source replaying the selected alpha rotation."""

    def history(self, *_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame({"icao24": ["a00001"], "timestamp": [1577754000]})

    def extended(self, *_args: object, **_kwargs: object) -> None:
        return None

    def flightlist(self, *_args: object, **_kwargs: object) -> None:
        return None


def _selection_rows() -> list[dict[str, Any]]:
    common: dict[str, Any] = {
        "icao24": "a00001",
        "callsign": "ALPHA",
        "firstseen": 1577750400,
        "lastseen": 1577754000,
        "msn": "alpha",
        "split": "train",
        "selection_id": "alpha",
    }
    return [{**common, "cohort": cohort} for cohort in ("C1", "C2")]


@pytest.fixture
def staged_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FleetPlan, Path]:
    staging_root = tmp_path / "campaign-staging"
    triples: list[tuple[str, Path, Path]] = []
    for cohort in ("C1", "C2"):
        config_path = tmp_path / f"{cohort}.yaml"
        write_config(config_path, tmp_path / cohort)
        triples.append((cohort, config_path, tmp_path / f"{cohort}-selection.csv"))

    selection = compile_selection(_selection_rows())
    plan = build_fleet_plan(triples, data_root=staging_root, selection=selection)
    source = RecordedOpenSky()
    monkeypatch.setattr(_fleet_fetch, "_get_opensky", lambda: source)

    download_fleet(plan)

    return plan, staging_root


@pytest.mark.integration
def test_shared_flight_is_staged_once_outside_cohort_raw_directories(
    staged_campaign: tuple[FleetPlan, Path],
) -> None:
    """AC1: both consumers share one staged history payload outside their raw silos."""
    plan, _staging_root = staged_campaign
    roots = {_raw_cache.cache_root(cohort.cfg, "history") for cohort in plan.cohorts}

    assert len(roots) == 1
    shared_root = roots.pop()
    assert [path.name for path in shared_root.rglob("data.parquet")] == ["data.parquet"]
    assert all(not (Path(cohort.cfg.paths.data_dir) / "raw").exists() for cohort in plan.cohorts)


@pytest.mark.integration
def test_staged_day_records_every_consumer_as_pending(
    staged_campaign: tuple[FleetPlan, Path],
) -> None:
    """AC2: the staged history day ledger reloads both consumers as pending."""
    plan, _staging_root = staged_campaign
    day_root = _raw_cache.cache_root(plan.cohorts[0].cfg, "history") / "date=20191231"
    ledgers = list(day_root.rglob("*.json"))

    assert len(ledgers) == 1
    assert json.loads(ledgers[0].read_text(encoding="utf-8")) == {
        "C1": "pending",
        "C2": "pending",
    }
