"""Unit tests for date-major fleet weather planning."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _fleet_enrich
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigestMismatch,
    compute_resume_digest,
)
from node_fdm_pipeline.config import FleetRunConfig


def test_build_plan_is_date_major_and_uses_decoded_dates(
    mocker: MockerFixture,
) -> None:
    configs: dict[str, Any] = {}
    for name in ("B", "A"):
        paths = SimpleNamespace(
            resolve=lambda key, name=name: (
                Path(f"/data/{name}/flights.delta")
                if key == "delta_table"
                else Path("/data/era5_cache")
            )
        )
        configs[name] = SimpleNamespace(paths=paths, era5_features=["temperature"])

    mocker.patch(
        "node_fdm_pipeline.config.PipelineConfig.from_yaml",
        side_effect=lambda path, data_root=None: configs[path.stem],
    )

    def day_reader(path: Path) -> tuple[str, ...]:
        if "/A/" in str(path):
            return ("2024-01-02",)
        return ("2024-01-01", "2024-01-02")

    triples = [
        ("B", Path("B"), Path("selection_B.csv")),
        ("A", Path("A"), Path("selection_A.csv")),
    ]

    plan = _fleet_enrich.build_enrichment_plan(
        triples,
        start_date="2024-01-01",
        end_date="2024-01-03",
        day_reader=day_reader,
    )

    assert list(plan.days) == ["2024-01-01", "2024-01-02"]
    assert [cohort.name for cohort in plan.days["2024-01-02"]] == ["A", "B"]
    assert plan.assignments == 3


def test_build_plan_rejects_incompatible_weather_features(
    mocker: MockerFixture,
) -> None:
    def load(path: Path, data_root: Path | None = None) -> Any:
        del data_root
        paths = SimpleNamespace(
            resolve=lambda key: (
                Path(f"/data/{path.stem}/flights.delta")
                if key == "delta_table"
                else Path("/data/era5_cache")
            )
        )
        return SimpleNamespace(paths=paths, era5_features=[path.stem])

    mocker.patch("node_fdm_pipeline.config.PipelineConfig.from_yaml", side_effect=load)

    with pytest.raises(SystemExit, match="different ERA5 feature"):
        _fleet_enrich.build_enrichment_plan(
            [
                ("A", Path("A"), Path("selection_A.csv")),
                ("B", Path("B"), Path("selection_B.csv")),
            ],
            day_reader=lambda path: ("2024-01-01",),
        )


def test_enrich_fleet_rejects_mismatching_recorded_digest() -> None:
    """AC2: enrich rejects when the recorded profile digest differs from the current one."""
    selection_digest = "recorded-offline-selection"
    resolved_config: DigestInput = {"workers": 1, "mode": "recorded-offline"}
    profile: DigestInput = {"aircraft": "A320", "version": 1}
    current = compute_resume_digest(selection_digest, resolved_config, profile)
    recorded = current.model_copy(update={"profile": "recorded-profile-digest"})
    fleet_config = FleetRunConfig(
        lease_path=Path("/shared/enrich-fleet.lease"),
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )
    plan = _fleet_enrich.EnrichmentPlan(
        days={},
        cache_root=Path("/recorded-offline/era5-cache"),
        features=("temperature",),
    )

    with pytest.raises(ResumeDigestMismatch, match="profile changed"):
        _fleet_enrich.enrich_fleet(
            plan,
            fleet_config=fleet_config,
            recorded_digest=recorded,
            selection_digest=selection_digest,
            resolved_config=resolved_config,
            profile=profile,
        )
