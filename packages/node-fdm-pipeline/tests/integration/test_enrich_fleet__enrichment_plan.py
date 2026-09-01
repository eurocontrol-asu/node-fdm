"""Integration tests for resumable day-scoped ERA5 cache lifecycle."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import polars as pl
import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _fleet_enrich
from node_fdm_pipeline.commands.data import EnrichOutcome, enrich_with_grid
from node_fdm_pipeline.config import PipelineConfig

pytestmark = pytest.mark.integration

_ERA_VALUES = {
    "era_temp_K": [280.0, 281.0],
    "era_u_wind_ms": [1.0, 2.0],
    "era_v_wind_ms": [2.0, 3.0],
    "era_tas_kt": [300.0, 301.0],
    "era_mach": [0.7, 0.71],
    "era_cas_kt": [250.0, 251.0],
}
_INPUT_VALUES = {
    "raw_timestamp": [
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 2, tzinfo=UTC),
    ],
    "raw_lat_deg": [48.0, 49.0],
    "raw_lon_deg": [2.0, 3.0],
    "raw_alt_ft": [30_000.0, 31_000.0],
    "raw_gs_kt": [400.0, 410.0],
    "raw_track_deg": [90.0, 91.0],
}


def _cohort(name: str, root: Path) -> _fleet_enrich.EnrichmentCohort:
    paths = SimpleNamespace(resolve=lambda key: root / name / "flights.delta")
    cfg = cast(PipelineConfig, SimpleNamespace(paths=paths, era5_null_threshold=0.05))
    return _fleet_enrich.EnrichmentCohort(name=name, cfg=cfg)


def test_plan_and_status_read_actual_delta_partitions(
    tmp_path: Path, make_config: Callable[..., Path]
) -> None:
    data_dir = tmp_path / "A"
    table = data_dir / "flights.delta"
    config = make_config(tmp_path / "config.yaml", data_dir)
    frame = pl.DataFrame(
        {
            "meta_batch_date": ["20240101", "20240102"],
            **_INPUT_VALUES,
            **_ERA_VALUES,
        }
    )
    frame.write_delta(
        str(table),
        mode="overwrite",
        delta_write_options={"partition_by": ["meta_batch_date"]},
    )

    plan = _fleet_enrich.build_enrichment_plan([("A", config, tmp_path / "selection_A.csv")])
    complete, rows, null_fraction = _fleet_enrich.weather_status(
        plan.days["2024-01-01"][0], "2024-01-01"
    )

    assert list(plan.days) == ["2024-01-01", "2024-01-02"]
    assert complete is True
    assert rows == 1
    assert null_fraction == 0.0


def test_enrich_with_grid_reads_and_overwrites_only_one_delta_day(
    tmp_path: Path, make_config: Callable[..., Path], mocker: MockerFixture
) -> None:
    data_dir = tmp_path / "A"
    table = data_dir / "flights.delta"
    config = make_config(tmp_path / "config.yaml", data_dir)
    source = pl.DataFrame(
        {
            "meta_batch_date": ["20240101", "20240102"],
            "raw_marker": [1, 2],
            **_INPUT_VALUES,
        }
    )
    source.write_delta(
        str(table),
        mode="overwrite",
        delta_write_options={"partition_by": ["meta_batch_date"]},
    )

    def fake_enrich(frame: pl.DataFrame, grid: object) -> pl.DataFrame:
        del grid
        assert frame["meta_batch_date"].to_list() == ["20240101"]
        return frame.with_columns(
            pl.lit(280.0).alias("era_temp_K"),
            pl.lit(1.0).alias("era_u_wind_ms"),
            pl.lit(2.0).alias("era_v_wind_ms"),
            pl.lit(300.0).alias("era_tas_kt"),
            pl.lit(0.7).alias("era_mach"),
            pl.lit(250.0).alias("era_cas_kt"),
        )

    mocker.patch("node_fdm_data.meteo.enrich_era5", side_effect=fake_enrich)
    cfg = PipelineConfig.from_yaml(config)

    outcome = enrich_with_grid(
        cfg,
        object(),
        start_date="2024-01-01",
        end_date="2024-01-02",
    )
    persisted = pl.read_delta(str(table)).sort("meta_batch_date")

    assert outcome.rows == 1
    assert persisted["raw_marker"].to_list() == [1, 2]
    assert persisted["era_temp_K"].to_list() == [280.0, None]


def test_enrich_fleet_shares_then_purges_one_cache_per_day(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    completed: set[tuple[str, str]] = set()
    providers: list[tuple[Path, object]] = []
    cache_root = tmp_path / "era5_cache"
    cohorts = (_cohort("A", tmp_path), _cohort("B", tmp_path))
    plan = _fleet_enrich.EnrichmentPlan(
        days={"2024-01-01": cohorts, "2024-01-02": cohorts},
        cache_root=cache_root,
        features=("temperature",),
    )

    def status(cohort: Any, day: str) -> tuple[bool, int, float]:
        return ((cohort.name, day) in completed, 12, 0.0)

    def enrich(cfg: Any, provider: object, *, start_date: str, end_date: str) -> EnrichOutcome:
        del provider
        del end_date
        name = next(cohort.name for cohort in cohorts if cohort.cfg is cfg)
        completed.add((name, start_date))
        return EnrichOutcome(rows=12, era_columns=(), max_null_fraction=0.0)

    def provider_factory(path: Path, features: tuple[str, ...]) -> object:
        assert features == ("temperature",)
        (path / "zarr.json").write_text("weather")
        provider = object()
        providers.append((path, provider))
        return provider

    enrich_mock = mocker.Mock(side_effect=enrich)
    runtime = _fleet_enrich.EnrichmentRuntime(
        provider_factory=provider_factory,
        status_checker=status,
        enrich_function=enrich_mock,
    )
    manifest = tmp_path / "enrich-fleet.manifest.jsonl"

    outcomes = _fleet_enrich.enrich_fleet(
        plan,
        manifest_path=manifest,
        runtime=runtime,
    )

    assert len(providers) == 2
    assert enrich_mock.call_count == 4
    assert all(outcome.cache_purged for outcome in outcomes)
    assert not cache_root.joinpath("date=2024-01-01").exists()
    assert not cache_root.joinpath("date=2024-01-02").exists()
    events = [json.loads(line)["event"] for line in manifest.read_text().splitlines()]
    assert events.count("date_finished") == 2
    assert events[-1] == "run_finished"


def test_failure_retains_cache_and_stops_before_next_day(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    cache_root = tmp_path / "era5_cache"
    cohort = _cohort("A", tmp_path)
    plan = _fleet_enrich.EnrichmentPlan(
        days={"2024-01-01": (cohort,), "2024-01-02": (cohort,)},
        cache_root=cache_root,
        features=(),
    )
    status = mocker.Mock(return_value=(False, 0, 0.0))
    enrich = mocker.Mock(side_effect=RuntimeError("boom"))

    def provider_factory(path: Path, features: tuple[str, ...]) -> object:
        del features
        (path / "zarr.json").write_text("weather")
        return object()

    runtime = _fleet_enrich.EnrichmentRuntime(
        provider_factory=provider_factory,
        status_checker=status,
        enrich_function=enrich,
    )
    outcomes = _fleet_enrich.enrich_fleet(plan, runtime=runtime)

    assert [outcome.day for outcome in outcomes] == ["2024-01-01"]
    assert outcomes[0].error == "RuntimeError: boom"
    assert cache_root.joinpath("date=2024-01-01", "zarr.json").exists()
    assert not cache_root.joinpath("date=2024-01-02").exists()


def test_resume_skips_durable_cohorts_and_purges_stale_day_cache(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    cache_root = tmp_path / "era5_cache"
    stale = cache_root / "date=2024-01-01"
    stale.mkdir(parents=True)
    stale.joinpath("zarr.json").write_text("weather")
    cohort = _cohort("A", tmp_path)
    plan = _fleet_enrich.EnrichmentPlan(
        days={"2024-01-01": (cohort,)},
        cache_root=cache_root,
        features=(),
    )
    status = mocker.Mock(return_value=(True, 12, 0.0))
    factory = mocker.Mock()

    runtime = _fleet_enrich.EnrichmentRuntime(
        provider_factory=factory,
        status_checker=status,
        enrich_function=mocker.Mock(),
    )
    outcomes = _fleet_enrich.enrich_fleet(plan, runtime=runtime)

    factory.assert_not_called()
    assert outcomes[0].skipped == 1
    assert outcomes[0].cache_purged is True
    assert not stale.exists()
