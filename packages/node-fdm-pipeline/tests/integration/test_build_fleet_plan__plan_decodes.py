"""Integration coverage for shared ownership in fleet decode planning."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from node_fdm_pipeline.commands._fleet_decode import plan_decodes
from node_fdm_pipeline.commands._fleet_plan import build_fleet_plan, discover_cohorts


@pytest.mark.integration
def test_plan_decodes_emits_a_job_for_every_owner_of_a_shared_aircraft(
    tmp_path: Path, make_config: Callable[..., Path]
) -> None:
    """AC3: each cohort owning a shared aircraft receives its own decode job."""
    fleet_dir = tmp_path / "fleet"
    for name in ("C1", "C2"):
        cohort_dir = fleet_dir / "types" / name
        results_dir = cohort_dir / "results"
        results_dir.mkdir(parents=True)
        make_config(cohort_dir / "config.yaml", tmp_path / "data" / name)
        pl.DataFrame({"icao24": ["a00001"], "day": ["2019-12-31"]}).write_csv(
            results_dir / f"selection_{name}.csv"
        )

    plan = build_fleet_plan(discover_cohorts(fleet_dir))
    jobs = plan_decodes(fleet_dir)

    assert tuple(cohort.name for cohort in plan.owner["a00001"]) == ("C1", "C2")
    assert [(job.name, job.start_date, job.end_date, job.days) for job in jobs] == [
        ("C1", "2019-12-31", "2020-01-01", 1),
        ("C2", "2019-12-31", "2020-01-01", 1),
    ]
