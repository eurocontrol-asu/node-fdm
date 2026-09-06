from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Protocol, cast

import polars as pl
import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _day_runner as day_runner
from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._fleet_selection import SelectionPlan
from node_fdm_pipeline.commands._science_profile import ScienceProfile

type _RunnerStepHook = Callable[
    [Literal["stage", "validate", "publish", "day_committed"]],
    None,
]


class _OpenSkyReader(Protocol):
    def __call__(
        self,
        source_day: str,
        selection_ids: frozenset[str],
    ) -> pl.DataFrame: ...


pytestmark = pytest.mark.integration

_PUBLICATION_STEPS = {"stage_partition", "validate_partition", "publish_partition"}


def _plan(day: str, cohorts: tuple[str, ...]) -> DayPlan:
    keys = tuple(DayPartitionKey(cohort, day) for cohort in cohorts)
    return DayPlan(
        meta_selection_day=day,
        partition_keys=keys,
        selection_ids=frozenset(f"sel-{cohort}" for cohort in cohorts),
        selection_ids_by_key={key: frozenset({f"sel-{key.cohort}"}) for key in keys},
        source_days=(day,),
    )


def _frame(key: DayPartitionKey, value: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "cohort": [key.cohort],
            "meta_selection_day": [key.meta_selection_day],
            "meta_source_day": [key.meta_selection_day],
            "selection_id": [f"sel-{key.cohort}"],
            "msn": [f"msn-{key.cohort}"],
            "split": ["train"],
            "profile_id": ["test-profile"],
            "profile_digest": ["test-digest"],
            "code_version": ["test-code-v1"],
            "value": [value],
        }
    )


def _opened_grids() -> object:
    return SimpleNamespace(handles={}, opens={}, close=lambda: None)


def _install_common(mocker: MockerFixture, plan: DayPlan) -> None:
    mocker.patch.object(day_runner, "admit_day")
    mocker.patch.object(day_runner, "append_event")
    mocker.patch.object(day_runner, "build_day_plan", return_value=plan)
    mocker.patch.object(day_runner, "commit_day")
    mocker.patch.object(day_runner, "cleanup_day")


def _install_in_memory_pipeline(
    mocker: MockerFixture,
    frames: dict[DayPartitionKey, pl.DataFrame],
    assembled_plans: list[tuple[DayPartitionKey, ...]] | None = None,
) -> None:
    mocker.patch.object(
        day_runner,
        "_run_parent_acquisition",
        return_value=(pl.DataFrame(), _opened_grids()),
    )

    def assemble(
        _source: pl.DataFrame,
        received_plan: DayPlan,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[DayPartitionKey, pl.DataFrame], tuple[int, ...]]:
        if assembled_plans is not None:
            assembled_plans.append(tuple(received_plan.partition_keys))
        return (
            {key: frames[key] for key in received_plan.partition_keys},
            (101,),
        )

    mocker.patch.object(day_runner, "_assemble_partitions", side_effect=assemble)


def _run(
    plan: DayPlan,
    root: Path,
    *,
    step_hook: (Callable[[str], None] | Callable[[str, DayPartitionKey], None] | None) = None,
    opensky_reader: _OpenSkyReader | None = None,
    grid_opener: Callable[[str, str], object] | None = None,
) -> day_runner.DayRunReport:
    lease = SimpleNamespace(
        acquire=lambda: None,
        heartbeat=lambda: None,
        assert_owned=lambda: None,
        release=lambda: None,
    )
    return day_runner.run_selection_day(
        plan.meta_selection_day,
        selection=SelectionPlan(flights=(), digest="test-selection"),
        opensky_reader=opensky_reader or (lambda _day, _ids: pl.DataFrame()),
        grid_opener=grid_opener or (lambda _day, _field: SimpleNamespace(interpolate=True)),
        grid_closer=lambda _handle: None,
        era5_fields=("temperature",),
        lease=lease,
        staging_root=root / "staging",
        published_root=root / "published",
        journal_path=root / "journal.jsonl",
        counters_path=root / "counters.json",
        cleanup_artifacts={},
        cleanup_decrements=(),
        science_profile=ScienceProfile(
            profile_id="test-profile",
            source_commit="test-commit",
            artifact_digests={
                "retained.csv": "retained-digest",
                "grids.json": "grids-digest",
                "metrics.yaml": "metrics-digest",
            },
        ),
        versions={"node-fdm-pipeline": "test-code-v1"},
        local_workers=1,
        max_resident_gib=2.0,
        step_hook=cast("_RunnerStepHook | None", step_hook),
    )


def _fail_after(
    step: str,
    target: DayPartitionKey,
) -> Callable[[str, DayPartitionKey], None]:
    def hook(completed_step: str, key: DayPartitionKey) -> None:
        if completed_step == f"{step}_partition" and key == target:
            raise RuntimeError(f"injected fault after {step}")

    return hook


def _events(root: Path) -> list[dict[str, object]]:
    return read_events(root / "journal.jsonl")


def _steps_for(root: Path, key: DayPartitionKey) -> tuple[str, ...]:
    return tuple(
        str(event["event"])
        for event in _events(root)
        if event.get("event") in _PUBLICATION_STEPS
        and event.get("cohort") == key.cohort
        and event.get("meta_selection_day") == key.meta_selection_day
    )


def _published_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted((root / "published").rglob("*.parquet")))


def _digests(paths: tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)


def test_resume_after_stage_records_each_boundary_once(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC2: an independent resume after stage records each durable boundary exactly once."""
    plan = _plan("20200101", ("P",))
    key = plan.partition_keys[0]
    _install_common(mocker, plan)
    _install_in_memory_pipeline(mocker, {key: _frame(key)})

    with pytest.raises(RuntimeError, match="after stage"):
        _run(plan, tmp_path, step_hook=_fail_after("stage", key))
    _run(plan, tmp_path)

    assert Counter(_steps_for(tmp_path, key)) == Counter(
        {"stage_partition": 1, "validate_partition": 1, "publish_partition": 1}
    )


def test_resume_after_stage_uses_staged_payload_without_source_io(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC3: a staged payload resumes after its raw OpenSky and ERA5 inputs are deleted."""
    plan = _plan("20200101", ("P",))
    key = plan.partition_keys[0]
    root = tmp_path / "resume"
    raw_path = root / "inputs" / "opensky.parquet"
    grid_path = root / "inputs" / "era5.grid"
    raw_path.parent.mkdir(parents=True)
    _frame(key).write_parquet(raw_path)
    grid_path.write_bytes(b"local-era5-grid")
    _install_common(mocker, plan)

    def read_opensky(source_day: str, selection_ids: frozenset[str]) -> pl.DataFrame:
        return pl.read_parquet(raw_path)

    def open_grid(_source_day: str, _field: str) -> object:
        grid_path.read_bytes()
        return SimpleNamespace(interpolate=True)

    mocker.patch.object(
        day_runner,
        "_assemble_partitions",
        return_value=({key: _frame(key)}, (101,)),
    )

    with pytest.raises(RuntimeError, match="after stage"):
        _run(
            plan,
            root,
            step_hook=_fail_after("stage", key),
            opensky_reader=read_opensky,
            grid_opener=open_grid,
        )
    raw_path.unlink()
    grid_path.unlink()

    _run(
        plan,
        root,
        opensky_reader=read_opensky,
        grid_opener=open_grid,
    )

    assert len(_published_files(root)) == 1


def test_resume_after_validate_only_publishes(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC4: a validated partition resumes with publication and no repeated boundary."""
    plan = _plan("20200101", ("Q",))
    key = plan.partition_keys[0]
    _install_common(mocker, plan)
    _install_in_memory_pipeline(mocker, {key: _frame(key)})

    with pytest.raises(RuntimeError, match="after validate"):
        _run(plan, tmp_path, step_hook=_fail_after("validate", key))
    _run(plan, tmp_path)

    assert Counter(_steps_for(tmp_path, key)) == Counter(
        {"stage_partition": 1, "validate_partition": 1, "publish_partition": 1}
    )


def test_resume_after_publish_skips_completed_partition(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC5: a published partition is skipped while another pending partition completes."""
    plan = _plan("20200101", ("R", "S"))
    published_key, pending_key = plan.partition_keys
    assembled_plans: list[tuple[DayPartitionKey, ...]] = []
    frames = {key: _frame(key) for key in plan.partition_keys}
    _install_common(mocker, plan)
    _install_in_memory_pipeline(mocker, frames, assembled_plans)

    with pytest.raises(RuntimeError, match="after publish"):
        _run(
            plan,
            tmp_path,
            step_hook=_fail_after("publish", published_key),
        )
    _run(plan, tmp_path)

    assert Counter(_steps_for(tmp_path, published_key)) == Counter(
        {"stage_partition": 1, "validate_partition": 1, "publish_partition": 1}
    )
    assert Counter(_steps_for(tmp_path, pending_key)) == Counter(
        {"stage_partition": 1, "validate_partition": 1, "publish_partition": 1}
    )
    assert assembled_plans[-1] == (pending_key,)


def test_resumed_publication_matches_fault_free_bytes_and_order(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC6: resumed publication has the reference run's bytes and publication order."""
    plan = _plan("20200101", ("P", "S"))
    first_key, second_key = plan.partition_keys
    _install_common(mocker, plan)
    mocker.patch.object(
        day_runner,
        "_run_parent_acquisition",
        return_value=(pl.DataFrame(), _opened_grids()),
    )
    assembly_call = 0

    def assemble(
        _source: pl.DataFrame,
        received_plan: DayPlan,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[DayPartitionKey, pl.DataFrame], tuple[int, ...]]:
        nonlocal assembly_call
        assembly_call += 1
        changed_first = assembly_call >= 3
        frames = {
            first_key: _frame(first_key, value=999 if changed_first else 1),
            second_key: _frame(second_key, value=2),
        }
        return (
            {key: frames[key] for key in received_plan.partition_keys},
            (101,),
        )

    mocker.patch.object(day_runner, "_assemble_partitions", side_effect=assemble)
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"

    _run(plan, reference_root)
    with pytest.raises(RuntimeError, match="after stage"):
        _run(
            plan,
            resumed_root,
            step_hook=_fail_after("stage", first_key),
        )
    _run(plan, resumed_root)

    reference_paths = _published_files(reference_root)
    resumed_paths = _published_files(resumed_root)
    reference_order = tuple(
        event["cohort"]
        for event in _events(reference_root)
        if event.get("event") == "publish_partition"
    )
    resumed_order = tuple(
        event["cohort"]
        for event in _events(resumed_root)
        if event.get("event") == "publish_partition"
    )
    assert _digests(resumed_paths) == _digests(reference_paths)
    assert resumed_order == reference_order
