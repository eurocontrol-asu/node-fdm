from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from pytest_mock import MockerFixture


def test_day_commit_pending_requires_every_partition_published() -> None:
    """AC1: commit is pending only after every planned partition is published."""
    day_runner = importlib.import_module("node_fdm_pipeline.commands._day_runner")
    day_plan = importlib.import_module("node_fdm_pipeline.commands._day_plan")
    keys = tuple(
        day_plan.DayPartitionKey(cohort, "20200101")
        for cohort in ("A319", "A320", "A321", "A332", "A359")
    )
    plan = day_plan.DayPlan(
        meta_selection_day="20200101",
        partition_keys=keys,
        selection_ids=frozenset(f"selection-{key.cohort}" for key in keys),
        selection_ids_by_key={key: frozenset({f"selection-{key.cohort}"}) for key in keys},
        source_days=("20200101",),
    )
    published = dict.fromkeys(keys, "published")
    staged = {**published, keys[-1]: "staged"}
    pending = {**published, keys[-1]: "pending"}

    assert day_runner.day_commit_pending(plan, published) is True
    assert day_runner.day_commit_pending(plan, staged) is False
    assert day_runner.day_commit_pending(plan, pending) is False


class _CapturedSlice:
    key: object


def test_every_cohort_slice_receives_parent_grid_handle(
    mocker: MockerFixture,
) -> None:
    """AC2: every cohort slice receives the one handle opened for its source day."""
    day_runner = importlib.import_module("node_fdm_pipeline.commands._day_runner")
    day_plan = importlib.import_module("node_fdm_pipeline.commands._day_plan")
    keys = (
        day_plan.DayPartitionKey("A20N", "20200101"),
        day_plan.DayPartitionKey("B738", "20200101"),
    )
    plan = day_plan.DayPlan(
        meta_selection_day="20200101",
        partition_keys=keys,
        selection_ids=frozenset({"sel-a20n", "sel-b738"}),
        selection_ids_by_key={
            keys[0]: frozenset({"sel-a20n"}),
            keys[1]: frozenset({"sel-b738"}),
        },
        source_days=("20200101",),
    )
    handle = object()
    captured_slices: list[object] = []

    def run_inline(*, slices: Iterable[_CapturedSlice], budget: object, fn: object) -> object:
        del budget, fn
        materialized: tuple[_CapturedSlice, ...] = tuple(slices)
        captured_slices.extend(materialized)
        return SimpleNamespace(
            results=tuple(
                day_runner._AssemblyResult(key=slice_.key, frame=pl.DataFrame(), pid=42)
                for slice_ in materialized
            )
        )

    def contains_handle(value: object) -> bool:
        if value is handle:
            return True
        if isinstance(value, dict):
            return any(contains_handle(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_handle(item) for item in value)
        return False

    mocker.patch.object(day_runner, "run_bounded_slices", side_effect=run_inline)
    day_runner._assemble_partitions(
        pl.DataFrame(),
        plan,
        day_runner.ScienceProfile(
            profile_id="test-profile",
            source_commit="test-commit",
            artifact_digests={},
        ),
        {"node-fdm-pipeline": "test-code-v1"},
        2,
        8.0,
        grid_handles={"20200101": handle},
    )

    assert len(captured_slices) == 2
    assert all(contains_handle(vars(slice_)) for slice_ in captured_slices)


def test_previous_cleanup_failure_blocks_day_admission() -> None:
    """AC1: the last failed cleanup state names the predecessor that blocks admission."""
    day_runner = importlib.import_module("node_fdm_pipeline.commands._day_runner")
    snapshot = (
        {"event": "cleanup_completed", "day": "20200101"},
        {"event": "cleanup_failed", "day": "20200101", "error": "injected cleanup fault"},
    )

    with pytest.raises(day_runner.PreviousDayCleanupFailed) as exc_info:
        day_runner.assert_day_admissible(snapshot, "20200102")

    assert "20200101" in str(exc_info.value)

    def test_every_cohort_slice_receives_parent_grid_handle(
        mocker: MockerFixture,
    ) -> None:
        """AC2: every cohort slice receives the one handle opened for its source day."""
        day_runner = importlib.import_module("node_fdm_pipeline.commands._day_runner")
        day_plan = importlib.import_module("node_fdm_pipeline.commands._day_plan")
        keys = (
            day_plan.DayPartitionKey("A20N", "20200101"),
            day_plan.DayPartitionKey("B738", "20200101"),
        )
        plan = day_plan.DayPlan(
            meta_selection_day="20200101",
            partition_keys=keys,
            selection_ids=frozenset({"sel-a20n", "sel-b738"}),
            selection_ids_by_key={
                keys[0]: frozenset({"sel-a20n"}),
                keys[1]: frozenset({"sel-b738"}),
            },
            source_days=("20200101",),
        )
        handle = object()
        open_calls: list[tuple[str, str]] = []
        captured_slices: list[object] = []

        def open_grid(source_day: str, field_name: str) -> object:
            open_calls.append((source_day, field_name))
            return handle

        def run_inline(*, slices: Iterable[_CapturedSlice], budget: object, fn: object) -> object:
            del budget, fn
            materialized: tuple[_CapturedSlice, ...] = tuple(slices)
            captured_slices.extend(materialized)
            return SimpleNamespace(
                results=tuple(
                    day_runner._AssemblyResult(key=slice_.key, frame=pl.DataFrame(), pid=42)
                    for slice_ in materialized
                )
            )

        def contains_handle(value: object) -> bool:
            if value is handle:
                return True
            if isinstance(value, dict):
                return any(contains_handle(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(contains_handle(item) for item in value)
            return False

        mocker.patch.object(day_runner, "admit_day")
        mocker.patch.object(day_runner, "append_event")
        mocker.patch.object(day_runner, "build_day_plan", return_value=plan)
        mocker.patch.object(day_runner, "run_bounded_slices", side_effect=run_inline)
        mocker.patch.object(day_runner, "publish_day")
        mocker.patch.object(day_runner, "commit_day")
        mocker.patch.object(day_runner, "cleanup_day")
        lease = SimpleNamespace(
            acquire=lambda: None,
            heartbeat=lambda: None,
            assert_owned=lambda: None,
            release=lambda: None,
        )

        day_runner.run_selection_day(
            "20200101",
            selection=mocker.sentinel.selection,
            opensky_reader=lambda _day, _ids: pl.DataFrame(),
            grid_opener=open_grid,
            grid_closer=lambda _handle: None,
            era5_fields=(
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
            ),
            lease=lease,
            staging_root=Path("."),
            published_root=Path("."),
            journal_path=Path("journal.jsonl"),
            counters_path=Path("counters.json"),
            cleanup_artifacts={},
            cleanup_decrements=(),
            science_profile=day_runner.ScienceProfile(
                profile_id="test-profile",
                source_commit="test-commit",
                artifact_digests={},
            ),
            versions={"node-fdm-pipeline": "test-code-v1"},
            local_workers=2,
            max_resident_gib=8.0,
        )

        assert len(captured_slices) == 2
        assert all(contains_handle(vars(slice_)) for slice_ in captured_slices)
        assert [source_day for source_day, _field in open_calls] == ["20200101"]

        def test_partitions_to_assemble_keeps_only_pending_in_plan_order() -> None:
            """AC1: only pending partition keys remain, preserving the plan order."""
            day_runner = importlib.import_module("node_fdm_pipeline.commands._day_runner")
            day_plan = importlib.import_module("node_fdm_pipeline.commands._day_plan")
            keys = tuple(
                day_plan.DayPartitionKey(cohort, "20200101")
                for cohort in ("A20N", "B738", "C56X", "E190", "LJ45")
            )
            plan = day_plan.DayPlan(
                meta_selection_day="20200101",
                partition_keys=keys,
                selection_ids=frozenset(f"sel-{key.cohort}" for key in keys),
                selection_ids_by_key={key: frozenset({f"sel-{key.cohort}"}) for key in keys},
                source_days=("20200101",),
            )
            resume_state = {
                keys[0]: "pending",
                keys[1]: "staged",
                keys[2]: "pending",
                keys[3]: "published",
                keys[4]: "pending",
            }

            assert day_runner.partitions_to_assemble(plan, resume_state) == (
                keys[0],
                keys[2],
                keys[4],
            )


def test_partitions_to_assemble_keeps_only_pending_in_plan_order() -> None:
    """AC1: only pending partition keys remain, preserving the plan order."""
    day_runner = importlib.import_module("node_fdm_pipeline.commands._day_runner")
    day_plan = importlib.import_module("node_fdm_pipeline.commands._day_plan")
    keys = tuple(
        day_plan.DayPartitionKey(cohort, "20200101")
        for cohort in ("A20N", "B738", "C56X", "E190", "LJ45")
    )
    plan = day_plan.DayPlan(
        meta_selection_day="20200101",
        partition_keys=keys,
        selection_ids=frozenset(f"sel-{key.cohort}" for key in keys),
        selection_ids_by_key={key: frozenset({f"sel-{key.cohort}"}) for key in keys},
        source_days=("20200101",),
    )
    resume_state = {
        keys[0]: "pending",
        keys[1]: "staged",
        keys[2]: "pending",
        keys[3]: "published",
        keys[4]: "pending",
    }

    assert day_runner.partitions_to_assemble(plan, resume_state) == (
        keys[0],
        keys[2],
        keys[4],
    )
