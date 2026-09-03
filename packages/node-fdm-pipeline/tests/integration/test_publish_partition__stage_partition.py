from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan, build_day_plan
from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan

pytestmark = pytest.mark.integration


class InjectedPublicationError(RuntimeError):
    """Fault raised by a publication step hook in crash-recovery tests."""


def _day_publish() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_publish")


def _plan_and_frames(
    cohorts: tuple[str, ...] = ("A20N",),
    *,
    rows_per_partition: int = 1,
) -> tuple[DayPlan, dict[DayPartitionKey, pl.DataFrame]]:
    flights = tuple(
        SelectedFlight(
            icao24=f"abc{index:03d}",
            callsign=f"AXM{index:03d}",
            firstseen=1_577_836_800 + index,
            lastseen=1_577_837_100 + index,
            msn=f"{10_000 + index}",
            split="train",
            cohorts=frozenset({cohort}),
            selection_id=f"sel-{index:03d}",
            utc_days=("20200101",),
            acquisition_key=f"acq-{index:03d}",
        )
        for index, cohort in enumerate(cohorts)
    )
    plan = build_day_plan(
        SelectionPlan(flights=flights, digest="selection-digest"),
        "20200101",
    )
    flights_by_cohort = {next(iter(flight.cohorts)): flight for flight in flights}
    frames = {}
    for key in plan.partition_keys:
        flight = flights_by_cohort[key.cohort]
        frames[key] = pl.DataFrame(
            {
                "cohort": [key.cohort] * rows_per_partition,
                "meta_selection_day": [key.meta_selection_day] * rows_per_partition,
                "meta_source_day": ["20200101"] * rows_per_partition,
                "selection_id": [flight.selection_id] * rows_per_partition,
                "msn": [flight.msn] * rows_per_partition,
                "split": [flight.split] * rows_per_partition,
                "profile_id": ["science-v1"] * rows_per_partition,
                "profile_digest": ["profile-digest"] * rows_per_partition,
                "code_version": ["test-version"] * rows_per_partition,
                "raw_timestamp": [1_577_836_900 + sample for sample in range(rows_per_partition)],
            }
        )
    return plan, frames


def _row_digest(frame: pl.DataFrame) -> str:
    ordered = frame.sort(["selection_id", "raw_timestamp"])
    payload = json.dumps(ordered.to_dicts(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _byte_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partition_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.parquet")))


def test_publish_day_crash_after_stage_resumes_missing_partition(tmp_path: Path) -> None:
    """AC1: a crash after staging leaves the key absent, and resume publishes it."""
    day_publish = _day_publish()
    plan, frames = _plan_and_frames()
    key = plan.partition_keys[0]
    staging_root = tmp_path / "staging"
    published_root = tmp_path / "published"
    event_log = tmp_path / "publication.jsonl"

    def fail_after_stage(step: str, _key: DayPartitionKey) -> None:
        if step == "stage_partition":
            raise InjectedPublicationError(step)

    with pytest.raises(InjectedPublicationError):
        day_publish.publish_day(
            partitions=frames,
            plan=plan,
            staging_root=staging_root,
            published_root=published_root,
            event_log=event_log,
            step_hook=fail_after_stage,
        )

    assert _partition_files(published_root) == ()
    assert day_publish.missing_partitions(plan, published_root) == (key,)
    staged_files = _partition_files(staging_root)
    assert len(staged_files) == 1
    staged_oracle_digest = _row_digest(pl.read_parquet(staged_files[0]))

    outcome = day_publish.resume_publication(
        partitions=frames,
        plan=plan,
        staging_root=staging_root,
        published_root=published_root,
        event_log=event_log,
    )

    published_files = _partition_files(published_root)
    assert outcome.published_now == (key,)
    assert len(published_files) == 1
    assert _row_digest(pl.read_parquet(published_files[0])) == staged_oracle_digest


def test_publish_day_crash_after_validation_resumes_once(tmp_path: Path) -> None:
    """AC2: a crash after validation stays absent, then resume publishes one copy."""
    day_publish = _day_publish()
    plan, frames = _plan_and_frames(rows_per_partition=2)
    key = plan.partition_keys[0]
    oracle_count = frames[key].height
    staging_root = tmp_path / "staging"
    published_root = tmp_path / "published"
    event_log = tmp_path / "publication.jsonl"

    def fail_after_validation(step: str, _key: DayPartitionKey) -> None:
        if step == "validate_partition":
            raise InjectedPublicationError(step)

    with pytest.raises(InjectedPublicationError):
        day_publish.publish_day(
            partitions=frames,
            plan=plan,
            staging_root=staging_root,
            published_root=published_root,
            event_log=event_log,
            step_hook=fail_after_validation,
        )

    assert _partition_files(published_root) == ()

    outcome = day_publish.resume_publication(
        partitions=frames,
        plan=plan,
        staging_root=staging_root,
        published_root=published_root,
        event_log=event_log,
    )

    published_files = _partition_files(published_root)
    assert outcome.published_now == (key,)
    assert len(published_files) == 1
    published = pl.read_parquet(published_files[0])
    assert published.height == oracle_count
    assert published.unique(subset=["selection_id", "raw_timestamp"]).height == oracle_count


def test_publish_partition_is_idempotent(tmp_path: Path) -> None:
    """AC3: publishing one staged partition twice preserves its path, digest, and rows."""
    day_publish = _day_publish()
    plan, frames = _plan_and_frames(rows_per_partition=2)
    key = plan.partition_keys[0]
    staging_root = tmp_path / "staging"
    published_root = tmp_path / "published"
    event_log = tmp_path / "publication.jsonl"
    staged = day_publish.stage_partition(
        frame=frames[key],
        key=key,
        staging_root=staging_root,
        event_log=event_log,
    )
    day_publish.validate_partition(
        frame=pl.read_parquet(staged.path),
        plan=plan,
        key=key,
        event_log=event_log,
    )

    first = day_publish.publish_partition(
        staged=staged,
        published_root=published_root,
        event_log=event_log,
    )
    first_count = pl.read_parquet(first.path).height
    second = day_publish.publish_partition(
        staged=staged,
        published_root=published_root,
        event_log=event_log,
    )

    assert second.digest == first.digest
    assert second.path == first.path
    assert pl.read_parquet(second.path).height == first_count


def test_resume_publication_publishes_only_missing_partitions(tmp_path: Path) -> None:
    """AC5: resume reports only two missing keys and preserves three published bytes."""
    day_publish = _day_publish()
    plan, frames = _plan_and_frames(("A20N", "A21N", "B38M", "B39M", "B77W"))
    staging_root = tmp_path / "staging"
    published_root = tmp_path / "published"
    event_log = tmp_path / "publication.jsonl"
    already_published = plan.partition_keys[:3]

    published_paths = {}
    for key in already_published:
        staged = day_publish.stage_partition(
            frame=frames[key],
            key=key,
            staging_root=staging_root,
            event_log=event_log,
        )
        day_publish.validate_partition(
            frame=pl.read_parquet(staged.path),
            plan=plan,
            key=key,
            event_log=event_log,
        )
        result = day_publish.publish_partition(
            staged=staged,
            published_root=published_root,
            event_log=event_log,
        )
        published_paths[key] = result.path
    original_digests = {key: _byte_digest(path) for key, path in published_paths.items()}

    outcome = day_publish.resume_publication(
        partitions=frames,
        plan=plan,
        staging_root=staging_root,
        published_root=published_root,
        event_log=event_log,
    )

    assert outcome.published_now == plan.partition_keys[3:]
    assert len(_partition_files(published_root)) == 5
    assert {key: _byte_digest(path) for key, path in published_paths.items()} == original_digests
