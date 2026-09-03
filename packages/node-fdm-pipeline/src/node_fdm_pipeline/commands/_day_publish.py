from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, cast

import polars as pl
from pydantic import BaseModel, ConfigDict

from node_fdm_pipeline.commands._day_assemble import PARTITION_IDENTITY_COLUMNS
from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan
from node_fdm_pipeline.commands._fleet_manifest import append_event, read_events
from node_fdm_pipeline.commands._raw_cache import write_atomic

__all__ = [
    "DayPublication",
    "PartitionValidationError",
    "PublishedPartition",
    "StagedPartition",
    "missing_partitions",
    "publish_day",
    "publish_partition",
    "resume_publication",
    "stage_partition",
    "validate_partition",
]

type PublicationStep = Literal[
    "stage_partition",
    "validate_partition",
    "publish_partition",
]
type BoundaryStep = Literal["stage", "validate", "publish", "day_committed"]
type StepHook = Callable[[BoundaryStep], None]
type LegacyPublicationStep = PublicationStep
type LegacyStepHook = Callable[[LegacyPublicationStep, DayPartitionKey], None]
type PublicationHook = StepHook | LegacyStepHook


class PartitionValidationError(ValueError):
    """Raised when a staged partition violates its active day plan."""


class StagedPartition(BaseModel):
    """Immutable reference to a completely staged partition."""

    model_config = ConfigDict(frozen=True)

    key: DayPartitionKey
    path: Path
    digest: str
    row_count: int


class PublishedPartition(BaseModel):
    """Immutable result of an atomic partition publication."""

    model_config = ConfigDict(frozen=True)

    key: DayPartitionKey
    path: Path
    digest: str
    row_count: int


class DayPublication(BaseModel):
    """Immutable report of the partitions published by one invocation."""

    model_config = ConfigDict(frozen=True)

    published_now: tuple[DayPartitionKey, ...]


def _partition_path(root: Path, key: DayPartitionKey) -> Path:
    return (
        root
        / f"cohort={key.cohort}"
        / f"meta_selection_day={key.meta_selection_day}"
        / "data.parquet"
    )


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _event(
    event_log: Path,
    step: PublicationStep,
    key: DayPartitionKey,
    *,
    digest: str,
    row_count: int,
) -> None:
    append_event(
        event_log,
        {
            "event": step,
            "cohort": key.cohort,
            "meta_selection_day": key.meta_selection_day,
            "digest": digest,
            "row_count": row_count,
        },
    )


def stage_partition(
    *,
    frame: pl.DataFrame,
    key: DayPartitionKey,
    staging_root: Path,
    event_log: Path,
) -> StagedPartition:
    """Durably stage one complete frame outside the published root."""
    path = _partition_path(staging_root, key)
    write_atomic(path, frame)
    result = StagedPartition(
        key=key,
        path=path,
        digest=_file_digest(path),
        row_count=frame.height,
    )
    _event(
        event_log,
        "stage_partition",
        key,
        digest=result.digest,
        row_count=result.row_count,
    )
    return result


def _require_partition_key(
    frame: pl.DataFrame,
    plan: DayPlan,
    key: DayPartitionKey,
) -> None:
    if key not in plan.partition_keys:
        raise PartitionValidationError(f"partition key {tuple(key)!r} is absent from DayPlan")
    if frame.filter(pl.col("cohort").is_null() | (pl.col("cohort") != key.cohort)).height:
        raise PartitionValidationError(f"staged frame does not belong to partition {tuple(key)!r}")
    if frame.filter(
        pl.col("meta_selection_day").is_null()
        | (pl.col("meta_selection_day") != key.meta_selection_day)
    ).height:
        raise PartitionValidationError(f"staged frame does not belong to partition {tuple(key)!r}")


def _require_identity_columns(frame: pl.DataFrame) -> None:
    required = ("cohort", *PARTITION_IDENTITY_COLUMNS)
    missing = tuple(column for column in required if column not in frame.columns)
    if missing:
        raise PartitionValidationError(f"staged frame is missing identity columns {missing!r}")
    null_counts = frame.select(required).null_count().row(0)
    if any(count != 0 for count in null_counts):
        raise PartitionValidationError("partition identity columns must not contain null values")


def _require_authorised_selections(
    frame: pl.DataFrame,
    plan: DayPlan,
    key: DayPartitionKey,
) -> None:
    authorised = plan.selection_ids_by_key[key]
    offending = frame.filter(~pl.col("selection_id").is_in(authorised)).select("selection_id")
    if offending.height:
        selection_id = str(offending.item(0, 0))
        raise PartitionValidationError(
            f"selection_id {selection_id!r} is absent from partition {tuple(key)!r}"
        )


def validate_partition(
    *,
    frame: pl.DataFrame,
    plan: DayPlan,
    key: DayPartitionKey,
    event_log: Path | None = None,
) -> None:
    """Validate identity and selection scope before a staged frame is published."""
    _require_identity_columns(frame)
    _require_partition_key(frame, plan, key)
    _require_authorised_selections(frame, plan, key)
    if event_log is not None:
        _event(
            event_log,
            "validate_partition",
            key,
            digest="validated",
            row_count=frame.height,
        )


def publish_partition(
    *,
    staged: StagedPartition,
    published_root: Path,
    event_log: Path,
) -> PublishedPartition:
    """Atomically publish a staged frame, without rewriting an existing partition."""
    path = _partition_path(published_root, staged.key)
    if path.exists():
        result = PublishedPartition(
            key=staged.key,
            path=path,
            digest=_file_digest(path),
            row_count=pl.read_parquet(path).height,
        )
        return result

    frame = pl.read_parquet(staged.path)
    write_atomic(path, frame)
    result = PublishedPartition(
        key=staged.key,
        path=path,
        digest=_file_digest(path),
        row_count=frame.height,
    )
    _event(
        event_log,
        "publish_partition",
        staged.key,
        digest=result.digest,
        row_count=result.row_count,
    )
    return result


def missing_partitions(
    plan: DayPlan,
    published_root: Path,
) -> tuple[DayPartitionKey, ...]:
    """Return the planned keys that have no atomically published file."""
    return tuple(
        key for key in plan.partition_keys if not _partition_path(published_root, key).is_file()
    )


def _split_step_hook(
    step_hook: PublicationHook | None,
) -> tuple[StepHook | None, LegacyStepHook | None]:
    if step_hook is None:
        return None, None
    try:
        inspect.signature(step_hook).bind("stage")
    except TypeError:
        return None, cast("LegacyStepHook", step_hook)
    return cast("StepHook", step_hook), None


def _publish_missing(  # noqa: PLR0913
    *,
    partitions: Mapping[DayPartitionKey, pl.DataFrame],
    plan: DayPlan,
    staging_root: Path,
    published_root: Path,
    event_log: Path,
    step_hook: PublicationHook | None,
) -> DayPublication:
    boundary_hook, legacy_hook = _split_step_hook(step_hook)
    published_now: list[DayPartitionKey] = []
    for key in missing_partitions(plan, published_root):
        try:
            frame = partitions[key]
        except KeyError:
            raise PartitionValidationError(
                f"no assembled frame supplied for {tuple(key)!r}"
            ) from None

        if boundary_hook is not None:
            boundary_hook("stage")
        staged = stage_partition(
            frame=frame,
            key=key,
            staging_root=staging_root,
            event_log=event_log,
        )
        if legacy_hook is not None:
            legacy_hook("stage_partition", key)

        staged_frame = pl.read_parquet(staged.path)
        if boundary_hook is not None:
            boundary_hook("validate")
        validate_partition(
            frame=staged_frame,
            plan=plan,
            key=key,
            event_log=event_log,
        )
        if legacy_hook is not None:
            legacy_hook("validate_partition", key)

        if boundary_hook is not None:
            boundary_hook("publish")
        publish_partition(
            staged=staged,
            published_root=published_root,
            event_log=event_log,
        )
        published_now.append(key)
        if legacy_hook is not None:
            legacy_hook("publish_partition", key)

    return DayPublication(published_now=tuple(published_now))


def publish_day(  # noqa: PLR0913
    *,
    partitions: Mapping[DayPartitionKey, pl.DataFrame],
    plan: DayPlan,
    staging_root: Path,
    published_root: Path,
    event_log: Path,
    step_hook: PublicationHook | None = None,
) -> DayPublication:
    """Publish only absent partitions through the stage, validate, publish sequence."""
    return _publish_missing(
        partitions=partitions,
        plan=plan,
        staging_root=staging_root,
        published_root=published_root,
        event_log=event_log,
        step_hook=step_hook,
    )


def resume_publication(
    *,
    partitions: Mapping[DayPartitionKey, pl.DataFrame],
    plan: DayPlan,
    staging_root: Path,
    published_root: Path,
    event_log: Path,
) -> DayPublication:
    """Resume a day after checking its durable event history for corruption."""
    if event_log.exists():
        read_events(event_log)
    return _publish_missing(
        partitions=partitions,
        plan=plan,
        staging_root=staging_root,
        published_root=published_root,
        event_log=event_log,
        step_hook=None,
    )
