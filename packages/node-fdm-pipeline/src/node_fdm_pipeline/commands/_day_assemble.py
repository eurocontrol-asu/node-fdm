from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import polars as pl

from node_fdm_pipeline.commands._day_plan import (
    DayPartitionKey,
    DayPlan,
    DayScopeViolation,
    require_selection_in_scope,
)
from node_fdm_pipeline.commands._science_profile import ScienceProfile, profile_manifest

__all__ = ["PARTITION_IDENTITY_COLUMNS", "assemble_partition"]

PARTITION_IDENTITY_COLUMNS = (
    "meta_selection_day",
    "meta_source_day",
    "selection_id",
    "msn",
    "split",
    "profile_id",
    "profile_digest",
    "code_version",
)


def _profile_digest(profile: ScienceProfile) -> str:
    manifest = profile_manifest(profile)
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _partition_key(key: DayPartitionKey | tuple[str, str]) -> DayPartitionKey:
    return DayPartitionKey(*key)


def _identity_column(frame: pl.DataFrame, target: str, sources: tuple[str, ...]) -> pl.Expr:
    for source in sources:
        if source in frame.columns:
            return pl.col(source).alias(target)
    joined = ", ".join(sources)
    raise ValueError(f"partition input requires one of these identity columns: {joined}")


def _candidate_rows(
    frame: pl.DataFrame,
    plan: DayPlan,
    key: DayPartitionKey,
) -> pl.DataFrame:
    candidate = frame
    if "cohort" in candidate.columns:
        candidate = candidate.filter(pl.col("cohort") == key.cohort)
    if "meta_source_day" in candidate.columns:
        candidate = candidate.filter(pl.col("meta_source_day").is_in(plan.source_days))
    return candidate


def _require_candidate_scope(frame: pl.DataFrame, plan: DayPlan) -> None:
    for selection_id in frame.get_column("selection_id").unique().to_list():
        if not isinstance(selection_id, str):
            raise DayScopeViolation("selection_id must be a non-null string")
        require_selection_in_scope(plan, selection_id)


def assemble_partition(
    frame: pl.DataFrame,
    plan: DayPlan,
    key: DayPartitionKey | tuple[str, str],
    profile: ScienceProfile,
    versions: Mapping[str, str],
) -> pl.DataFrame:
    """Assemble one deterministic, provenance-complete day partition."""
    partition_key = _partition_key(key)
    try:
        authorised_ids = plan.selection_ids_by_key[partition_key]
    except KeyError:
        raise DayScopeViolation(
            f"partition {tuple(partition_key)!r} is outside the day plan"
        ) from None

    candidate = _candidate_rows(frame, plan, partition_key)
    _require_candidate_scope(candidate, plan)
    selected = candidate.filter(pl.col("selection_id").is_in(authorised_ids))

    manifest = profile_manifest(profile)
    try:
        code_version = versions["node-fdm-pipeline"]
    except KeyError:
        raise ValueError("versions requires a node-fdm-pipeline entry") from None

    assembled = (
        selected.with_columns(
            pl.lit(plan.meta_selection_day).alias("meta_selection_day"),
            _identity_column(selected, "msn", ("msn", "meta_msn", "raw_icao24")),
            _identity_column(selected, "split", ("split", "meta_split")),
            pl.lit(manifest["profile_id"]).alias("profile_id"),
            pl.lit(_profile_digest(profile)).alias("profile_digest"),
            pl.lit(code_version).alias("code_version"),
        )
        .unique(subset=["selection_id", "raw_timestamp"], keep="first")
        .sort(["selection_id", "meta_source_day", "raw_timestamp"])
    )

    if assembled.select(PARTITION_IDENTITY_COLUMNS).null_count().row(0) != (0,) * len(
        PARTITION_IDENTITY_COLUMNS
    ):
        raise ValueError("partition identity columns must not contain null values")
    return assembled
