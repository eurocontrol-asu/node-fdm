from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from node_fdm_pipeline.commands._day_plan import DayPartitionKey, DayPlan
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._raw_cache import write_atomic
from node_fdm_pipeline.commands._science_profile import (
    ScienceProfile,
    profile_manifest,
    resolve_science_profile,
)

DAY = "20200102"
EXPECTED_KEYS = tuple(
    DayPartitionKey(cohort, DAY) for cohort in ("A319", "A320", "A321", "A332", "A359")
)


def _day_commit() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_commit")


def _plan() -> DayPlan:
    selection_ids_by_key = {key: frozenset({f"selection-{key.cohort}"}) for key in EXPECTED_KEYS}
    return DayPlan(
        meta_selection_day=DAY,
        partition_keys=EXPECTED_KEYS,
        selection_ids=frozenset().union(*selection_ids_by_key.values()),
        selection_ids_by_key=selection_ids_by_key,
        source_days=(DAY,),
    )


def _profile_digest(profile: ScienceProfile) -> str:
    payload = json.dumps(
        profile_manifest(profile),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _partition_path(root: Path, key: DayPartitionKey) -> Path:
    return (
        root
        / f"cohort={key.cohort}"
        / f"meta_selection_day={key.meta_selection_day}"
        / "data.parquet"
    )


def _publish_keys(
    root: Path,
    plan: DayPlan,
    profile: ScienceProfile,
    keys: tuple[DayPartitionKey, ...],
) -> None:
    profile_digest = _profile_digest(profile)
    for key in keys:
        frame = pl.DataFrame(
            {
                "cohort": [key.cohort],
                "meta_selection_day": [key.meta_selection_day],
                "meta_source_day": [key.meta_selection_day],
                "selection_id": [next(iter(plan.selection_ids_by_key[key]))],
                "msn": [f"MSN-{key.cohort}"],
                "split": ["train"],
                "profile_id": [profile.profile_id],
                "profile_digest": [profile_digest],
                "code_version": ["test-version"],
            }
        )
        write_atomic(_partition_path(root, key), frame)


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _committed_events(journal: Path) -> list[dict[str, object]]:
    return [event for event in read_events(journal) if event.get("event") == "day_committed"]


def _assert_valid_marker(
    event: dict[str, object],
    plan: DayPlan,
    profile: ScienceProfile,
) -> None:
    assert event["meta_selection_day"] == DAY
    assert event["partition_keys"] == [list(key) for key in plan.partition_keys]
    assert event["profile_manifest"] == profile_manifest(profile)


@pytest.fixture
def day_state(
    tmp_path: Path,
) -> tuple[DayPlan, ScienceProfile, Path, Path]:
    plan = _plan()
    profile = resolve_science_profile("opensky26-exp03-v1")
    published_root = tmp_path / "published"
    journal = tmp_path / "journal.jsonl"
    journal.touch()
    return plan, profile, published_root, journal


def test_an_incomplete_day_is_not_committed(
    day_state: tuple[DayPlan, ScienceProfile, Path, Path],
) -> None:
    """AC1: commit refuses one absent partition and leaves the journal uncommitted."""
    day_commit = _day_commit()
    plan, profile, published_root, journal = day_state
    missing_key = plan.partition_keys[-1]
    _publish_keys(published_root, plan, profile, plan.partition_keys[:-1])

    with pytest.raises(day_commit.DayIncomplete) as excinfo:
        day_commit.commit_day(plan, published_root, journal, profile)

    assert repr(tuple(missing_key)) in str(excinfo.value)
    assert _committed_events(journal) == []


def test_a_complete_day_is_committed_exactly_once(
    day_state: tuple[DayPlan, ScienceProfile, Path, Path],
) -> None:
    """AC2: a complete day gets one manifest-rich marker and repeat commit is a no-op."""
    day_commit = _day_commit()
    plan, profile, published_root, journal = day_state
    _publish_keys(published_root, plan, profile, plan.partition_keys)

    day_commit.commit_day(plan, published_root, journal, profile)
    day_commit.commit_day(plan, published_root, journal, profile)

    committed = _committed_events(journal)
    assert len(committed) == 1
    _assert_valid_marker(committed[0], plan, profile)


def test_a_crash_during_marker_write_reconciles_to_one_valid_marker(
    day_state: tuple[DayPlan, ScienceProfile, Path, Path],
) -> None:
    """AC3: marker-write recovery is unique and never republishes partition content."""
    day_commit = _day_commit()
    plan, profile, published_root, journal = day_state
    _publish_keys(published_root, plan, profile, plan.partition_keys)
    before_digests = {
        key: _digest(_partition_path(published_root, key)) for key in plan.partition_keys
    }

    def crash_on_marker(step: str) -> None:
        if step == "day_committed":
            raise RuntimeError("injected marker-write crash")

    with pytest.raises(RuntimeError, match="injected marker-write crash"):
        day_commit.commit_day(
            plan,
            published_root,
            journal,
            profile,
            step_hook=crash_on_marker,
        )

    interrupted = _committed_events(journal)
    assert len(interrupted) in {0, 1}
    if interrupted:
        _assert_valid_marker(interrupted[0], plan, profile)

    day_commit.reconcile_commit(journal, published_root)
    day_commit.reconcile_commit(journal, published_root)

    reconciled = _committed_events(journal)
    assert len(reconciled) == 1
    _assert_valid_marker(reconciled[0], plan, profile)
    after_digests = {
        key: _digest(_partition_path(published_root, key)) for key in plan.partition_keys
    }
    assert after_digests == before_digests
