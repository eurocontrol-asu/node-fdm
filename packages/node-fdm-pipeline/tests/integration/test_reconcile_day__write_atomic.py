"""Integration contracts for crash-safe absence-payload reconciliation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands import _fleet_fetch, _raw_cache
from node_fdm_pipeline.commands._cache_retention import reconcile_day
from node_fdm_pipeline.commands._fleet_fetch import DateOutcome, download_fleet
from node_fdm_pipeline.commands._fleet_plan import Cohort, FleetPlan
from node_fdm_pipeline.config import PathsConfig, PipelineConfig

_DAY = "20191231"
_ICAO24S = ("a00001",)
_KINDS: tuple[_raw_cache.Kind, ...] = ("history", "extended", "flightlist")


def _plan(tmp_path: Path) -> FleetPlan:
    cfg = PipelineConfig.model_construct(
        paths=PathsConfig(
            data_dir=tmp_path / "cohort-a",
            era5_cache_dir=str(tmp_path / "era5-cache"),
        )
    )
    cohort = Cohort(
        name="A",
        config_path=tmp_path / "cohort-a.yaml",
        selection_path=tmp_path / "cohort-a.csv",
        cfg=cfg,
        icao24=frozenset(_ICAO24S),
    )
    return FleetPlan(
        dates={_DAY: list(_ICAO24S)},
        owner={_ICAO24S[0]: (cohort,)},
        cohorts=(cohort,),
    )


def _root(plan: FleetPlan, kind: _raw_cache.Kind = "history") -> Path:
    return _raw_cache.cache_root(plan.cohorts[0].cfg, kind)


def _payload_path(root: Path, digest: str) -> Path:
    return root / f"date={_DAY}" / ".absences" / f"{digest}.parquet"


def _receipt_path(root: Path, kind: _raw_cache.Kind = "history") -> Path:
    return root / ".absences" / f"{_DAY}.{kind}.json"


def _write_payload(root: Path, kind: _raw_cache.Kind = "history") -> tuple[str, Path]:
    digest = _raw_cache.absence_digest(_DAY, kind, _ICAO24S)
    payload = _payload_path(root, digest)
    _raw_cache.write_atomic(payload, pl.DataFrame(schema={"icao24": pl.String}))
    return digest, payload


def _stage_other_kinds(plan: FleetPlan) -> None:
    for kind in _KINDS:
        if kind != "history":
            _raw_cache.publish_absence(_root(plan, kind), _DAY, kind, _ICAO24S)


def _recording_boundary(
    root: Path,
    payload: Path,
    calls: list[str],
    payload_present_at_call: list[bool],
) -> Callable[..., DateOutcome]:
    def fetch(
        plan: FleetPlan,
        date_str: str,
        *,
        force: bool = False,
    ) -> DateOutcome:
        del plan, force
        calls.append(date_str)
        payload_present_at_call.append(payload.exists())
        _raw_cache.publish_absence(root, date_str, "history", _ICAO24S)
        return DateOutcome(
            date=date_str,
            requested=1,
            written=0,
            requests=1,
            empty_kinds=("history",),
        )

    return fetch


def _run(
    plan: FleetPlan,
    tmp_path: Path,
    mocker: MockerFixture,
    payload: Path,
) -> tuple[list[str], list[bool]]:
    calls: list[str] = []
    payload_present_at_call: list[bool] = []
    mocker.patch.object(_fleet_fetch, "_get_opensky")
    download_fleet(
        plan,
        manifest_path=tmp_path / "download-fleet.manifest.jsonl",
        fetch_boundary=_recording_boundary(
            root=_root(plan),
            payload=payload,
            calls=calls,
            payload_present_at_call=payload_present_at_call,
        ),
    )
    return calls, payload_present_at_call


@pytest.mark.integration
def test_complete_payload_without_receipt_is_recovered_without_boundary_call(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC1: an intact payload republishes its receipt without acquisition."""
    plan = _plan(tmp_path)
    root = _root(plan)
    digest, payload = _write_payload(root)
    _stage_other_kinds(plan)

    calls, _ = _run(plan, tmp_path, mocker, payload)

    reconciliation = reconcile_day(root, _DAY, "history")
    assert calls == []
    assert reconciliation.status == "visible"
    assert reconciliation.digest == digest
    assert _receipt_path(root).is_file()


@pytest.mark.integration
def test_truncated_payload_is_removed_before_single_reacquisition(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC2: a truncated payload is removed before one boundary re-acquisition."""
    plan = _plan(tmp_path)
    root = _root(plan)
    digest, payload = _write_payload(root)
    payload.write_bytes(b"truncated-parquet")
    _stage_other_kinds(plan)

    calls, payload_present_at_call = _run(plan, tmp_path, mocker, payload)

    reconciliation = reconcile_day(root, _DAY, "history")
    assert calls == [_DAY]
    assert payload_present_at_call == [False]
    assert reconciliation.status == "visible"
    assert reconciliation.digest == digest
    assert _raw_cache.read_parquet(payload).height == 0


@pytest.mark.integration
def test_mismatched_receipt_removes_payload_and_reacquires(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC3: a mismatched receipt invalidates the payload before re-acquisition."""
    plan = _plan(tmp_path)
    root = _root(plan)
    digest, payload = _write_payload(root)
    foreign_receipt = _raw_cache.AbsenceReceipt(
        day=_DAY,
        kind="history",
        digest="f" * 64,
        icao24s=_ICAO24S,
    )
    receipt_path = _receipt_path(root)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(f"{foreign_receipt.model_dump_json()}\n", encoding="utf-8")
    _stage_other_kinds(plan)

    calls, payload_present_at_call = _run(plan, tmp_path, mocker, payload)

    receipt = _raw_cache.read_absence_receipt(root, _DAY, "history")
    reconciliation = reconcile_day(root, _DAY, "history")
    assert calls == [_DAY]
    assert payload_present_at_call == [False]
    assert receipt is not None
    assert receipt.digest == digest
    assert reconciliation.status == "visible"
    assert reconciliation.digest == receipt.digest
