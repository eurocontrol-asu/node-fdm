from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import polars as pl
import pytest

from node_fdm_pipeline.commands import _fleet_fetch as fleet_fetch
from node_fdm_pipeline.commands import _raw_cache as raw_cache
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._fleet_plan import Cohort, FleetPlan
from node_fdm_pipeline.commands._fleet_selection import compile_selection

if TYPE_CHECKING:
    from node_fdm_pipeline.config import PipelineConfig

_DAY = "20191231"
_ICAO24 = "z00002"
_SELECTION_ID = "sel-zulu"


class _RecordingBoundary:
    def __init__(
        self,
        cfg: PipelineConfig,
        receipt_ids: tuple[str, ...],
        *,
        stage_payload: bool,
    ) -> None:
        self.cfg = cfg
        self.receipt_ids = receipt_ids
        self.stage_payload = stage_payload
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def __call__(
        self,
        plan: FleetPlan,
        date_str: str,
        *,
        force: bool,
    ) -> fleet_fetch.DateOutcome:
        assert force is False
        aircraft = tuple(plan.dates[date_str])
        self.calls.append((date_str, "history", aircraft))
        if self.stage_payload:
            for icao24 in aircraft:
                raw_cache.write_atomic(
                    raw_cache.cache_path(self.cfg, "history", date_str, icao24),
                    pl.DataFrame({"icao24": [icao24]}),
                )
        receipt = raw_cache.publish_absence(
            raw_cache.cache_root(self.cfg, "history"),
            date_str,
            "history",
            self.receipt_ids,
        )
        return fleet_fetch.DateOutcome(
            date=date_str,
            requested=len(aircraft),
            written=len(aircraft) if self.stage_payload else 0,
            requests=1,
            empty_kinds=() if self.stage_payload else ("history",),
            staged_digest=receipt.digest,
        )


@pytest.fixture(autouse=True)
def history_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fleet_fetch, "_KINDS", ("history",))
    monkeypatch.setattr(fleet_fetch, "_get_opensky", lambda: object())


def _campaign(tmp_path: Path, *, with_selection: bool) -> tuple[FleetPlan, PipelineConfig]:
    cfg = cast(
        "PipelineConfig",
        SimpleNamespace(paths=SimpleNamespace(data_dir=tmp_path / "campaign" / "C1")),
    )
    cohort = Cohort(
        name="C1",
        config_path=tmp_path / "config.yaml",
        selection_path=tmp_path / "selection.csv",
        cfg=cfg,
        icao24=frozenset({_ICAO24}),
    )
    selection = (
        compile_selection(
            [
                {
                    "icao24": _ICAO24,
                    "callsign": "ZULU2",
                    "firstseen": 1577793600,
                    "lastseen": 1577797200,
                    "msn": "M2",
                    "split": "test",
                    "cohort": "C1",
                    "selection_id": _SELECTION_ID,
                }
            ]
        )
        if with_selection
        else None
    )
    plan = FleetPlan(
        dates={_DAY: [_ICAO24]},
        owner={_ICAO24: (cohort,)},
        cohorts=(cohort,),
        selection=selection,
    )
    return plan, cfg


def _run(
    plan: FleetPlan,
    manifest_path: Path,
    boundary: _RecordingBoundary,
) -> list[fleet_fetch.DateOutcome]:
    return fleet_fetch.download_fleet(
        plan,
        manifest_path=manifest_path,
        fetch_boundary=boundary,
    )


@pytest.mark.integration
def test_verified_staged_day_is_reused_and_journalled_with_digest(tmp_path: Path) -> None:
    """AC1: a fresh campaign reuses a verified day and journals its receipt digest."""
    plan, cfg = _campaign(tmp_path, with_selection=False)
    first_boundary = _RecordingBoundary(cfg, (_ICAO24,), stage_payload=True)
    _run(plan, tmp_path / "first.jsonl", first_boundary)
    receipt = raw_cache.read_absence_receipt(
        raw_cache.cache_root(cfg, "history"),
        _DAY,
        "history",
    )
    assert receipt is not None

    second_boundary = _RecordingBoundary(cfg, (_ICAO24,), stage_payload=True)
    second_manifest = tmp_path / "second.jsonl"
    _run(plan, second_manifest, second_boundary)

    reused = [
        event
        for event in read_events(second_manifest)
        if event.get("event") == "date_reused"
        and event.get("date") == _DAY
        and event.get("kind") == "history"
    ]
    assert second_boundary.calls == []
    assert len(reused) == 1
    assert reused[0]["receipt_digest"] == receipt.digest


@pytest.mark.integration
def test_published_absence_is_not_requeried_and_rejection_is_restored(tmp_path: Path) -> None:
    """AC2: resume restores sel-zulu's absent rejection without a boundary call."""
    plan, cfg = _campaign(tmp_path, with_selection=True)
    first_boundary = _RecordingBoundary(cfg, (_SELECTION_ID,), stage_payload=False)
    _run(plan, tmp_path / "first.jsonl", first_boundary)

    second_boundary = _RecordingBoundary(cfg, (_SELECTION_ID,), stage_payload=False)
    outcomes = _run(plan, tmp_path / "second.jsonl", second_boundary)

    rejection = outcomes[0].rejections[_SELECTION_ID]
    assert second_boundary.calls == []
    assert rejection.kind == "absent"
    assert rejection.selection_id == _SELECTION_ID


@pytest.mark.integration
def test_staged_payload_without_receipt_is_reacquired_and_republished(tmp_path: Path) -> None:
    """AC3: a staged payload without its receipt is acquired exactly once again."""
    plan, cfg = _campaign(tmp_path, with_selection=False)
    first_boundary = _RecordingBoundary(cfg, (_ICAO24,), stage_payload=True)
    _run(plan, tmp_path / "first.jsonl", first_boundary)
    root = raw_cache.cache_root(cfg, "history")
    receipt_path = root / ".absences" / f"{_DAY}.history.json"
    assert receipt_path.is_file()
    receipt_path.unlink()

    second_boundary = _RecordingBoundary(cfg, (_ICAO24,), stage_payload=True)
    _run(plan, tmp_path / "second.jsonl", second_boundary)

    assert second_boundary.calls == [(_DAY, "history", (_ICAO24,))]
    assert raw_cache.read_absence_receipt(root, _DAY, "history") is not None
