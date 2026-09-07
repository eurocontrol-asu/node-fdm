from __future__ import annotations

import hashlib
import importlib
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import JsonValue

from node_fdm_pipeline import config as config_module
from node_fdm_pipeline.commands import _campaign_plan, _fleet_fetch, data
from node_fdm_pipeline.commands._campaign_plan import CampaignPlan
from node_fdm_pipeline.commands._fleet_journal import JournalEvent, RunState, record_state
from node_fdm_pipeline.commands._fleet_manifest import append_event
from node_fdm_pipeline.config import (
    FleetRunConfig,
    PathsConfig,
    PipelineConfig,
    SelectedParamConfig,
)

pytestmark = pytest.mark.integration

_OPERATOR_FIELDS = {
    "duration",
    "bytes",
    "rows",
    "flight_identities",
    "rejects",
    "split_batches",
    "cache_hits",
    "cache_misses",
    "throughput",
    "eta",
    "errors",
    "caches",
}


def _report_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._campaign_report")


def _metric_receipt(
    *,
    duration: float = 2.0,
    byte_count: int = 400,
    day_journals: list[str] | None = None,
) -> dict[str, JsonValue]:
    journals: list[JsonValue] = list(day_journals or [])
    return {
        "duration": duration,
        "bytes": byte_count,
        "rows": 20,
        "flight_identities": ["flight-001", "flight-002"],
        "rejects": 3,
        "split_batches": 2,
        "cache_hits": 7,
        "cache_misses": 1,
        "errors": [],
        "caches": ["opensky-history", "era5"],
        "day_journals": journals,
    }


def _record_step(
    journal: Path,
    receipts: Path,
    step: str,
    *,
    state: RunState = RunState.COMMITTED,
    receipt: dict[str, JsonValue] | None = None,
) -> None:
    record_state(
        journal,
        receipts,
        JournalEvent(
            acquisition_key=step,
            state=state,
            timestamp=datetime(2026, 9, 7, tzinfo=UTC),
            receipt=receipt or _metric_receipt(),
        ),
    )


def _install_local_inputs(
    report_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config_path: Path,
    planned_bytes: int = 1_200,
    **paths: Path,
) -> None:
    journal = paths["journal"]
    receipts = paths["receipts"]
    published_root = paths["published_root"]
    resolved = PipelineConfig.model_construct(
        paths=PathsConfig(data_dir=published_root),
        typecodes=["A320"],
        fleet_run=FleetRunConfig(
            lease_path=config_path.with_suffix(".lease"),
            lease_ttl_s=60,
            disk_min_gib=1.0,
            acquisition_journal=journal,
            acquisition_receipt_dir=receipts,
        ),
        selected_params=SelectedParamConfig.model_construct(),
    )
    plan = CampaignPlan(
        identified_flights=frozenset({"flight-001", "flight-002"}),
        identified_by_cohort={"alpha": frozenset({"flight-001", "flight-002"})},
        trino_batches=[],
        crossmidnight_dependencies=[],
        estimated_disk_bytes=planned_bytes,
        steps_to_resume=[],
    )

    def load_local_config(
        _cls: type[PipelineConfig],
        _path: Path,
        *,
        data_root: Path | None = None,
    ) -> PipelineConfig:
        del data_root
        return resolved

    def load_local_plan(_path: Path) -> CampaignPlan:
        return plan

    monkeypatch.setattr(
        config_module.PipelineConfig,
        "from_yaml",
        classmethod(load_local_config),
    )
    monkeypatch.setattr(_campaign_plan, "campaign_plan", load_local_plan)
    monkeypatch.setattr(report_module, "campaign_plan", load_local_plan, raising=False)


def _campaign_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    journal = tmp_path / "campaign.journal.jsonl"
    receipts = tmp_path / "receipts"
    published_root = tmp_path / "published"
    published_root.mkdir()
    return config_path, journal, receipts, published_root


def _published_partition(
    published_root: Path,
    day_journal: Path,
    *,
    cohort: str,
    day: str,
    content: bytes,
) -> tuple[Path, str]:
    partition = published_root / cohort / f"{day}.parquet"
    partition.parent.mkdir(parents=True, exist_ok=True)
    partition.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    append_event(
        day_journal,
        {
            "event": "publish_partition",
            "cohort": cohort,
            "meta_selection_day": day,
            "path": str(partition),
            "digest": digest,
            "row_count": 1,
        },
    )
    return partition, digest


def test_campaign_status_reports_every_journalled_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC1: status returns one complete operator row per journalled step."""
    config_path, journal, receipts, published_root = _campaign_files(tmp_path)
    for step in ("acquire", "decode", "publish"):
        _record_step(journal, receipts, step)
    report_module = _report_module()
    _install_local_inputs(
        report_module,
        monkeypatch,
        config_path=config_path,
        journal=journal,
        receipts=receipts,
        published_root=published_root,
    )

    report = report_module.campaign_status(config_path)

    assert [step.step for step in report.steps] == ["acquire", "decode", "publish"]
    assert all(_OPERATOR_FIELDS <= set(step.model_dump()) for step in report.steps)
    assert all(step.bytes == 400 and step.rows == 20 for step in report.steps)


def test_campaign_status_names_the_interrupted_step_to_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2: status names the exact step whose final durable event is interrupted."""
    config_path, journal, receipts, published_root = _campaign_files(tmp_path)
    _record_step(journal, receipts, "acquire")
    _record_step(journal, receipts, "decode", state=RunState.PROCESSING)
    _record_step(journal, receipts, "decode", state=RunState.INTERRUPTED)
    report_module = _report_module()
    _install_local_inputs(
        report_module,
        monkeypatch,
        config_path=config_path,
        journal=journal,
        receipts=receipts,
        published_root=published_root,
    )

    report = report_module.campaign_status(config_path)

    assert report.next_actions == ["resume decode"]
    decode = next(step for step in report.steps if step.step == "decode")
    assert decode.next_actions == ["resume decode"]


def test_campaign_validate_reports_a_missing_published_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3: validation names every missing or digest-divergent published partition."""
    config_path, journal, receipts, published_root = _campaign_files(tmp_path)
    day_journal = tmp_path / "day.journal.jsonl"
    missing, _ = _published_partition(
        published_root,
        day_journal,
        cohort="alpha",
        day="2026-01-02",
        content=b"missing",
    )
    divergent, recorded_digest = _published_partition(
        published_root,
        day_journal,
        cohort="beta",
        day="2026-01-02",
        content=b"recorded",
    )
    missing.unlink()
    divergent.write_bytes(b"changed")
    disk_digest = hashlib.sha256(b"changed").hexdigest()
    _record_step(
        journal,
        receipts,
        "publish",
        receipt=_metric_receipt(day_journals=[str(day_journal)]),
    )
    report_module = _report_module()
    _install_local_inputs(
        report_module,
        monkeypatch,
        config_path=config_path,
        journal=journal,
        receipts=receipts,
        published_root=published_root,
    )

    report = report_module.campaign_validate(config_path)
    errors = "\n".join(report.errors)

    assert "alpha" in errors and "2026-01-02" in errors
    assert "beta" in errors and recorded_digest in errors and disk_digest in errors


def test_campaign_validate_reports_a_digest_divergence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3: validation reports both recorded and observed digests for a partition."""
    config_path, journal, receipts, published_root = _campaign_files(tmp_path)
    day_journal = tmp_path / "day.journal.jsonl"
    partition, recorded_digest = _published_partition(
        published_root,
        day_journal,
        cohort="alpha",
        day="2026-01-02",
        content=b"recorded",
    )
    partition.write_bytes(b"rewritten")
    observed_digest = hashlib.sha256(b"rewritten").hexdigest()
    _record_step(
        journal,
        receipts,
        "publish",
        receipt=_metric_receipt(day_journals=[str(day_journal)]),
    )
    report_module = _report_module()
    _install_local_inputs(
        report_module,
        monkeypatch,
        config_path=config_path,
        journal=journal,
        receipts=receipts,
        published_root=published_root,
    )

    report = report_module.campaign_validate(config_path)
    errors = "\n".join(report.errors)

    assert "alpha" in errors and "2026-01-02" in errors
    assert recorded_digest in errors
    assert observed_digest in errors


def test_campaign_validate_returns_valid_for_an_agreeing_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC4: an agreeing journal/tree is valid and retains all operator figures."""
    config_path, journal, receipts, published_root = _campaign_files(tmp_path)
    day_journal = tmp_path / "day.journal.jsonl"
    _published_partition(
        published_root,
        day_journal,
        cohort="alpha",
        day="2026-01-02",
        content=b"stable",
    )
    _record_step(
        journal,
        receipts,
        "publish",
        receipt=_metric_receipt(day_journals=[str(day_journal)]),
    )
    report_module = _report_module()
    _install_local_inputs(
        report_module,
        monkeypatch,
        config_path=config_path,
        journal=journal,
        receipts=receipts,
        published_root=published_root,
    )

    report = report_module.campaign_validate(config_path)

    assert report.verdict == "valid"
    assert report.errors == []
    assert _OPERATOR_FIELDS <= set(report.steps[0].model_dump())


def test_status_and_validate_never_construct_a_remote_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC5: both report modes stay complete when every remote entry point is faulted."""
    config_path, journal, receipts, published_root = _campaign_files(tmp_path)
    day_journal = tmp_path / "day.journal.jsonl"
    _published_partition(
        published_root,
        day_journal,
        cohort="alpha",
        day="2026-01-02",
        content=b"stable",
    )
    _record_step(
        journal,
        receipts,
        "publish",
        receipt=_metric_receipt(day_journals=[str(day_journal)]),
    )
    report_module = _report_module()
    _install_local_inputs(
        report_module,
        monkeypatch,
        config_path=config_path,
        journal=journal,
        receipts=receipts,
        published_root=published_root,
    )
    calls = 0

    def raise_if_remote_is_constructed() -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("status/validate must not construct OpenSky or Trino clients")

    monkeypatch.setattr(_fleet_fetch, "_get_opensky", raise_if_remote_is_constructed)
    monkeypatch.setattr(data, "_get_opensky", raise_if_remote_is_constructed)

    status = report_module.campaign_status(config_path)
    validation = report_module.campaign_validate(config_path)

    assert len(status.steps) == 1
    assert validation.verdict == "valid"
    assert calls == 0
