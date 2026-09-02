"""Integration tests for the fleet download acquisition boundary."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from node_fdm_pipeline.commands import _fleet_fetch
from node_fdm_pipeline.commands._fleet_digest import (
    DigestInput,
    ResumeDigestMismatch,
    compute_resume_digest,
)
from node_fdm_pipeline.commands._fleet_journal import RunState, replay_journal
from node_fdm_pipeline.commands._fleet_manifest import read_events
from node_fdm_pipeline.commands._trino_lease import (
    LeaseConfigError,
    acquire_lease,
)
from node_fdm_pipeline.config import FleetRunConfig

_SELECTION_DIGEST = "recorded-offline-selection"
_RESOLVED_CONFIG: DigestInput = {"workers": 1, "mode": "recorded-offline"}
_PROFILE: DigestInput = {"aircraft": "A320", "version": 1}


def _plan() -> Any:
    return SimpleNamespace(
        cohorts=(SimpleNamespace(name="recorded"),),
        dates={"20240101": ["abc123"], "20240102": ["def456"]},
        aircraft_days=2,
        requests_saved=lambda: (2, 2),
    )


def _recorded_fetch(calls: list[str]) -> Any:
    def fetch(plan: Any, date: str, *, force: bool = False) -> Any:
        calls.append(date)
        return SimpleNamespace(
            date=date,
            requested=1,
            written=1,
            requests=1,
            empty_kinds=(),
            error=None,
        )

    return fetch


def _download_kwargs(
    run_dir: Path,
    fleet_config: FleetRunConfig,
    calls: list[str],
) -> dict[str, Any]:
    return {
        "fleet_config": fleet_config,
        "recorded_digest": compute_resume_digest(
            _SELECTION_DIGEST,
            _RESOLVED_CONFIG,
            _PROFILE,
        ),
        "selection_digest": _SELECTION_DIGEST,
        "resolved_config": _RESOLVED_CONFIG,
        "profile": _PROFILE,
        "manifest_path": run_dir / "download-fleet.manifest.jsonl",
        "journal_path": run_dir / "download-fleet.journal.jsonl",
        "receipt_dir": run_dir / "receipts",
        "fetch_boundary": _recorded_fetch(calls),
    }


@pytest.mark.integration
def test_download_fleet_rejects_unset_lease_before_fetch_or_lock(
    tmp_path: Path,
) -> None:
    """AC2: an unset shared lease rejects before fetching or creating a fallback lock."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[str] = []
    fleet_config = FleetRunConfig.model_construct(
        lease_path=None,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )

    with pytest.raises(LeaseConfigError, match="shared lease path"):
        _fleet_fetch.download_fleet(
            _plan(),
            **_download_kwargs(run_dir, fleet_config, calls),
        )

    assert calls == []
    assert list(run_dir.rglob("*.lock")) == []


@pytest.mark.integration
def test_download_fleet_journals_lifecycle_and_releases_lease(
    tmp_path: Path,
) -> None:
    """AC3: a recorded run replays acquiring then processing and releases its lease."""
    run_dir = tmp_path / "run"
    shared_dir = tmp_path / "shared"
    run_dir.mkdir()
    shared_dir.mkdir()
    lease_path = shared_dir / "download-fleet.lease"
    journal_path = run_dir / "download-fleet.journal.jsonl"
    calls: list[str] = []
    fleet_config = FleetRunConfig(
        lease_path=lease_path,
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )

    outcomes = _fleet_fetch.download_fleet(
        _plan(),
        **_download_kwargs(run_dir, fleet_config, calls),
    )

    events = read_events(journal_path)
    run_keys = {str(event["acquisition_key"]) for event in events}
    assert len(run_keys) == 1
    run_key = run_keys.pop()
    assert [RunState(str(event["state"])) for event in events] == [
        RunState.ACQUIRING,
        RunState.PROCESSING,
    ]
    assert replay_journal(journal_path).states[run_key] is RunState.PROCESSING
    assert [outcome.date for outcome in outcomes] == calls
    assert not lease_path.exists()


def _cli_plan(run_dir: Path) -> Any:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    cohort = SimpleNamespace(
        name="recorded",
        cfg=SimpleNamespace(paths=SimpleNamespace(data_dir=data_dir)),
    )
    return SimpleNamespace(
        cohorts=(cohort,),
        dates={"20240101": ["abc123"]},
        aircraft_days=1,
        requests_saved=lambda: (1, 1),
    )


def _campaign_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    selection = tmp_path / "selection.txt"
    resolved_config = tmp_path / "resolved-config.json"
    profile = tmp_path / "profile.json"
    selection.write_text(_SELECTION_DIGEST, encoding="utf-8")
    resolved_config.write_text('{"workers": 1}', encoding="utf-8")
    profile.write_text('{"aircraft": "A320"}', encoding="utf-8")
    return selection, resolved_config, profile


@pytest.mark.integration
def test_download_fleet_campaign_rejects_held_lease_before_provider_or_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC2: a live shared lease aborts the campaign before provider use or publication."""
    from node_fdm_pipeline.cli import download_fleet

    run_dir = tmp_path / "run"
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    plan = _cli_plan(run_dir)
    selection, resolved_config, profile = _campaign_files(tmp_path)
    lease_path = shared_dir / "download-fleet.lease"
    holder = acquire_lease(
        lease_path,
        owner="other-owner",
        ttl_s=60,
        now=time.time(),
    )
    provider_calls: list[str] = []

    monkeypatch.setattr(
        "node_fdm_pipeline.commands._fleet_plan.discover_cohorts",
        lambda _fleet_dir: [],
    )
    monkeypatch.setattr(
        "node_fdm_pipeline.commands._fleet_plan.build_fleet_plan",
        lambda _cohorts, *, data_root=None: plan,
    )
    monkeypatch.setattr(
        _fleet_fetch,
        "_get_opensky",
        lambda: provider_calls.append("provider"),
    )

    try:
        with pytest.raises(SystemExit) as raised:
            download_fleet(
                fleet_dir=tmp_path,
                workers=1,
                data_root=run_dir / "data",
                dry_run=False,
                force_refresh=False,
                selection=selection,
                resolved_config=resolved_config,
                profile=profile,
                lease_path=lease_path,
                lease_ttl_s=60,
                disk_min_gib=1.0,
            )
    finally:
        holder.release()

    captured = capsys.readouterr()
    assert raised.value.code == 1
    assert "Lease is held by other-owner" in captured.err
    assert provider_calls == []
    assert list(run_dir.rglob("*.parquet")) == []


@pytest.mark.integration
def test_download_fleet_partial_campaign_names_all_missing_inputs_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """AC3: a partial campaign reports every omission before lease or lock creation."""
    from node_fdm_pipeline.cli import download_fleet

    run_dir = tmp_path / "run"
    plan = _cli_plan(run_dir)
    selection = tmp_path / "selection.txt"
    selection.write_text(_SELECTION_DIGEST, encoding="utf-8")
    lease_path = tmp_path / "shared" / "download-fleet.lease"
    lease_path.parent.mkdir()
    provider_calls: list[str] = []

    monkeypatch.setattr(
        "node_fdm_pipeline.commands._fleet_plan.discover_cohorts",
        lambda _fleet_dir: [],
    )
    monkeypatch.setattr(
        "node_fdm_pipeline.commands._fleet_plan.build_fleet_plan",
        lambda _cohorts, *, data_root=None: plan,
    )
    monkeypatch.setattr(
        _fleet_fetch,
        "_get_opensky",
        lambda: provider_calls.append("provider"),
    )

    with pytest.raises(SystemExit) as raised:
        download_fleet(
            fleet_dir=tmp_path,
            workers=1,
            data_root=run_dir / "data",
            dry_run=False,
            force_refresh=False,
            selection=selection,
            lease_path=lease_path,
        )

    captured = capsys.readouterr()
    diagnostic = captured.err.lower().replace("_", "-")
    assert raised.value.code == 1
    for missing in ("resolved-config", "profile", "lease-ttl-s", "disk-min-gib"):
        assert missing in diagnostic
    assert not lease_path.exists()
    assert list((run_dir / "data").rglob("*.lock")) == []
    assert provider_calls == []


def _one_date_plan(run_dir: Path) -> Any:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True)
    cohort = SimpleNamespace(
        name="recorded",
        cfg=SimpleNamespace(paths=SimpleNamespace(data_dir=data_dir)),
    )
    return SimpleNamespace(
        cohorts=(cohort,),
        dates={"20240101": ["abc123"]},
        aircraft_days=1,
        requests_saved=lambda: (1, 1),
    )


def _publishing_fetch(calls: list[str], payload_path: Path) -> Any:
    def fetch(plan: Any, date: str, *, force: bool = False) -> Any:
        calls.append(date)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(f"payload-{len(calls)}".encode())
        return SimpleNamespace(
            date=date,
            requested=1,
            written=1,
            requests=1,
            empty_kinds=(),
            error=None,
        )

    return fetch


def _campaign_download(
    tmp_path: Path,
) -> tuple[Any, Path, FleetRunConfig, list[str], dict[str, Any]]:
    run_dir = tmp_path / "run"
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True)
    calls: list[str] = []
    fleet_config = FleetRunConfig(
        lease_path=shared_dir / "download-fleet.lease",
        lease_ttl_s=60,
        disk_min_gib=1.0,
    )
    kwargs = _download_kwargs(run_dir, fleet_config, calls)
    kwargs["fetch_boundary"] = _publishing_fetch(calls, run_dir / "raw-cache" / "payload.bin")
    return _one_date_plan(run_dir), run_dir, fleet_config, calls, kwargs


def _current_digest(kwargs: dict[str, Any]) -> Any:
    return compute_resume_digest(
        kwargs["selection_digest"],
        kwargs["resolved_config"],
        kwargs["profile"],
    )


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.integration
def test_download_fleet_rejects_each_mutated_identity_before_second_provider_call(
    tmp_path: Path,
) -> None:
    """AC1: every one-byte identity mutation is rejected before provider re-entry."""
    mutations: dict[str, DigestInput] = {
        "selection_digest": f"{_SELECTION_DIGEST}x",
        "resolved_config": {"workers": 1, "mode": "recorded-offlinf"},
        "profile": {"aircraft": "A320", "version": 2},
    }

    for dimension, mutated_value in mutations.items():
        plan, _run_dir, _fleet_config, calls, kwargs = _campaign_download(tmp_path / dimension)
        _fleet_fetch.download_fleet(plan, **kwargs)
        retry_kwargs = {**kwargs, dimension: mutated_value}
        retry_kwargs["recorded_digest"] = _current_digest(retry_kwargs)

        with pytest.raises(ResumeDigestMismatch):
            _fleet_fetch.download_fleet(plan, **retry_kwargs)

        assert calls == ["20240101"], dimension


@pytest.mark.integration
def test_download_fleet_mismatch_keeps_published_tree_byte_identical(
    tmp_path: Path,
) -> None:
    """AC2: rejection leaves raw-cache, manifests, and all campaign bytes unchanged."""
    plan, run_dir, _fleet_config, _calls, kwargs = _campaign_download(tmp_path)
    _fleet_fetch.download_fleet(plan, **kwargs)
    published_after_first_run = _snapshot_files(run_dir)

    with pytest.raises(ResumeDigestMismatch):
        _fleet_fetch.download_fleet(
            plan,
            **{
                **kwargs,
                "profile": {"aircraft": "A320", "version": 2},
                "recorded_digest": compute_resume_digest(
                    _SELECTION_DIGEST,
                    _RESOLVED_CONFIG,
                    {"aircraft": "A320", "version": 2},
                ),
            },
        )

    assert _snapshot_files(run_dir) == published_after_first_run


@pytest.mark.integration
def test_download_fleet_mismatch_creates_no_lease_record_or_fallback_lock(
    tmp_path: Path,
) -> None:
    """AC3: rejection occurs before acquisition records, leases, or fallback locks."""
    plan, run_dir, fleet_config, _calls, kwargs = _campaign_download(tmp_path)
    _fleet_fetch.download_fleet(plan, **kwargs)
    journal_path = run_dir / "download-fleet.journal.jsonl"
    events_before_retry = read_events(journal_path)
    lock_paths_before_retry = sorted(run_dir.rglob("*.lock"))
    assert fleet_config.lease_path is not None
    assert not fleet_config.lease_path.exists()

    with pytest.raises(ResumeDigestMismatch):
        _fleet_fetch.download_fleet(
            plan,
            **{
                **kwargs,
                "resolved_config": {"workers": 2, "mode": "recorded-offline"},
                "recorded_digest": compute_resume_digest(
                    _SELECTION_DIGEST,
                    {"workers": 2, "mode": "recorded-offline"},
                    _PROFILE,
                ),
            },
        )

    assert read_events(journal_path) == events_before_retry
    assert sorted(run_dir.rglob("*.lock")) == lock_paths_before_retry == []
    assert not fleet_config.lease_path.exists()


@pytest.mark.integration
def test_download_fleet_crash_identity_rejects_mutated_fresh_retry_without_provider(
    tmp_path: Path,
) -> None:
    """AC4: an identity survives a pre-provider crash and guards a fresh retry."""
    plan, run_dir, fleet_config, _calls, kwargs = _campaign_download(tmp_path)

    def crash_before_provider(plan: Any, date: str, *, force: bool = False) -> Any:
        raise RuntimeError("injected crash before provider")

    first_attempt = {**kwargs, "fetch_boundary": crash_before_provider}
    with pytest.raises(RuntimeError, match="injected crash"):
        _fleet_fetch.download_fleet(plan, **first_attempt)

    retry_calls: list[str] = []
    retry_kwargs = _download_kwargs(run_dir, fleet_config, retry_calls)
    retry_kwargs["profile"] = {"aircraft": "A320", "version": 2}
    retry_kwargs["recorded_digest"] = _current_digest(retry_kwargs)
    retry_kwargs["fetch_boundary"] = _publishing_fetch(
        retry_calls,
        run_dir / "raw-cache" / "payload.bin",
    )

    with pytest.raises(ResumeDigestMismatch):
        _fleet_fetch.download_fleet(plan, **retry_kwargs)

    assert retry_calls == []
    assert not (run_dir / "raw-cache").exists()
