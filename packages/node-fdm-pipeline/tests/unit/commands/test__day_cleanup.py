"""Unit contracts for durable day cleanup decisions."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def _cleanup() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_cleanup")


def test_zero_reference_artifacts_selects_only_zero_counts() -> None:
    """AC5: only artefacts with no future reference are eligible for deletion."""
    counters = {
        "raw/20200101/A20N": 0,
        "era5/20200101/t2m": 2,
        "raw/20191231/A20N": 0,
    }

    selected = _cleanup().zero_reference_artifacts(counters)

    assert selected == ("raw/20191231/A20N", "raw/20200101/A20N")
    assert "era5/20200101/t2m" not in selected


class _MemoryArtifact:
    def __init__(self) -> None:
        self.present = True

    def exists(self) -> bool:
        return self.present

    def unlink(self) -> None:
        self.present = False


def test_cleanup_reports_per_artifact_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC1: cleanup reports decrement and delete once for each eligible artefact."""
    cleanup = _cleanup()
    counters = {
        "raw/referenced": 2,
        "era5/referenced": 3,
        "raw/zero": 0,
    }
    artifacts = {artifact_id: _MemoryArtifact() for artifact_id in counters}
    boundaries: list[tuple[str, str]] = []

    monkeypatch.setattr(cleanup, "assert_purge_allowed", lambda _snapshot: None)
    monkeypatch.setattr(cleanup, "_events", lambda _path: [])
    monkeypatch.setattr(cleanup, "_read_counters", lambda _path: counters)
    monkeypatch.setattr(cleanup, "_write_counters", lambda _path, _counters: None)
    monkeypatch.setattr(cleanup, "append_event", lambda _path, _event: None)

    outcome = cleanup.cleanup_day(
        type(
            "CommittedSnapshot",
            (),
            {
                "meta_selection_day": "20200101",
                "published_keys": (),
                "day_committed": object(),
            },
        )(),
        journal_path=object(),
        counters_path=object(),
        artifacts=artifacts,
        decrements=("raw/referenced", "era5/referenced"),
        step_hook=lambda step, artifact_id: boundaries.append((step, artifact_id)),
    )

    assert outcome.state == "cleaned"
    assert boundaries == [
        ("decrement", "raw/referenced"),
        ("decrement", "era5/referenced"),
        ("delete", "raw/zero"),
    ]
