"""Integration tests for the durable campaign identity record."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType

import pytest

from node_fdm_pipeline.commands import _fleet_digest

pytestmark = pytest.mark.integration

_SELECTION_DIGEST = "campaign-selection-v1"
_RESOLVED_CONFIG: dict[str, _fleet_digest.DigestInput] = {
    "workers": 2,
    "mode": "offline",
}
_PROFILE: dict[str, _fleet_digest.DigestInput] = {
    "aircraft": "A320",
    "revision": 1,
}


def _digest_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_digest")


def _campaign_digest(*, profile_revision: int = 1) -> _fleet_digest.ResumeDigest:
    return _fleet_digest.compute_resume_digest(
        _SELECTION_DIGEST,
        _RESOLVED_CONFIG,
        {**_PROFILE, "revision": profile_revision},
    )


def test_recorded_identity_survives_disk_only_reload(tmp_path: Path) -> None:
    """AC1: a disk-only reload restores selection, resolved config, and profile."""
    digest_module = _digest_module()
    record = digest_module.record_campaign_identity
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    digest = _campaign_digest()
    expected_dimensions = (digest.selection, digest.config, digest.profile)

    record(campaign_root, digest)
    del digest
    digest_module = importlib.reload(_fleet_digest)

    loaded = digest_module.load_campaign_identity(campaign_root)

    assert isinstance(loaded, digest_module.ResumeDigest)
    assert (loaded.selection, loaded.config, loaded.profile) == expected_dimensions


def test_interrupted_overwrite_preserves_original_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2: a failure before atomic rename leaves the original identity readable."""
    digest_module = _digest_module()
    record = digest_module.record_campaign_identity
    load = digest_module.load_campaign_identity
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    original = _campaign_digest()
    mutated = _campaign_digest(profile_revision=2)
    original_dimensions = (original.selection, original.config, original.profile)
    record(campaign_root, original)

    def interrupt_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("injected failure before atomic rename")

    monkeypatch.setattr(os, "replace", interrupt_replace)

    with pytest.raises(OSError, match="injected failure before atomic rename"):
        record(campaign_root, mutated)

    digest_module = importlib.reload(_fleet_digest)
    loaded = load(campaign_root)
    assert (loaded.selection, loaded.config, loaded.profile) == original_dimensions


def test_successful_record_leaves_one_authoritative_file(tmp_path: Path) -> None:
    """AC3: successful replacement leaves one identity file and no temporary residue."""
    digest_module = _digest_module()
    record = digest_module.record_campaign_identity
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()

    record(campaign_root, _campaign_digest())
    record(campaign_root, _campaign_digest(profile_revision=2))

    entries = list(campaign_root.iterdir())
    assert len(entries) == 1
    assert entries[0].is_file()
    assert ".tmp" not in entries[0].name
