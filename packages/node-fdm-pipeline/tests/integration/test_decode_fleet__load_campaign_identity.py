"""Integration test for decode's shared campaign identity writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from node_fdm_pipeline.commands import _fleet_decode, _fleet_digest

pytestmark = pytest.mark.integration


def test_decode_writer_records_loadable_campaign_identity(tmp_path: Path) -> None:
    """AC4: decode's writer produces the identity consumed by the shared loader."""
    digest = _fleet_digest.compute_resume_digest(
        "decode-selection-v1",
        {"workers": 1, "mode": "offline"},
        {"aircraft": "A320", "revision": 1},
    )
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    identity_path = campaign_root / "decode-fleet.resume.json"

    _fleet_decode._record_resume_digest(identity_path, digest)

    # Read back from disk only: the loader must rebuild the identity from the
    # recorded file, never from in-process state.
    loaded = _fleet_digest.load_campaign_identity(campaign_root)
    assert isinstance(loaded, _fleet_digest.ResumeDigest)
    assert loaded.model_dump() == digest.model_dump()
