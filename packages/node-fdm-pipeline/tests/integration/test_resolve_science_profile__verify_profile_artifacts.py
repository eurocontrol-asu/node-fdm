from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.mark.integration
def test_verify_profile_artifacts_rejects_tampered_grids_json(tmp_path: Path) -> None:
    """AC4: verification names a supplied grids.json whose digest is incorrect."""
    science_profile = importlib.import_module("node_fdm_pipeline.commands._science_profile")
    grids_path = tmp_path / "grids.json"
    grids_path.write_bytes(b"tampered scientific grid")
    profile = science_profile.resolve_science_profile("opensky26-exp03-v1")

    with pytest.raises(science_profile.ScienceProfileDigestMismatch) as exc_info:
        science_profile.verify_profile_artifacts(profile, {"grids.json": grids_path})

    assert "grids.json" in str(exc_info.value)
