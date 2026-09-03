from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

PROFILE_ID = "opensky26-exp03-v1"
SOURCE_COMMIT = "80e8039f85a59dc214630386ca637ea7a296cfa3"
ARTIFACT_DIGESTS = {
    "retained.csv": "2654c9b688669722271246868633dca0598cc345651e54feb1245affa35b58e1",
    "grids.json": "55a80354047a8d5c61f41921b4541f98699e9a158db5c427a99a9d6f5c288805",
    "metrics.yaml": "3c14d9a77705511296121f63bf2d957679b200d23e63b150a5f0d8ffc54a0a14",
}


def test_resolve_science_profile_returns_frozen_pinned_profile() -> None:
    """AC1: resolution returns the immutable, exactly pinned campaign profile."""
    science_profile = importlib.import_module("node_fdm_pipeline.commands._science_profile")

    profile = science_profile.resolve_science_profile(PROFILE_ID)

    assert isinstance(profile, science_profile.ScienceProfile)
    assert profile.source_commit == SOURCE_COMMIT
    assert profile.artifact_digests == ARTIFACT_DIGESTS
    with pytest.raises(ValidationError, match="frozen"):
        profile.profile_id = "opensky26-exp03-v2"


def test_resolve_science_profile_refuses_unknown_id() -> None:
    """AC2: an unknown profile is rejected instead of resolving a default."""
    science_profile = importlib.import_module("node_fdm_pipeline.commands._science_profile")
    requested_id = "opensky26-exp03-v2"

    with pytest.raises(science_profile.UnknownScienceProfile) as exc_info:
        science_profile.resolve_science_profile(requested_id)

    assert requested_id in str(exc_info.value)


def test_profile_manifest_contains_exact_pinned_provenance() -> None:
    """AC3: the manifest exposes exactly the five pinned provenance fields."""
    science_profile = importlib.import_module("node_fdm_pipeline.commands._science_profile")
    profile = science_profile.resolve_science_profile(PROFILE_ID)

    manifest = science_profile.profile_manifest(profile)

    assert set(manifest) == {
        "profile_id",
        "source_commit",
        "retained_csv_sha256",
        "grids_json_sha256",
        "metrics_yaml_sha256",
    }
    assert manifest == {
        "profile_id": PROFILE_ID,
        "source_commit": SOURCE_COMMIT,
        "retained_csv_sha256": ARTIFACT_DIGESTS["retained.csv"],
        "grids_json_sha256": ARTIFACT_DIGESTS["grids.json"],
        "metrics_yaml_sha256": ARTIFACT_DIGESTS["metrics.yaml"],
    }
