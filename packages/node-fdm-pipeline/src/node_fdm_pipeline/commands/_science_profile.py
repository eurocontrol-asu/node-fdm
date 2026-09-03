from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "ScienceProfile",
    "ScienceProfileDigestMismatch",
    "UnknownScienceProfile",
    "profile_manifest",
    "resolve_science_profile",
    "verify_profile_artifacts",
]


class ScienceProfile(BaseModel, frozen=True):
    """Pinned identity of a campaign's scientific inputs."""

    profile_id: str
    source_commit: str
    artifact_digests: dict[str, str]


class UnknownScienceProfile(ValueError):  # noqa: N818
    """Raised when a requested scientific profile is not registered."""


class ScienceProfileDigestMismatch(ValueError):  # noqa: N818
    """Raised when a scientific artefact does not match its pinned digest."""


_OPEN_SKY_26_EXP_03_V1 = ScienceProfile(
    profile_id="opensky26-exp03-v1",
    source_commit="80e8039f85a59dc214630386ca637ea7a296cfa3",
    artifact_digests={
        "retained.csv": "2654c9b688669722271246868633dca0598cc345651e54feb1245affa35b58e1",
        "grids.json": "55a80354047a8d5c61f41921b4541f98699e9a158db5c427a99a9d6f5c288805",
        "metrics.yaml": "3c14d9a77705511296121f63bf2d957679b200d23e63b150a5f0d8ffc54a0a14",
    },
)
_PROFILES = {_OPEN_SKY_26_EXP_03_V1.profile_id: _OPEN_SKY_26_EXP_03_V1}


def resolve_science_profile(profile_id: str) -> ScienceProfile:
    """Resolve one explicitly requested scientific profile."""
    try:
        return _PROFILES[profile_id]
    except KeyError:
        msg = f"Unknown science profile: {profile_id}"
        raise UnknownScienceProfile(msg) from None


def profile_manifest(profile: ScienceProfile) -> dict[str, str]:
    """Render the flat provenance block embedded in published artefacts."""
    return {
        "profile_id": profile.profile_id,
        "source_commit": profile.source_commit,
        "retained_csv_sha256": profile.artifact_digests["retained.csv"],
        "grids_json_sha256": profile.artifact_digests["grids.json"],
        "metrics_yaml_sha256": profile.artifact_digests["metrics.yaml"],
    }


def verify_profile_artifacts(
    profile: ScienceProfile,
    files: Mapping[str, Path],
) -> None:
    """Verify every supplied local artefact against the profile's digest."""
    for artifact_name, path in files.items():
        expected_digest = profile.artifact_digests.get(artifact_name)
        if expected_digest is None:
            msg = f"Artifact {artifact_name!r} is absent from profile {profile.profile_id!r}"
            raise ScienceProfileDigestMismatch(msg)

        with path.open("rb") as artifact:
            actual_digest = hashlib.file_digest(artifact, "sha256").hexdigest()
        if actual_digest != expected_digest:
            msg = f"Digest mismatch for artifact {artifact_name!r}"
            raise ScienceProfileDigestMismatch(msg)
