from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import cast

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, JsonValue

from node_fdm_data.qualification.manifest import (
    QualificationEvidenceError,
    resolve_evidence_manifest,
)

__all__ = [
    "EvidenceProvenance",
    "QualificationEvidence",
    "load_qualification_evidence",
    "verify_digests",
]


class EvidenceProvenance(BaseModel):
    """Immutable origin of a qualification evidence bundle."""

    model_config = ConfigDict(frozen=True)

    source_commit: str


class QualificationEvidence(BaseModel):
    """Verified and parsed qualification evidence shipped with the package."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    retained: pl.DataFrame
    grids: dict[str, JsonValue]
    metrics: dict[str, JsonValue]
    provenance: EvidenceProvenance
    digests: MappingProxyType[str, str]
    evidence_root: Path


def verify_digests(
    directory: Path,
    expected: Mapping[str, str],
) -> MappingProxyType[str, str]:
    """Recompute every expected digest and fail closed on the first mismatch."""
    observed_digests: dict[str, str] = {}
    for filename, expected_digest in expected.items():
        evidence_file = directory / filename
        observed_digest = (
            hashlib.sha256(evidence_file.read_bytes()).hexdigest()
            if evidence_file.is_file()
            else "<missing>"
        )
        if observed_digest != expected_digest:
            raise QualificationEvidenceError(
                f"Digest mismatch for {filename}: expected {expected_digest}, "
                f"observed {observed_digest}"
            )
        observed_digests[filename] = observed_digest
    return MappingProxyType(observed_digests)


def load_qualification_evidence(profile_id: str) -> QualificationEvidence:
    """Load package evidence only after its committed bytes pass verification."""
    manifest = resolve_evidence_manifest(profile_id)
    package_resource = files("node_fdm_data")
    evidence_root = Path(str(package_resource.joinpath(*manifest.directory.split("/"))))
    digests = verify_digests(evidence_root, manifest.digests)
    grids = cast(
        "dict[str, JsonValue]",
        json.loads((evidence_root / "grids.json").read_text(encoding="utf-8")),
    )
    metrics = cast(
        "dict[str, JsonValue]",
        yaml.safe_load((evidence_root / "metrics.yaml").read_text(encoding="utf-8")),
    )
    return QualificationEvidence(
        retained=pl.read_csv(evidence_root / "retained.csv"),
        grids=grids,
        metrics=metrics,
        provenance=EvidenceProvenance(source_commit=manifest.source_commit),
        digests=digests,
        evidence_root=evidence_root,
    )
