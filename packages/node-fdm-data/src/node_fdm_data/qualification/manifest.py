from __future__ import annotations

from types import MappingProxyType

from pydantic import BaseModel, ConfigDict

__all__ = [
    "EvidenceManifest",
    "QualificationEvidenceError",
    "resolve_evidence_manifest",
]


class QualificationEvidenceError(ValueError):
    """Raised when qualification evidence cannot be trusted or resolved."""


class EvidenceManifest(BaseModel):
    """Immutable content-addressed declaration for one evidence bundle."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    profile_id: str
    source_commit: str
    directory: str
    digests: MappingProxyType[str, str]


_OPEN_SKY_EXP03_V1 = EvidenceManifest(
    profile_id="opensky26-exp03-v1",
    source_commit="80e8039f85a59dc214630386ca637ea7a296cfa3",
    directory="qualification/opensky26_exp03_v1",
    digests=MappingProxyType(
        {
            "retained.csv": "2654c9b688669722271246868633dca0598cc345651e54feb1245affa35b58e1",
            "grids.json": "55a80354047a8d5c61f41921b4541f98699e9a158db5c427a99a9d6f5c288805",
            "metrics.yaml": "3c14d9a77705511296121f63bf2d957679b200d23e63b150a5f0d8ffc54a0a14",
        }
    ),
)
_MANIFESTS: MappingProxyType[str, EvidenceManifest] = MappingProxyType(
    {_OPEN_SKY_EXP03_V1.profile_id: _OPEN_SKY_EXP03_V1}
)


def resolve_evidence_manifest(profile_id: str) -> EvidenceManifest:
    """Resolve a frozen evidence manifest or report every known profile id."""
    try:
        return _MANIFESTS[profile_id]
    except KeyError as exc:
        known = ", ".join(sorted(_MANIFESTS))
        raise QualificationEvidenceError(
            f"Unknown qualification evidence profile {profile_id!r}; known profile ids: {known}"
        ) from exc
