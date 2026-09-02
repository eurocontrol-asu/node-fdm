from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "ResumeDigest",
    "ResumeDigestMismatch",
    "check_resume_compatible",
    "compute_resume_digest",
]

type CanonicalScalar = None | bool | int | float | str
type CanonicalJson = CanonicalScalar | list[CanonicalJson] | dict[str, CanonicalJson]
type DigestInput = (
    CanonicalScalar | Path | BaseModel | Mapping[str, DigestInput] | Sequence[DigestInput]
)


class ResumeDigest(BaseModel, frozen=True):
    """Content digests that make a compiled fleet plan safe to resume."""

    selection: str
    config: str
    profile: str
    composite: str


class ResumeDigestMismatch(RuntimeError):  # noqa: N818
    """Raised when resume inputs differ from those used to compile the plan."""


def _normalize(value: DigestInput) -> CanonicalJson:
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python"))
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_normalize(item) for item in value]

    msg = f"Unsupported resume digest value: {type(value).__qualname__}"
    raise TypeError(msg)


def _digest(value: DigestInput) -> str:
    canonical = json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_resume_digest(
    selection_digest: str,
    resolved_config: DigestInput,
    profile: DigestInput,
) -> ResumeDigest:
    """Compute stable component and composite digests for resume inputs."""
    selection = _digest(selection_digest)
    config = _digest(resolved_config)
    profile_digest = _digest(profile)
    composite = _digest(
        {
            "selection": selection,
            "config": config,
            "profile": profile_digest,
        }
    )
    return ResumeDigest(
        selection=selection,
        config=config,
        profile=profile_digest,
        composite=composite,
    )


def check_resume_compatible(recorded: ResumeDigest, current: ResumeDigest) -> None:
    """Reject resume inputs at the first component that has changed."""
    components = (
        ("selection", recorded.selection, current.selection),
        ("config", recorded.config, current.config),
        ("profile", recorded.profile, current.profile),
    )
    for component, recorded_value, current_value in components:
        if recorded_value != current_value:
            msg = f"Resume digest mismatch: {component} changed"
            raise ResumeDigestMismatch(msg)
