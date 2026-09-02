from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "ResumeDigest",
    "ResumeDigestMismatch",
    "check_resume_compatible",
    "compute_resume_digest",
    "load_campaign_identity",
    "record_campaign_identity",
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


_CAMPAIGN_IDENTITY_NAME = "decode-fleet.resume.json"


def record_campaign_identity(root: Path, digest: ResumeDigest) -> None:
    """Atomically persist a campaign identity under its campaign root."""
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / _CAMPAIGN_IDENTITY_NAME
    descriptor, raw_path = tempfile.mkstemp(
        dir=root,
        prefix=f".{identity_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(digest.model_dump_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, identity_path)
        directory_descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_campaign_identity(root: Path) -> ResumeDigest:
    """Load a campaign identity exclusively from its durable record."""
    identity_path = root / _CAMPAIGN_IDENTITY_NAME
    return ResumeDigest.model_validate_json(identity_path.read_text(encoding="utf-8"))


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
