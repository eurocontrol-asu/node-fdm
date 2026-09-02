from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType

import pytest

_SELECTION_DIGEST = "ab" * 32
_RESOLVED_CONFIG = {
    "features": ["tas", "mach"],
    "paths": {"data_dir": Path("/tmp/cohort")},
    "threshold": 0.05,
}
_SCIENTIFIC_PROFILE = {
    "architecture": "node_adsb_v1",
    "constants": {"gravity": 9.80665},
}


def _digest_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_digest")


def test_resume_digest_is_stable_lowercase_sha256() -> None:
    """AC1: le digest est stable et chacun de ses champs est un SHA-256 hexadécimal."""
    digest_module = _digest_module()

    first = digest_module.compute_resume_digest(
        _SELECTION_DIGEST,
        _RESOLVED_CONFIG,
        _SCIENTIFIC_PROFILE,
    )
    second = digest_module.compute_resume_digest(
        _SELECTION_DIGEST,
        _RESOLVED_CONFIG,
        _SCIENTIFIC_PROFILE,
    )

    assert isinstance(first, digest_module.ResumeDigest)
    assert first == second
    for field_name in ("selection", "config", "profile", "composite"):
        assert re.fullmatch(r"[0-9a-f]{64}", getattr(first, field_name))


def test_each_component_byte_changes_its_digest_and_composite() -> None:
    """AC2: muter un octet de chaque entrée change son composant et le composite."""
    digest_module = _digest_module()
    baseline = digest_module.compute_resume_digest(
        _SELECTION_DIGEST,
        _RESOLVED_CONFIG,
        _SCIENTIFIC_PROFILE,
    )
    changed_config = {**_RESOLVED_CONFIG, "threshold": 0.06}
    changed_profile = {**_SCIENTIFIC_PROFILE, "architecture": "node_adsb_v2"}
    cases = {
        "selection": (
            f"{_SELECTION_DIGEST[:-2]}ac",
            _RESOLVED_CONFIG,
            _SCIENTIFIC_PROFILE,
        ),
        "config": (_SELECTION_DIGEST, changed_config, _SCIENTIFIC_PROFILE),
        "profile": (_SELECTION_DIGEST, _RESOLVED_CONFIG, changed_profile),
    }

    for component, inputs in cases.items():
        changed = digest_module.compute_resume_digest(*inputs)

        assert getattr(changed, component) != getattr(baseline, component)
        assert changed.composite != baseline.composite
        for unchanged_component in {"selection", "config", "profile"} - {component}:
            assert getattr(changed, unchanged_component) == getattr(
                baseline,
                unchanged_component,
            )


def test_resume_guard_names_each_divergent_component() -> None:
    """AC3: la garde nomme exactement le composant selection, config ou profile."""
    digest_module = _digest_module()
    recorded = digest_module.compute_resume_digest(
        _SELECTION_DIGEST,
        _RESOLVED_CONFIG,
        _SCIENTIFIC_PROFILE,
    )
    changed_config = {**_RESOLVED_CONFIG, "threshold": 0.06}
    changed_profile = {**_SCIENTIFIC_PROFILE, "architecture": "node_adsb_v2"}
    cases = {
        "selection": (
            f"{_SELECTION_DIGEST[:-2]}ac",
            _RESOLVED_CONFIG,
            _SCIENTIFIC_PROFILE,
        ),
        "config": (_SELECTION_DIGEST, changed_config, _SCIENTIFIC_PROFILE),
        "profile": (_SELECTION_DIGEST, _RESOLVED_CONFIG, changed_profile),
    }

    for component, inputs in cases.items():
        current = digest_module.compute_resume_digest(*inputs)

        with pytest.raises(
            digest_module.ResumeDigestMismatch,
            match=rf"\b{component}\b",
        ) as raised:
            digest_module.check_resume_compatible(recorded, current)

        assert component in str(raised.value)


def test_resume_guard_reports_first_divergence_in_deterministic_order() -> None:
    """AC3: avec config et profile divergents, la garde s'arrête sur config."""
    digest_module = _digest_module()
    recorded = digest_module.compute_resume_digest(
        _SELECTION_DIGEST,
        _RESOLVED_CONFIG,
        _SCIENTIFIC_PROFILE,
    )
    current = digest_module.compute_resume_digest(
        _SELECTION_DIGEST,
        {**_RESOLVED_CONFIG, "threshold": 0.06},
        {**_SCIENTIFIC_PROFILE, "architecture": "node_adsb_v2"},
    )

    with pytest.raises(
        digest_module.ResumeDigestMismatch,
        match=r"\bconfig\b",
    ) as raised:
        digest_module.check_resume_compatible(recorded, current)

    assert "profile" not in str(raised.value)
