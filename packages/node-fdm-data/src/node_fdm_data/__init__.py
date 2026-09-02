"""node-fdm-data — Flight data processing, physics, conversions, and schemas."""

from __future__ import annotations

from node_fdm_data.profiles import SegmentProfile, list_profiles, load_profile
from node_fdm_data.qualification import (
    QualificationEvidenceError,
    load_qualification_evidence,
    resolve_evidence_manifest,
    verify_digests,
)
from node_fdm_data.segments import (
    blank_frozen_endpoints,
    legacy_selected_params_cfg,
    selected_params_cfg_from_profile,
    selected_params_coverage,
)

__all__ = [
    "QualificationEvidenceError",
    "SegmentProfile",
    "blank_frozen_endpoints",
    "conversions",
    "delta",
    "lateral",
    "legacy_selected_params_cfg",
    "list_profiles",
    "load_profile",
    "load_qualification_evidence",
    "meteo",
    "physics",
    "preprocessing",
    "processor",
    "resolve_evidence_manifest",
    "schema",
    "schemas",
    "segments",
    "selected_params_cfg_from_profile",
    "selected_params_coverage",
    "split",
    "verify_digests",
]
