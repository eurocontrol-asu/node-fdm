"""Replay déterministe des couvertures de qualification empaquetées."""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from node_fdm_data.profiles import SegmentProfile, list_profiles, load_profile
from node_fdm_data.qualification.evidence import load_qualification_evidence
from node_fdm_data.segments import (
    build_selected_params,
    selected_params_cfg_from_profile,
    selected_params_coverage,
)

__all__ = [
    "QualificationCoverage",
    "replay_coverage_from_frame",
    "replay_profile_coverage",
]

_RETAINED_COLUMNS = frozenset({"channel", "admissible", "coverage_pct"})
_SELECTED_COLUMNS = frozenset(
    {
        "fdm_alt_sel_ft",
        "fdm_gamma_sel_rad",
        "fdm_vz_sel_ftmin",
        "fdm_mach_sel",
        "fdm_cas_sel_kt",
    }
)


class QualificationCoverage(BaseModel):
    """Couvertures immuables produites par un replay de qualification."""

    model_config = ConfigDict(frozen=True)

    vertical_pct: float
    speed_pct: float
    profile_id: str


def _registered_profile_id(profile: SegmentProfile) -> str:
    for profile_id in list_profiles():
        if load_profile(profile_id) == profile:
            return profile_id
    raise ValueError("Qualification replay requires a registered segment profile")


def _retained_family_coverage(
    frame: pl.DataFrame,
    channels: tuple[str, ...],
    configured_channels: frozenset[str],
) -> float:
    eligible_channels = tuple(channel for channel in channels if channel in configured_channels)
    total = (
        frame.filter(
            pl.col("admissible").fill_null(False) & pl.col("channel").is_in(eligible_channels)
        )
        .select(pl.col("coverage_pct").sum())
        .item()
    )
    return 0.0 if total is None else float(total)


def _coverage_from_retained(
    frame: pl.DataFrame,
    profile: SegmentProfile,
) -> dict[str, float]:
    configured_channels = frozenset(selected_params_cfg_from_profile(profile))
    return {
        "vertical": _retained_family_coverage(
            frame,
            profile.vertical_cascade,
            configured_channels,
        ),
        "speed": _retained_family_coverage(
            frame,
            profile.speed_cascade,
            configured_channels,
        ),
    }


def replay_coverage_from_frame(
    frame: pl.DataFrame,
    profile: SegmentProfile,
) -> QualificationCoverage:
    """Compute qualification coverage from evidence or an in-memory flight frame."""
    columns = frozenset(frame.columns)
    if _RETAINED_COLUMNS <= columns:
        coverage = _coverage_from_retained(frame, profile)
    elif columns & _SELECTED_COLUMNS:
        coverage = selected_params_coverage(frame)
    else:
        selected = build_selected_params(frame, profile=profile)
        coverage = selected_params_coverage(selected)

    return QualificationCoverage(
        vertical_pct=round(coverage["vertical"], 4),
        speed_pct=round(coverage["speed"], 4),
        profile_id=_registered_profile_id(profile),
    )


def replay_profile_coverage(profile_id: str) -> QualificationCoverage:
    """Replay coverage from the verified evidence bundled for a profile."""
    evidence = load_qualification_evidence(profile_id)
    profile = load_profile(profile_id)
    return replay_coverage_from_frame(evidence.retained, profile)
