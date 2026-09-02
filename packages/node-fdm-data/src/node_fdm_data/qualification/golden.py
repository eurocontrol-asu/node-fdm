from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from node_fdm_data.qualification.evidence import load_qualification_evidence
from node_fdm_data.qualification.manifest import QualificationEvidenceError
from node_fdm_data.qualification.replay import (
    QualificationCoverage,
    replay_profile_coverage,
)

__all__ = [
    "FrozenComparisonResult",
    "FrozenTargets",
    "QualificationMismatchError",
    "assert_coverage",
    "assert_qualification",
    "compare_to_frozen_targets",
]

type _Channel = Literal["vertical", "speed"]


class FrozenTargets(BaseModel):
    """Digest-verified coverage targets parsed from frozen evidence."""

    model_config = ConfigDict(frozen=True)

    vertical_pct: float
    speed_pct: float


class FrozenComparisonResult(BaseModel):
    """Immutable comparison between replayed coverage and frozen targets."""

    model_config = ConfigDict(frozen=True)

    matches: bool
    vertical_delta: float
    speed_delta: float
    failing_channels: tuple[_Channel, ...]


class QualificationMismatchError(ValueError):
    """Raised when replayed coverage diverges from frozen evidence targets."""

    def __init__(
        self,
        result: FrozenComparisonResult,
        coverage: QualificationCoverage,
        targets: FrozenTargets,
    ) -> None:
        self.vertical_delta = result.vertical_delta
        self.speed_delta = result.speed_delta
        details: list[str] = []
        if "vertical" in result.failing_channels:
            details.append(
                f"vertical: expected {targets.vertical_pct:.4f}, "
                f"observed {coverage.vertical_pct:.4f}"
            )
        if "speed" in result.failing_channels:
            details.append(
                f"speed: expected {targets.speed_pct:.4f}, observed {coverage.speed_pct:.4f}"
            )
        super().__init__("Qualification coverage mismatch: " + "; ".join(details))


def compare_to_frozen_targets(
    coverage: QualificationCoverage,
    targets: FrozenTargets,
) -> FrozenComparisonResult:
    """Compare replayed percentages at the evidence's four-decimal precision."""
    vertical_delta = round(coverage.vertical_pct - targets.vertical_pct, 4)
    speed_delta = round(coverage.speed_pct - targets.speed_pct, 4)
    failing_channels: list[_Channel] = []
    if vertical_delta != 0.0:
        failing_channels.append("vertical")
    if speed_delta != 0.0:
        failing_channels.append("speed")
    return FrozenComparisonResult(
        matches=not failing_channels,
        vertical_delta=vertical_delta,
        speed_delta=speed_delta,
        failing_channels=tuple(failing_channels),
    )


def assert_coverage(
    coverage: QualificationCoverage,
    targets: FrozenTargets,
) -> FrozenComparisonResult:
    """Return the comparison or fail closed when any channel diverges."""
    result = compare_to_frozen_targets(coverage, targets)
    if not result.matches:
        raise QualificationMismatchError(result, coverage, targets)
    return result


def assert_qualification(profile_id: str) -> FrozenTargets:
    """Replay verified evidence and return its targets when coverage matches."""
    evidence = load_qualification_evidence(profile_id)
    targets = _targets_from_metrics(evidence.metrics)
    assert_coverage(replay_profile_coverage(profile_id), targets)
    return targets


def _targets_from_metrics(metrics: dict[str, JsonValue]) -> FrozenTargets:
    result = metrics.get("result")
    if not isinstance(result, dict):
        raise QualificationEvidenceError("metrics.yaml has no result mapping")
    if result.get("name") != "vertical_cascade_committed_coverage":
        raise QualificationEvidenceError("metrics.yaml has no vertical coverage result")
    vertical_pct = _percentage(result.get("value"), "vertical coverage")

    context = metrics.get("context")
    if not isinstance(context, list):
        raise QualificationEvidenceError("metrics.yaml has no context sequence")
    speed_value: JsonValue | None = None
    for item in context:
        if isinstance(item, dict) and item.get("name") == "speed_cascade_coverage_pct":
            speed_value = item.get("value")
            break
    speed_pct = _percentage(speed_value, "speed coverage")
    return FrozenTargets(vertical_pct=vertical_pct, speed_pct=speed_pct)


def _percentage(value: JsonValue | None, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationEvidenceError(f"metrics.yaml has no numeric {label}")
    percentage = float(value)
    if not math.isfinite(percentage):
        raise QualificationEvidenceError(f"metrics.yaml has a non-finite {label}")
    return percentage
