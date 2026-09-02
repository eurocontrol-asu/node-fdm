from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from node_fdm_pipeline.commands._fleet_selection import SelectedFlight, SelectionPlan

__all__ = ["MatchRejection", "MatchResult", "match_selections"]

type Rotation = Mapping[str, object]


class MatchRejection(BaseModel):
    """Structured explanation for a selection that could not be matched uniquely."""

    model_config = ConfigDict(frozen=True)

    selection_id: str
    kind: Literal["absent", "ambiguous"]
    candidates: tuple[Rotation, ...] = ()


class MatchResult(BaseModel):
    """Immutable outcome of matching a selection plan to decoded rotations."""

    model_config = ConfigDict(frozen=True)

    admitted: tuple[Rotation, ...]
    rejections: Mapping[str, MatchRejection]


def _required_string(rotation: Rotation, field: str) -> str:
    value = rotation.get(field)
    if not isinstance(value, str):
        raise TypeError(f"rotation field {field!r} must be a string")
    return value


def _required_timestamp(rotation: Rotation, field: str) -> int:
    value = rotation.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"rotation field {field!r} must be an integer timestamp")
    return value


def _normalized_callsign(value: str) -> str:
    return value.strip().casefold()


def _overlaps(selection: SelectedFlight, rotation: Rotation) -> bool:
    rotation_firstseen = _required_timestamp(rotation, "firstseen")
    rotation_lastseen = _required_timestamp(rotation, "lastseen")
    return rotation_firstseen <= selection.lastseen and selection.firstseen <= rotation_lastseen


def _matches(selection: SelectedFlight, rotation: Rotation) -> bool:
    return (
        _required_string(rotation, "icao24") == selection.icao24
        and _normalized_callsign(_required_string(rotation, "callsign"))
        == _normalized_callsign(selection.callsign)
        and _overlaps(selection, rotation)
    )


def match_selections(plan: SelectionPlan, rotations: Iterable[Rotation]) -> MatchResult:
    """Match every selected flight to exactly one decoded rotation."""
    available = tuple(rotations)
    admitted: list[Rotation] = []
    rejections: dict[str, MatchRejection] = {}

    for selection in plan.flights:
        candidates = tuple(rotation for rotation in available if _matches(selection, rotation))
        selection_id = selection.acquisition_key
        if len(candidates) == 1:
            admitted.append(candidates[0])
            continue

        kind: Literal["absent", "ambiguous"] = "absent" if not candidates else "ambiguous"
        rejections[selection_id] = MatchRejection(
            selection_id=selection_id,
            kind=kind,
            candidates=candidates,
        )

    return MatchResult(admitted=tuple(admitted), rejections=rejections)
