"""Unit tests for pure selection-to-rotation matching."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import ModuleType

from node_fdm_pipeline.commands import _fleet_selection


def _load_selection_match() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._selection_match")


def _plan(
    icao24: str,
    callsign: str,
    firstseen: int,
    lastseen: int,
) -> _fleet_selection.SelectionPlan:
    return _fleet_selection.compile_selection(
        [
            {
                "icao24": icao24,
                "callsign": callsign,
                "firstseen": firstseen,
                "lastseen": lastseen,
                "msn": "MSN-1",
                "split": "train",
                "cohort": "cohort-1",
                "day": "2020-01-01",
            }
        ]
    )


def _rotation(
    rotation_id: str,
    icao24: str,
    callsign: str,
    firstseen: int,
    lastseen: int,
) -> Mapping[str, object]:
    return {
        "rotation_id": rotation_id,
        "icao24": icao24,
        "callsign": callsign,
        "firstseen": firstseen,
        "lastseen": lastseen,
    }


def test_match_selections_admits_only_the_overlapping_rotation() -> None:
    """AC1: another rotation of the same aircraft is never silently admitted."""
    selection_match = _load_selection_match()
    plan = _plan("a001", "FLT1", 100, 200)
    overlapping = _rotation("rotation-overlap", "a001", "FLT1", 120, 180)
    later = _rotation("rotation-later", "a001", "FLT1", 300, 400)

    result = selection_match.match_selections(plan, [overlapping, later])

    assert result.admitted == (overlapping,)
    assert later not in result.admitted


def test_match_selections_normalizes_callsign_case_and_padding() -> None:
    """AC2: callsigns match despite case and surrounding padding differences."""
    selection_match = _load_selection_match()
    plan = _plan("a001", "afr123", 100, 200)
    padded = _rotation("rotation-normalized", "a001", " AFR123 ", 120, 180)

    result = selection_match.match_selections(plan, [padded])

    assert result.admitted == (padded,)
    assert result.rejections == {}


def test_match_selections_indexes_absent_rejection_by_selection_id() -> None:
    """AC3: a selection without a candidate has an indexed absent rejection."""
    selection_match = _load_selection_match()
    plan = _plan("a002", "FLT2", 100, 200)
    selection_id = plan.flights[0].acquisition_key
    unrelated = _rotation("rotation-unrelated", "a999", "FLT2", 120, 180)

    result = selection_match.match_selections(plan, [unrelated])

    assert result.admitted == ()
    assert tuple(result.rejections) == (selection_id,)
    assert result.rejections[selection_id].kind == "absent"


def test_match_selections_rejects_all_ambiguous_candidates() -> None:
    """AC4: two overlapping candidates are listed as ambiguous and none is admitted."""
    selection_match = _load_selection_match()
    plan = _plan("a003", "FLT3", 100, 200)
    selection_id = plan.flights[0].acquisition_key
    first = _rotation("rotation-first", "a003", "FLT3", 110, 150)
    second = _rotation("rotation-second", "a003", "FLT3", 140, 220)

    result = selection_match.match_selections(plan, [first, second])

    assert result.admitted == ()
    rejection = result.rejections[selection_id]
    assert rejection.kind == "ambiguous"
    assert rejection.candidates == (first, second)


def test_match_selections_keeps_cross_midnight_rotation_whole() -> None:
    """AC5: an absolute interval crossing UTC midnight remains one admitted rotation."""
    selection_match = _load_selection_match()
    firstseen = 1_577_921_400
    lastseen = 1_577_925_000
    plan = _plan("a004", "FLT4", firstseen, lastseen)
    cross_midnight = _rotation(
        "rotation-cross-midnight",
        "a004",
        "FLT4",
        firstseen,
        lastseen,
    )

    result = selection_match.match_selections(plan, [cross_midnight])

    assert result.admitted == (cross_midnight,)
    assert len(result.admitted) == 1
    assert (result.admitted[0]["firstseen"], result.admitted[0]["lastseen"]) == (
        firstseen,
        lastseen,
    )
