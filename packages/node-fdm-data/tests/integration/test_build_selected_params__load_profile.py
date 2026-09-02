"""Oracle d'intégration figé pour le profil OpenSky exp03."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import polars as pl
import pytest

import node_fdm_data.segments as segments_module
from node_fdm_data import load_profile
from node_fdm_data.segments import build_selected_params

pytestmark = pytest.mark.integration

_PROFILE = "opensky26-exp03-v1"
_FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "opensky26_exp03_v1"
_COLUMNS = {
    "alt": "fdm_alt_sel_ft",
    "gamma": "fdm_gamma_sel_rad",
    "vz": "fdm_vz_sel_ftmin",
    "mach": "fdm_mach_sel",
    "cas": "fdm_cas_sel_kt",
}


class SegmentExpectation(TypedDict):
    start_idx: int
    end_idx: int
    values: list[float]


class PerturbationExpectation(TypedDict):
    altitude_step_index: int
    retained_mach_bounds: list[int]


class GoldenExpectation(TypedDict):
    perturbations: PerturbationExpectation
    segments: dict[str, list[SegmentExpectation]]
    coverage: dict[str, float]


def _load_inputs() -> tuple[pl.DataFrame, GoldenExpectation]:
    frame = pl.read_csv(_FIXTURE_DIR / "flight.csv")
    golden = cast(
        GoldenExpectation,
        json.loads((_FIXTURE_DIR / "expected_segments.json").read_text()),
    )
    return frame, golden


def _canonical_segments(frame: pl.DataFrame) -> dict[str, list[dict[str, object]]]:
    canonical: dict[str, list[dict[str, object]]] = {}
    for family, column in _COLUMNS.items():
        values = frame[column].to_numpy()
        finite = np.isfinite(values)
        runs: list[dict[str, object]] = []
        start: int | None = None
        for index, present in enumerate(finite):
            if present and start is None:
                start = index
            at_end = index == len(values) - 1
            if start is not None and ((not present) or at_end):
                end = index if present and at_end else index - 1
                rounded = sorted({round(float(value), 6) for value in values[start : end + 1]})
                runs.append({"start_idx": start, "end_idx": end, "values": rounded})
                start = None
        canonical[family] = runs
    return canonical


def _reproduce(frame: pl.DataFrame) -> pl.DataFrame:
    profile = load_profile(_PROFILE)
    return build_selected_params(frame, profile=profile)


def test_frozen_fixture_reproduces_golden_five_channel_segments() -> None:
    """AC1: la fixture reproduit exactement bornes et valeurs des cinq canaux."""
    frame, golden = _load_inputs()

    result = _reproduce(frame)

    assert _canonical_segments(result) == golden["segments"]


def test_reproduction_never_imports_papers_subproject() -> None:
    """AC2: la reproduction n'importe ni paper_opensky26 ni un module papers."""
    assert "paper_opensky26" not in sys.modules
    before = set(sys.modules)
    frame, _ = _load_inputs()

    _reproduce(frame)

    assert "paper_opensky26" not in sys.modules
    imported = set(sys.modules) - before
    for name in imported:
        if not name.startswith("node_fdm_data"):
            continue
        module_file = str(getattr(sys.modules[name], "__file__", ""))
        assert "/papers/" not in module_file.replace("\\", "/")


def test_altitude_step_changes_vertical_only() -> None:
    """AC3: un pas d'altitude change le vertical, jamais Mach ni CAS."""
    frame, golden = _load_inputs()
    altitude = frame["raw_alt_ft"].to_numpy().copy()
    altitude[golden["perturbations"]["altitude_step_index"]] += 200.0
    perturbed = frame.with_columns(pl.Series("raw_alt_ft", altitude))

    actual = _canonical_segments(_reproduce(perturbed))

    assert any(actual[name] != golden["segments"][name] for name in ("alt", "gamma", "vz"))
    assert actual["mach"] == golden["segments"]["mach"]
    assert actual["cas"] == golden["segments"]["cas"]


def test_rejecting_retained_mach_plateau_changes_cas() -> None:
    """AC4: rejeter le plateau Mach retenu modifie la cascade CAS."""
    frame, golden = _load_inputs()
    mach = frame["bds_mach_clean"].to_numpy().copy()
    start, end = golden["perturbations"]["retained_mach_bounds"]
    mach[start : end + 1] = np.linspace(0.55, 0.90, end - start + 1)
    perturbed = frame.with_columns(pl.Series("bds_mach_clean", mach))

    result = _reproduce(perturbed)

    assert _canonical_segments(result)["cas"] != golden["segments"]["cas"]


def test_fixture_and_in_memory_coverage_match_contract() -> None:
    """AC5: couverture golden arrondie et verticale 50 % sur dix lignes."""
    frame, golden = _load_inputs()
    result = _reproduce(frame)

    fixture_coverage = segments_module.selected_params_coverage(result)

    assert fixture_coverage == golden["coverage"]
    assert fixture_coverage == {
        family: round(value, 4) for family, value in fixture_coverage.items()
    }

    nan = float("nan")
    in_memory = pl.DataFrame(
        {
            "fdm_alt_sel_ft": [1000.0] * 5 + [nan] * 5,
            "fdm_gamma_sel_rad": [nan] * 10,
            "fdm_vz_sel_ftmin": [nan] * 10,
            "fdm_mach_sel": [nan] * 10,
            "fdm_cas_sel_kt": [nan] * 10,
        }
    )
    coverage = segments_module.selected_params_coverage(in_memory)
    assert coverage["vertical"] == 50.0
