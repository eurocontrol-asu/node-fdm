"""Smoke test for Paper 0 QAR target validation pipeline.

Loads one parquet from the QAR3 cohort, runs ``process_flight``, and asserts
the returned dict carries the expected keys, shapes, and non-zero coverage on
at least the altitude channel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts/_validation to import path so the test can pull `process_flight`.
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "scripts" / "_validation"))

import paper0_qar_target_validation as p0  # noqa: E402

QAR_DIR = Path("/Users/gabriel/Downloads/QAR3")


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qar_parquet_path() -> Path:
    paths = sorted(QAR_DIR.glob("*A320*.parquet"))
    if not paths:
        pytest.skip(f"No QAR3 A320 parquets at {QAR_DIR}")
    return paths[0]


def test_process_flight_returns_expected_keys(qar_parquet_path: Path) -> None:
    result = p0.process_flight(qar_parquet_path)
    assert result is not None, "process_flight returned None on a valid parquet"
    expected_keys = {
        "flight_id",
        "n_samples",
        "alt_pred_m",
        "alt_truth_m",
        "alt_known",
        "mach_pred",
        "mach_truth",
        "mach_known",
        "cas_pred_kt",
        "cas_truth_kt",
        "cas_known",
        "gamma_pred_rad",
        "gamma_truth_rad",
        "gamma_known",
        "track_pred_deg",
        "track_truth_deg",
        "track_known",
        "in_turn",
        "phase",
    }
    missing = expected_keys - result.keys()
    assert not missing, f"missing keys: {missing}"


def test_aligned_arrays_have_matching_lengths(qar_parquet_path: Path) -> None:
    result = p0.process_flight(qar_parquet_path)
    assert result is not None
    n = result["n_samples"]
    paired = (
        ("alt_pred_m", "alt_truth_m"),
        ("mach_pred", "mach_truth"),
        ("cas_pred_kt", "cas_truth_kt"),
        ("gamma_pred_rad", "gamma_truth_rad"),
        ("track_pred_deg", "track_truth_deg"),
    )
    for pred_key, truth_key in paired:
        assert result[pred_key].shape == (n,), pred_key
        assert result[truth_key].shape == (n,), truth_key


def test_altitude_channel_has_nonzero_coverage(qar_parquet_path: Path) -> None:
    result = p0.process_flight(qar_parquet_path)
    assert result is not None
    eligible = (
        result["alt_known"].astype(bool)
        & np.isfinite(result["alt_pred_m"])
        & np.isfinite(result["alt_truth_m"])
    )
    coverage = eligible.mean()
    assert coverage > 0.0, f"altitude coverage is 0 (n={result['n_samples']})"
