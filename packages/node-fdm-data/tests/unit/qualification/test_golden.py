from __future__ import annotations

import pytest

from node_fdm_data import qualification


def test_compare_to_frozen_targets_reports_vertical_ten_thousandth_drift() -> None:
    """AC1: a 0.0001 vertical drift is a precisely reported mismatch."""
    coverage = qualification.QualificationCoverage(
        vertical_pct=73.9449,
        speed_pct=85.2183,
        profile_id="opensky26-exp03-v1",
    )
    targets = qualification.FrozenTargets(
        vertical_pct=73.9448,
        speed_pct=85.2183,
    )

    result = qualification.compare_to_frozen_targets(coverage, targets)

    assert result.matches is False
    assert result.vertical_delta == pytest.approx(0.0001, abs=1e-9)
    assert list(result.failing_channels) == ["vertical"]
