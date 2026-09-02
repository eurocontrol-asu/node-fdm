"""Tests unitaires du replay de qualification."""

from __future__ import annotations

import polars as pl

import node_fdm_data as node_fdm_data_module


def test_coverage_over_in_memory_frame_is_rounded_selected_ratio() -> None:
    """AC2: trois lignes verticales valides sur sept donnent 42,8571 %."""
    profile = node_fdm_data_module.load_profile("opensky26-exp03-v1")
    per_row_pct = 100.0 / 7.0
    frame = pl.DataFrame(
        {
            "channel": ["alt", "gamma", "vz", "cas", "mach", "other-a", "other-b"],
            "admissible": [True] * 7,
            "coverage_pct": [per_row_pct] * 7,
        }
    )

    result = node_fdm_data_module.replay_coverage_from_frame(frame, profile)

    assert isinstance(result, node_fdm_data_module.QualificationCoverage)
    assert result.vertical_pct == 42.8571
