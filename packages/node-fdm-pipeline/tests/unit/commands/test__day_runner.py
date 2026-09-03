from __future__ import annotations

import importlib

import pytest


def test_previous_cleanup_failure_blocks_day_admission() -> None:
    """AC1: the last failed cleanup state names the predecessor that blocks admission."""
    day_runner = importlib.import_module("node_fdm_pipeline.commands._day_runner")
    snapshot = (
        {"event": "cleanup_completed", "day": "20200101"},
        {"event": "cleanup_failed", "day": "20200101", "error": "injected cleanup fault"},
    )

    with pytest.raises(day_runner.PreviousDayCleanupFailed) as exc_info:
        day_runner.assert_day_admissible(snapshot, "20200102")

    assert "20200101" in str(exc_info.value)
