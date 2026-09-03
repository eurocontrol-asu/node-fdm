"""Unit contracts for durable day cleanup decisions."""

from __future__ import annotations

import importlib
from types import ModuleType


def _cleanup() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_cleanup")


def test_zero_reference_artifacts_selects_only_zero_counts() -> None:
    """AC5: only artefacts with no future reference are eligible for deletion."""
    counters = {
        "raw/20200101/A20N": 0,
        "era5/20200101/t2m": 2,
        "raw/20191231/A20N": 0,
    }

    selected = _cleanup().zero_reference_artifacts(counters)

    assert selected == ("raw/20191231/A20N", "raw/20200101/A20N")
    assert "era5/20200101/t2m" not in selected
