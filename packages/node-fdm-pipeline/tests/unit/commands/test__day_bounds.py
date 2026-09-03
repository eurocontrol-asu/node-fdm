from __future__ import annotations

import importlib
from types import ModuleType


def _day_bounds() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._day_bounds")


def test_admit_slice_defers_when_resident_budget_would_be_exceeded() -> None:
    """AC1: a slice exceeding max_resident_gib is deferred with an explicit reason."""
    day_bounds = _day_bounds()
    budget = day_bounds.ResourceBudget(local_workers=2, max_resident_gib=8.0)

    admission = day_bounds.admit_slice(
        budget=budget,
        resident_gib=6.0,
        slice_gib=4.0,
    )

    assert admission.deferred is True
    assert "max_resident_gib" in admission.reason
