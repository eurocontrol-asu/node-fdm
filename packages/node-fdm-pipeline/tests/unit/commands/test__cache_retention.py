"""Unit contracts for guarded raw-cache retention."""

from __future__ import annotations

import importlib
from types import ModuleType


def _retention() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._cache_retention")


def test_invalid_receipt_forbids_purge() -> None:
    """AC4: an invalid receipt forbids purge with a named reason."""
    retention = _retention()
    reconciliation = retention.DayReconciliation(
        status="invalid",
        digest=None,
        row_count=None,
        icao24s=frozenset(),
    )

    decision = retention.purge_decision(reconciliation, {})

    assert decision.allowed is False
    assert decision.reason == "receipt_invalid"


def test_pending_consumer_forbids_purge() -> None:
    """AC4: a declared consumer not committed forbids purge."""
    retention = _retention()
    reconciliation = retention.DayReconciliation(
        status="visible",
        digest="verified-digest",
        row_count=0,
        icao24s=frozenset({"abc123"}),
    )

    decision = retention.purge_decision(reconciliation, {"warehouse": "pending"})

    assert decision.allowed is False
    assert decision.reason == "consumer_pending"
