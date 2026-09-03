from __future__ import annotations

import importlib
from types import ModuleType


def _lease_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._trino_lease")


def test_renewed_record_extends_expiration() -> None:
    """AC2: un heartbeat repousse l'expiration au-delà du TTL initial."""
    lease_module = _lease_module()
    initial = lease_module.LeaseRecord(
        owner="A",
        acquired_at=1_000.0,
        expires_at=1_060.0,
    )

    renewed = initial.model_copy(
        update={
            "heartbeat_at": 1_050.0,
            "expires_at": 1_110.0,
        }
    )

    assert renewed.expires_at == 1_110.0
    assert renewed.is_expired(1_070.0) is False


def test_wait_policy_keeps_waiting_inside_budget_and_gives_up_after_it() -> None:
    """AC3: the pure wait decision is bounded and preserves the live holder identity."""

    lease_module = _lease_module()
    record = lease_module.LeaseRecord(
        owner="owner-a",
        acquired_at=1_000.0,
        heartbeat_at=1_000.0,
        expires_at=2_000.0,
    )

    inside = lease_module.decide_lease_wait(
        record,
        now=1_010.0,
        elapsed_s=0.5,
        wait_budget_s=1.0,
        poll_interval_s=0.1,
    )
    exhausted = lease_module.decide_lease_wait(
        record,
        now=1_011.0,
        elapsed_s=1.0,
        wait_budget_s=1.0,
        poll_interval_s=0.1,
    )

    assert inside.should_wait is True
    assert inside.delay_s == 0.1
    assert inside.holder == "owner-a"
    assert exhausted.should_wait is False
    assert exhausted.delay_s == 0.0
    assert exhausted.holder == "owner-a"
