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
