from __future__ import annotations

import importlib

from node_fdm_pipeline.commands._trino_lease import LeaseNotOwned


class _UnownedLease:
    def __init__(self) -> None:
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1
        raise LeaseNotOwned


def test_release_owned_lease_ignores_an_unowned_handle() -> None:
    """AC3: an unowned lease is left alone without leaking LeaseNotOwned."""
    campaign_interrupt = importlib.import_module("node_fdm_pipeline.commands._campaign_interrupt")
    lease = _UnownedLease()

    released = campaign_interrupt.release_owned_lease(lease)

    assert released is False
    assert lease.release_calls == 1
