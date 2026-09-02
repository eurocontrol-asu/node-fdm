from __future__ import annotations

import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest


def _lease_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._trino_lease")


def _stored_owner(path: Path) -> str:
    payload = json.loads(path.read_bytes())
    assert isinstance(payload, dict)
    return str(payload["owner"])


@pytest.mark.integration
def test_live_lease_blocks_a_second_owner(tmp_path: Path) -> None:
    """AC1: un lease vivant refuse un autre propriétaire sans changer le stockage."""
    lease_module = _lease_module()
    lease_path = tmp_path / "trino.lease"
    lease_module.acquire_lease(lease_path, owner="A", ttl_s=60, now=1_000.0)

    with pytest.raises(lease_module.LeaseUnavailable):
        lease_module.acquire_lease(lease_path, owner="B", ttl_s=60, now=1_010.0)

    assert _stored_owner(lease_path) == "A"


@pytest.mark.integration
def test_heartbeat_keeps_the_lease_beyond_its_initial_ttl(tmp_path: Path) -> None:
    """AC2: le heartbeat du propriétaire prolonge le lease et bloque un concurrent."""
    lease_module = _lease_module()
    lease_path = tmp_path / "trino.lease"
    lease = lease_module.acquire_lease(
        lease_path,
        owner="A",
        ttl_s=60,
        now=1_000.0,
    )

    lease.heartbeat(now=1_050.0)

    with pytest.raises(lease_module.LeaseUnavailable):
        lease_module.acquire_lease(lease_path, owner="B", ttl_s=60, now=1_070.0)
    stored = lease_module.LeaseRecord.model_validate_json(lease_path.read_bytes())
    assert stored.owner == "A"
    assert stored.expires_at == 1_110.0
    assert stored.is_expired(1_070.0) is False


@pytest.mark.integration
def test_expired_lease_transfers_ownership_and_rejects_stale_owner(tmp_path: Path) -> None:
    """AC3: après reprise, l'ancien propriétaire ne peut modifier aucun octet."""
    lease_module = _lease_module()
    lease_path = tmp_path / "trino.lease"
    stale = lease_module.acquire_lease(
        lease_path,
        owner="A",
        ttl_s=60,
        now=1_000.0,
    )

    lease_module.acquire_lease(lease_path, owner="B", ttl_s=60, now=1_061.0)

    assert _stored_owner(lease_path) == "B"
    before_stale_calls = lease_path.read_bytes()
    with pytest.raises(lease_module.LeaseNotOwned):
        stale.heartbeat(now=1_062.0)
    assert lease_path.read_bytes() == before_stale_calls
    with pytest.raises(lease_module.LeaseNotOwned):
        stale.release()
    assert lease_path.read_bytes() == before_stale_calls


@pytest.mark.integration
def test_missing_shared_path_fails_without_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC4: un chemin absent échoue sans créer de verrou local de remplacement."""
    lease_module = _lease_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)

    with pytest.raises(lease_module.LeaseConfigError):
        lease_module.require_shared_lease_path(None)

    assert [path for path in tmp_path.rglob("*") if path.is_file()] == []


@pytest.mark.integration
def test_shared_path_requires_writable_parent_and_resolves_absolute_path(
    tmp_path: Path,
) -> None:
    """AC5: le parent doit être inscriptible et le chemin valide est résolu."""
    lease_module = _lease_module()
    blocked_parent = tmp_path / "blocked"
    blocked_parent.mkdir()
    blocked_path = blocked_parent / "trino.lease"
    blocked_parent.chmod(0o500)
    try:
        with pytest.raises(lease_module.LeaseConfigError):
            lease_module.require_shared_lease_path(blocked_path)
    finally:
        blocked_parent.chmod(0o700)

    assert list(blocked_parent.iterdir()) == []
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir()
    shared_path = shared_parent / ".." / "shared" / "trino.lease"
    assert lease_module.require_shared_lease_path(shared_path) == shared_path.resolve()


@pytest.mark.integration
def test_eight_contenders_have_one_maximum_simultaneous_holder(tmp_path: Path) -> None:
    """AC6: huit workers sur le même fichier ont au plus un détenteur à la fois."""
    lease_module = _lease_module()
    lease_path = tmp_path / "trino.lease"
    contenders = 8
    start = threading.Barrier(contenders)
    counter_guard = threading.Lock()
    retry_pause = threading.Event()
    active = 0
    peak = 0

    def contend(worker_number: int) -> None:
        nonlocal active, peak
        owner = f"worker-{worker_number}"
        start.wait()
        while True:
            try:
                current = lease_module.acquire_lease(
                    lease_path,
                    owner=owner,
                    ttl_s=60,
                    now=1_000.0,
                )
            except lease_module.LeaseUnavailable:
                retry_pause.wait(0.001)
                continue
            break

        with counter_guard:
            active += 1
            peak = max(peak, active)
        retry_pause.wait(0.01)
        with counter_guard:
            current.release()
            active -= 1

    with ThreadPoolExecutor(max_workers=contenders) as executor:
        list(executor.map(contend, range(contenders)))

    assert peak == 1
