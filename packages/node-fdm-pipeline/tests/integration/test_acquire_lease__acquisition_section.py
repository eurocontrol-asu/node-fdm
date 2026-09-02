from __future__ import annotations

import importlib
import time
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType

import pytest
from pytest_mock import MockerFixture

from node_fdm_pipeline.commands._trino_lease import LeaseRecord


def _boundary_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._fleet_boundary")


def _lease_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._trino_lease")


def _read_record(path: Path) -> LeaseRecord:
    return LeaseRecord.model_validate_json(path.read_bytes())


@pytest.mark.integration
def test_held_section_excludes_second_remote_operation(
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """AC1: un second entrant est refusé avant tout appel distant."""
    boundary = _boundary_module()
    lease_module = _lease_module()
    lease_path = tmp_path / "trino.lease"
    first_operation = mocker.Mock()
    blocked_operation = mocker.Mock()

    with ExitStack() as stack:
        stack.enter_context(
            boundary.acquisition_section(
                lease_path,
                owner="owner-a",
                ttl_s=60.0,
            )
        )
        first_operation()

        with pytest.raises(lease_module.LeaseUnavailable):
            with boundary.acquisition_section(
                lease_path,
                owner="owner-b",
                ttl_s=60.0,
            ):
                blocked_operation()

    first_operation.assert_called_once_with()
    blocked_operation.assert_not_called()


@pytest.mark.integration
def test_heartbeat_renews_record_and_keeps_competitor_out(tmp_path: Path) -> None:
    """AC2: le heartbeat prolonge le lease pendant une section plus longue que son TTL."""
    boundary = _boundary_module()
    lease_module = _lease_module()
    lease_path = tmp_path / "trino.lease"
    ttl_s = 0.2

    with boundary.acquisition_section(
        lease_path,
        owner="owner-a",
        ttl_s=ttl_s,
    ):
        acquired = _read_record(lease_path)
        acquisition_expiry = acquired.acquired_at + ttl_s
        time.sleep(ttl_s * 1.75)
        renewed = _read_record(lease_path)

        assert renewed.expires_at > acquisition_expiry
        with pytest.raises(lease_module.LeaseUnavailable):
            lease_module.acquire_lease(
                lease_path,
                owner="owner-b",
                ttl_s=ttl_s,
                now=time.time(),
            )


@pytest.mark.integration
def test_stale_section_exit_preserves_successor_record(tmp_path: Path) -> None:
    """AC3: la sortie d'un propriétaire périmé ne modifie pas le lease de son successeur."""
    boundary = _boundary_module()
    lease_module = _lease_module()
    lease_path = tmp_path / "trino.lease"
    successor = None
    successor_record = None

    with pytest.raises(lease_module.LeaseNotOwned):
        with boundary.acquisition_section(
            lease_path,
            owner="owner-a",
            ttl_s=60.0,
        ):
            stale_record = _read_record(lease_path)
            expired_record = stale_record.model_copy(update={"expires_at": time.time() - 1.0})
            lease_path.write_bytes(expired_record.model_dump_json().encode())
            successor = lease_module.acquire_lease(
                lease_path,
                owner="owner-b",
                ttl_s=60.0,
                now=time.time(),
            )
            successor_record = _read_record(lease_path)

    assert successor_record is not None
    stored_after_stale_exit = _read_record(lease_path)
    assert stored_after_stale_exit.owner == successor_record.owner == "owner-b"
    assert stored_after_stale_exit.expires_at == successor_record.expires_at
    assert successor is not None
    successor.release()


@pytest.mark.integration
def test_body_exception_propagates_and_releases_lease(tmp_path: Path) -> None:
    """AC4: une exception du corps ressort inchangée et libère le lease partagé."""
    boundary = _boundary_module()
    lease_path = tmp_path / "trino.lease"
    marker = "remote section failed"
    body_error = ValueError(marker)

    with pytest.raises(ValueError, match=marker) as captured:
        with boundary.acquisition_section(
            lease_path,
            owner="owner-a",
            ttl_s=60.0,
        ):
            raise body_error

    assert captured.value is body_error
    with boundary.acquisition_section(
        lease_path,
        owner="owner-b",
        ttl_s=60.0,
    ):
        assert lease_path.is_file()
