from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

__all__ = [
    "Lease",
    "LeaseConfigError",
    "LeaseNotOwned",
    "LeaseRecord",
    "LeaseUnavailable",
    "acquire_lease",
    "require_shared_lease_path",
]


class LeaseUnavailable(RuntimeError):  # noqa: N818
    """Raised when another owner holds a live lease."""


class LeaseNotOwned(RuntimeError):  # noqa: N818
    """Raised when a caller no longer owns the stored lease."""


class LeaseConfigError(ValueError):
    """Raised when the shared lease path cannot be used safely."""


class LeaseRecord(BaseModel):
    """Durable state of a shared filesystem lease."""

    model_config = ConfigDict(frozen=True)

    owner: str
    acquired_at: float
    expires_at: float
    heartbeat_at: float | None = None

    def is_expired(self, now: float) -> bool:
        """Return whether the lease has expired at the supplied time."""
        return now >= self.expires_at


class Lease:
    """Handle used by one owner to maintain and release its lease."""

    def __init__(self, path: Path, owner: str, ttl_s: float) -> None:
        self._path = path
        self._owner = owner
        self._ttl_s = ttl_s

    def heartbeat(self, *, now: float) -> LeaseRecord:
        """Extend this owner's lease using an atomic replacement."""
        current = _read_owned_record(self._path, self._owner)
        renewed = current.model_copy(update={"heartbeat_at": now, "expires_at": now + self._ttl_s})
        _replace_record(self._path, renewed)
        return renewed

    def release(self) -> None:
        """Release the lease if this handle still owns the stored record."""
        _read_owned_record(self._path, self._owner)
        self._path.unlink()


def require_shared_lease_path(path: str | os.PathLike[str] | None) -> Path:
    """Validate and resolve a caller-provided shared lease path."""
    if path is None:
        raise LeaseConfigError("A shared lease path is required")

    resolved = Path(path).expanduser().resolve()
    parent = resolved.parent
    try:
        mode = parent.stat().st_mode
    except OSError as exc:
        raise LeaseConfigError(f"Shared lease parent is unavailable: {parent}") from exc

    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    if not parent.is_dir() or not mode & write_bits or not os.access(parent, os.W_OK):
        raise LeaseConfigError(f"Shared lease parent is not writable: {parent}")
    return resolved


def acquire_lease(
    path: str | os.PathLike[str],
    *,
    owner: str,
    ttl_s: float,
    now: float,
) -> Lease:
    """Acquire an exclusive lease, replacing only an expired record."""
    lease_path = Path(path).expanduser().resolve()
    record = LeaseRecord(
        owner=owner,
        acquired_at=now,
        heartbeat_at=now,
        expires_at=now + ttl_s,
    )
    try:
        _create_record_exclusively(lease_path, record)
    except FileExistsError:
        try:
            current = _read_record(lease_path)
        except LeaseNotOwned:
            try:
                _create_record_exclusively(lease_path, record)
            except FileExistsError as exc:
                raise LeaseUnavailable("Lease changed during acquisition") from exc
        else:
            if not current.is_expired(now):
                raise LeaseUnavailable(f"Lease is held by {current.owner}") from None
            try:
                lease_path.unlink()
                _create_record_exclusively(lease_path, record)
            except (FileExistsError, FileNotFoundError) as exc:
                raise LeaseUnavailable("Lease changed during acquisition") from exc
    return Lease(lease_path, owner, ttl_s)


def _record_bytes(record: LeaseRecord) -> bytes:
    payload = record.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _create_record_exclusively(path: Path, record: LeaseRecord) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as lease_file:
            lease_file.write(_record_bytes(record))
            lease_file.flush()
            os.fsync(lease_file.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_record(path: Path) -> LeaseRecord:
    try:
        return LeaseRecord.model_validate_json(path.read_bytes())
    except FileNotFoundError as exc:
        raise LeaseNotOwned("The lease no longer exists") from exc
    except ValidationError as exc:
        raise LeaseUnavailable("The lease record is not readable yet") from exc


def _read_owned_record(path: Path, owner: str) -> LeaseRecord:
    record = _read_record(path)
    if record.owner != owner:
        raise LeaseNotOwned(f"Lease belongs to {record.owner}, not {owner}")
    return record


def _replace_record(path: Path, record: LeaseRecord) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as lease_file:
            lease_file.write(_record_bytes(record))
            lease_file.flush()
            os.fsync(lease_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
