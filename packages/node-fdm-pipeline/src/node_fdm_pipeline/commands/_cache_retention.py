from __future__ import annotations

import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from polars.exceptions import PolarsError
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from node_fdm_pipeline.commands._raw_cache import (
    AbsenceReceipt,
    Kind,
    absence_digest,
    publish_absence,
    read_absence_receipt,
    read_parquet,
)

__all__ = [
    "DayReconciliation",
    "PurgeDecision",
    "purge_day",
    "purge_decision",
    "reconcile_day",
]

type ReconciliationStatus = Literal["invalid", "visible"]
type ConsumerState = Literal["committed", "pending"]
type PurgeReason = Literal["receipt_invalid", "consumer_pending"]


class DayReconciliation(BaseModel):
    """Immutable visibility report for one raw-cache day."""

    model_config = ConfigDict(frozen=True)

    status: ReconciliationStatus
    digest: str | None = None
    row_count: int | None = None
    icao24s: frozenset[str] = frozenset()


class PurgeDecision(BaseModel):
    """Immutable result of the pure retention gate."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: PurgeReason | None = None
    pending_consumers: tuple[str, ...] = ()


def _recover_missing_receipt(
    root: Path,
    day: str,
    kind: Kind,
    icao24s: Iterable[str],
) -> DayReconciliation:
    canonical_icao24s = tuple(sorted(set(icao24s)))
    digest = absence_digest(day, kind, canonical_icao24s)
    payload = _payload_path(root, day, digest)
    candidate = AbsenceReceipt(
        day=day,
        kind=kind,
        digest=digest,
        icao24s=canonical_icao24s,
    )
    if not payload.is_file():
        return _invalid_reconciliation()
    if _verified_payload_digest(payload, candidate, day, kind) != digest:
        return _invalid_reconciliation()
    return _visible_reconciliation(publish_absence(root, day, kind, canonical_icao24s))


def _visible_reconciliation(receipt: AbsenceReceipt) -> DayReconciliation:
    return DayReconciliation(
        status="visible",
        digest=receipt.digest,
        row_count=receipt.row_count,
        icao24s=frozenset(receipt.icao24s),
    )


def invalidate_day(root: Path, day: str, kind: Kind) -> None:
    """Remove invalid absence artefacts so acquisition can publish a clean day."""
    shutil.rmtree(root / f"date={day}" / ".absences", ignore_errors=True)
    receipt = root / ".absences" / f"{day}.{kind}.json"
    receipt.unlink(missing_ok=True)
    for temporary in receipt.parent.glob(f".{receipt.name}.*.tmp"):
        temporary.unlink(missing_ok=True)


def _invalid_reconciliation() -> DayReconciliation:
    return DayReconciliation(status="invalid")


def _payload_path(root: Path, day: str, digest: str) -> Path:
    return root / f"date={day}" / ".absences" / f"{digest}.parquet"


def _verified_payload_digest(
    payload: Path,
    receipt: AbsenceReceipt,
    day: str,
    kind: Kind,
) -> str | None:
    try:
        row_count = read_parquet(payload).height
    except (OSError, PolarsError):
        return None
    if row_count != receipt.row_count:
        return None
    return absence_digest(day, kind, receipt.icao24s)


def reconcile_day(
    root: Path,
    day: str,
    kind: Kind,
    *,
    icao24s: Iterable[str] | None = None,
) -> DayReconciliation:
    """Classify a day as visible only when its payload and receipt agree."""
    try:
        receipt = read_absence_receipt(root, day, kind)
    except (OSError, ValueError):
        return _invalid_reconciliation()
    if receipt is None:
        if icao24s is None:
            return _invalid_reconciliation()
        return _recover_missing_receipt(root, day, kind, icao24s)

    payload = _payload_path(root, day, receipt.digest)
    if not payload.is_file():
        return _invalid_reconciliation()

    payload_digest = _verified_payload_digest(payload, receipt, day, kind)
    if payload_digest != receipt.digest:
        return _invalid_reconciliation()

    return _visible_reconciliation(receipt)


def purge_decision(
    reconciliation: DayReconciliation,
    consumers: Mapping[str, ConsumerState],
) -> PurgeDecision:
    """Decide whether a reconciled day is safe to purge, without I/O."""
    if reconciliation.status != "visible":
        return PurgeDecision(allowed=False, reason="receipt_invalid")
    pending = tuple(sorted(name for name, state in consumers.items() if state != "committed"))
    if pending:
        return PurgeDecision(
            allowed=False,
            reason="consumer_pending",
            pending_consumers=pending,
        )
    return PurgeDecision(allowed=True)


def _read_consumer_ledger(root: Path, day: str) -> dict[str, ConsumerState] | None:
    path = root / f"date={day}" / "consumers.json"
    try:
        return TypeAdapter(dict[str, ConsumerState]).validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None


def purge_day(
    root: Path,
    day: str,
    kind: Kind,
) -> PurgeDecision:
    """Remove a day directory only after receipt and consumer verification."""
    consumers = _read_consumer_ledger(root, day) or {"consumer_ledger": "pending"}
    reconciliation = reconcile_day(root, day, kind)
    decision = purge_decision(reconciliation, consumers)
    if decision.allowed:
        receipt_path = root / ".absences" / f"{day}.{kind}.json"
        receipt_path.unlink(missing_ok=True)
        day_directory = root / f"date={day}"
        if day_directory.is_dir():
            shutil.rmtree(day_directory)
    return decision
