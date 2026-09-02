from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from polars.exceptions import PolarsError
from pydantic import BaseModel, ConfigDict

from node_fdm_pipeline.commands._raw_cache import (
    AbsenceReceipt,
    Kind,
    absence_digest,
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


def reconcile_day(root: Path, day: str, kind: Kind) -> DayReconciliation:
    """Classify a day as visible only when its payload and receipt agree."""
    try:
        receipt = read_absence_receipt(root, day, kind)
    except (OSError, ValueError):
        return _invalid_reconciliation()
    if receipt is None:
        return _invalid_reconciliation()

    payload = _payload_path(root, day, receipt.digest)
    if not payload.is_file():
        return _invalid_reconciliation()

    payload_digest = _verified_payload_digest(payload, receipt, day, kind)
    if payload_digest != receipt.digest:
        return _invalid_reconciliation()

    return DayReconciliation(
        status="visible",
        digest=receipt.digest,
        row_count=receipt.row_count,
        icao24s=frozenset(receipt.icao24s),
    )


def purge_decision(
    reconciliation: DayReconciliation,
    consumers: Mapping[str, ConsumerState],
) -> PurgeDecision:
    """Decide whether a reconciled day is safe to purge, without I/O."""
    if reconciliation.status != "visible":
        return PurgeDecision(allowed=False, reason="receipt_invalid")
    if any(state != "committed" for state in consumers.values()):
        return PurgeDecision(allowed=False, reason="consumer_pending")
    return PurgeDecision(allowed=True)


def purge_day(
    root: Path,
    day: str,
    kind: Kind,
    consumers: Mapping[str, ConsumerState],
) -> PurgeDecision:
    """Remove a day directory only after receipt and consumer verification."""
    reconciliation = reconcile_day(root, day, kind)
    decision = purge_decision(reconciliation, consumers)
    if decision.allowed:
        day_directory = root / f"date={day}"
        if day_directory.is_dir():
            shutil.rmtree(day_directory)
    return decision
