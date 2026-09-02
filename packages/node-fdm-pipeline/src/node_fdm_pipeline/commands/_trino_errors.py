from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = ["TrinoFailure", "classify_trino_failure"]

type TrinoFailureKind = Literal["split", "retry_later", "terminal"]

_SPLIT_CODES = (
    "EXCEEDED_TIME_LIMIT",
    "EXCEEDED_MEMORY_LIMIT",
    "EXCEEDED_LOCAL_MEMORY_LIMIT",
    "EXCEEDED_GLOBAL_MEMORY_LIMIT",
)
_RETRY_LATER_CODES = (
    "QUERY_QUEUE_FULL",
    "TOO_MANY_REQUESTS_FAILED",
    "SERVER_BUSY",
)
_CODE_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")


@dataclass(frozen=True, slots=True)
class TrinoFailure:
    code: str
    kind: TrinoFailureKind
    raw_message: str


def classify_trino_failure(error: BaseException) -> TrinoFailure:
    raw_message = str(error)
    normalized_message = raw_message.upper()

    for code in _SPLIT_CODES:
        if code in normalized_message:
            return TrinoFailure(code=code, kind="split", raw_message=raw_message)

    for code in _RETRY_LATER_CODES:
        if code in normalized_message:
            return TrinoFailure(code=code, kind="retry_later", raw_message=raw_message)

    match = _CODE_PATTERN.search(normalized_message)
    code = match.group() if match is not None else "UNKNOWN"
    return TrinoFailure(code=code, kind="terminal", raw_message=raw_message)
