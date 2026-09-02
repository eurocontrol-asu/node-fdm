from __future__ import annotations

from importlib import import_module
from types import ModuleType

import pytest


@pytest.fixture
def trino_errors() -> ModuleType:
    return import_module("node_fdm_pipeline.commands._trino_errors")


def test_exceeded_time_limit_is_split(trino_errors: ModuleType) -> None:
    """AC1: EXCEEDED_TIME_LIMIT est classé comme une erreur à découper."""
    failure = trino_errors.classify_trino_failure(
        RuntimeError("Query exceeded maximum time limit: EXCEEDED_TIME_LIMIT")
    )

    assert failure.kind == "split"
    assert failure.code == "EXCEEDED_TIME_LIMIT"


def test_exceeded_memory_limit_is_split(trino_errors: ModuleType) -> None:
    """AC2: EXCEEDED_MEMORY_LIMIT est classé comme une erreur à découper."""
    failure = trino_errors.classify_trino_failure(
        RuntimeError("EXCEEDED_MEMORY_LIMIT: query exceeded per-node memory")
    )

    assert failure.kind == "split"
    assert failure.code == "EXCEEDED_MEMORY_LIMIT"


def test_query_queue_full_is_retry_later(trino_errors: ModuleType) -> None:
    """AC3: QUERY_QUEUE_FULL est classé comme un refus temporaire."""
    failure = trino_errors.classify_trino_failure(
        RuntimeError("QUERY_QUEUE_FULL: too many queued queries")
    )

    assert failure.kind == "retry_later"
    assert failure.code == "QUERY_QUEUE_FULL"


def test_unknown_error_is_terminal_and_preserves_message(trino_errors: ModuleType) -> None:
    """AC4: une erreur inconnue est terminale et conserve son message verbatim."""
    message = "SYNTAX_ERROR: line 1:8 mismatched input"

    failure = trino_errors.classify_trino_failure(RuntimeError(message))

    assert failure.kind == "terminal"
    assert failure.raw_message == message
