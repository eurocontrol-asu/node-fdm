from __future__ import annotations

import importlib
from random import Random
from types import ModuleType

from node_fdm_pipeline.commands._trino_errors import classify_trino_failure


def _load_backoff_module() -> ModuleType:
    return importlib.import_module("node_fdm_pipeline.commands._fetch_backoff")


def test_delays_are_bounded_and_reproducible() -> None:
    """AC1: chaque délai est borné et reproductible pour une graine identique."""
    backoff = _load_backoff_module()
    policy = backoff.BackoffPolicy(
        min_delay_s=1.0,
        max_delay_s=30.0,
        max_retries=4,
        base_s=2.0,
    )

    first_rng = Random(0)  # noqa: S311 - deterministic test input, not cryptography
    second_rng = Random(0)  # noqa: S311 - deterministic test input, not cryptography
    first_delays = [
        backoff.next_delay(attempt=attempt, policy=policy, rng=first_rng)
        for attempt in range(1, policy.max_retries + 1)
    ]
    second_delays = [
        backoff.next_delay(attempt=attempt, policy=policy, rng=second_rng)
        for attempt in range(1, policy.max_retries + 1)
    ]

    assert all(policy.min_delay_s <= delay <= policy.max_delay_s for delay in first_delays)
    assert first_delays == second_delays


def test_retry_later_allows_a_bounded_number_of_attempts() -> None:
    """AC2: un refus temporaire autorise les relances plafonnées par la politique."""
    backoff = _load_backoff_module()
    policy = backoff.BackoffPolicy(
        min_delay_s=1.0,
        max_delay_s=30.0,
        max_retries=4,
        base_s=2.0,
    )
    failure = classify_trino_failure(RuntimeError("QUERY_QUEUE_FULL"))

    decision = backoff.retry_policy_for(failure, policy)

    assert decision.allowed is True
    assert decision.max_attempts == policy.max_retries


def test_non_retryable_failures_are_refused() -> None:
    """AC3: les erreurs split et terminal sont refusées avec leur raison explicite."""
    backoff = _load_backoff_module()
    policy = backoff.BackoffPolicy(
        min_delay_s=1.0,
        max_delay_s=30.0,
        max_retries=4,
        base_s=2.0,
    )
    for message, expected_reason in [
        ("EXCEEDED_TIME_LIMIT", "split"),
        ("SYNTAX_ERROR: invalid query", "terminal"),
    ]:
        failure = classify_trino_failure(RuntimeError(message))
        decision = backoff.retry_policy_for(failure, policy)

        assert decision.allowed is False
        assert decision.reason == expected_reason
