from __future__ import annotations

from random import Random

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ._trino_errors import TrinoFailure

__all__ = ["BackoffPolicy", "RetryDecision", "next_delay", "retry_policy_for"]


class BackoffPolicy(BaseModel):
    """Configuration immutable d'une politique de backoff bornée."""

    model_config = ConfigDict(frozen=True)

    min_delay_s: float = Field(ge=0.0)
    max_delay_s: float = Field(gt=0.0)
    max_retries: int = Field(ge=0)
    base_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_delay_bounds(self) -> BackoffPolicy:
        """Garantit un intervalle de délai cohérent."""
        if self.min_delay_s > self.max_delay_s:
            msg = "min_delay_s must be less than or equal to max_delay_s"
            raise ValueError(msg)
        return self


class RetryDecision(BaseModel):
    """Décision pure indiquant si et pourquoi une erreur peut être relancée."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    max_attempts: int = Field(ge=0)
    reason: str


def next_delay(*, attempt: int, policy: BackoffPolicy, rng: Random) -> float:
    """Calcule un délai exponentiel jitteré, sans effectuer d'attente."""
    if attempt < 1:
        msg = "attempt must be greater than or equal to 1"
        raise ValueError(msg)

    exponential_delay = policy.base_s * (2 ** (attempt - 1))
    capped_delay = min(
        max(exponential_delay, policy.min_delay_s),
        policy.max_delay_s,
    )
    jittered_delay = rng.uniform(policy.min_delay_s, capped_delay)
    return min(max(jittered_delay, policy.min_delay_s), policy.max_delay_s)


def retry_policy_for(failure: TrinoFailure, policy: BackoffPolicy) -> RetryDecision:
    """Autorise uniquement les échecs temporaires de capacité."""
    if failure.kind == "retry_later":
        return RetryDecision(
            allowed=True,
            max_attempts=policy.max_retries,
            reason="retry_later",
        )

    return RetryDecision(
        allowed=False,
        max_attempts=0,
        reason=failure.kind,
    )
