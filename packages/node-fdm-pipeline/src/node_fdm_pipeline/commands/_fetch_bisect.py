from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = ["BisectionPlan", "plan_bisection"]


@dataclass(frozen=True, slots=True)
class BisectionPlan:
    """Describe either an ordered split or an indivisible batch."""

    kind: Literal["split", "terminal_floor"]
    children: tuple[tuple[str, ...], ...]
    batch: tuple[str, ...]


def plan_bisection(batch: Sequence[str], *, floor: int = 5) -> BisectionPlan:
    """Plan an ordered split without creating a child below the floor."""
    normalized_batch = tuple(batch)
    if len(normalized_batch) < 2 * floor:
        return BisectionPlan(
            kind="terminal_floor",
            children=(),
            batch=normalized_batch,
        )

    midpoint = len(normalized_batch) // 2
    return BisectionPlan(
        kind="split",
        children=(normalized_batch[:midpoint], normalized_batch[midpoint:]),
        batch=normalized_batch,
    )
