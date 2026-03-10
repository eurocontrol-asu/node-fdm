"""Configurable flight data processing pipeline.

``FlightProcessor`` applies a sequence of transformation steps to a
Polars ``LazyFrame``.  Each step is a callable that takes a ``LazyFrame``
and returns a ``LazyFrame``.

Example::

    proc = (
        FlightProcessor()
        .add_step(apply_conversions)
        .add_step(compute_derivatives)
    )
    result = proc.process(raw_data).collect()
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    pass

__all__ = [
    "FlightProcessor",
    "TransformFn",
]

type TransformFn = Callable[[pl.LazyFrame], pl.LazyFrame]


class FlightProcessor:
    """Configurable Polars processing pipeline.

    Args:
        steps: Initial sequence of transformation functions.  Each function
            receives a ``pl.LazyFrame`` and must return a ``pl.LazyFrame``.
    """

    def __init__(self, steps: Sequence[TransformFn] = ()) -> None:
        self._steps: list[TransformFn] = list(steps)

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_step(self, step: TransformFn) -> FlightProcessor:
        """Append a transformation step, returns *self* for chaining."""
        self._steps.append(step)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def process(self, df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
        """Apply all steps sequentially.

        If *df* is an eager ``DataFrame`` it is automatically converted to
        a ``LazyFrame`` before processing.

        Args:
            df: Input data — lazy or eager.

        Returns:
            Transformed ``LazyFrame``.
        """
        lf: pl.LazyFrame = df.lazy() if isinstance(df, pl.DataFrame) else df

        for step in self._steps:
            lf = step(lf)

        return lf
