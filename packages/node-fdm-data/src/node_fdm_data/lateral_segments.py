"""Segment-assignment primitives for lateral trajectory analysis.

These low-level helpers operate on raw ``numpy`` arrays of turning-start
indices and produce the per-sample segment bounds and ``in_turn`` mask
used by :func:`node_fdm_data.lateral.augment_lateral`.

They live in a dedicated module (rather than as private helpers under
``lateral``) so they can be tested independently through a stable public
API without exposing them at the ``node_fdm_data.lateral`` top-level
surface.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "build_in_turn_mask",
    "segment_bounds",
]


def segment_bounds(
    starts: npt.NDArray[np.intp],
    ends: npt.NDArray[np.intp],
    n: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Compute per-sample ``(A_idx, B_idx)`` -- enclosing straight-segment bounds.

    Given the turn intervals ``[s_k, e_k]`` produced by
    :func:`node_fdm_data.lateral.detect_turn_intervals`, the straight legs
    are everything *outside* those intervals.  For each sample ``i``:

    - ``B(i) = starts[k+1]`` when ``i ∈ [ends[k]+1, starts[k+1]-1]``
      (gap between two turns).
    - ``B(i) = n - 1`` when ``i > ends[-1]`` (tail).
    - ``B(i) = starts[0]`` when ``i < starts[0]`` (head).
    - For samples *inside* an interval, ``B`` still resolves to the next
      ``start`` (or ``n - 1`` if no further turn exists), so callers can
      safely compute a bearing before the back-fill step rewrites these
      positions.
    - ``A`` mirrors the construction with ``ends[k] + 1`` (or ``0`` before
      the first start).

    Returns parallel arrays of length ``n`` and dtype ``np.intp``.
    """
    if starts.size == 0:
        a = np.zeros(n, dtype=np.intp)
        b = np.full(n, n - 1, dtype=np.intp)
        return a, b

    pivots = np.arange(n, dtype=np.intp)

    next_start_idx = np.searchsorted(starts, pivots, side="right")
    has_next = next_start_idx < starts.size
    b = np.where(
        has_next,
        starts[np.clip(next_start_idx, 0, starts.size - 1)],
        np.intp(n - 1),
    ).astype(np.intp)

    last_end_idx = np.searchsorted(ends, pivots, side="left") - 1
    has_prev = last_end_idx >= 0
    a = np.where(
        has_prev,
        ends[np.clip(last_end_idx, 0, ends.size - 1)] + 1,
        np.intp(0),
    ).astype(np.intp)
    return a, b


def build_in_turn_mask(
    starts: npt.NDArray[np.intp],
    ends: npt.NDArray[np.intp],
    n: int,
) -> npt.NDArray[np.bool_]:
    """Mark samples sitting inside any ``[s_k, e_k]`` turn interval.

    V3 sémantique: ``in_turn`` is True iff the sample falls inside a
    detected turn interval (inclusive bounds).  The leading head (before
    the first start) and the trailing tail (after the last end) are
    *straight* segments and therefore False.
    """
    in_turn = np.zeros(n, dtype=np.bool_)
    for s, e in zip(starts, ends, strict=True):
        in_turn[int(s) : int(e) + 1] = True
    return in_turn
