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
    turning_starts: npt.NDArray[np.intp],
    n: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Compute per-sample ``(A_idx, B_idx)`` -- enclosing segment bounds.

    For each sample ``i``, finds the nearest turning-start at or before
    ``i`` (= ``A``, segment start) and the nearest turning-start strictly
    after ``i`` (= ``B``, segment end).  When ``i`` is before the first
    turn, ``A = 0``; after the last turn, ``B = n - 1``.

    Returns parallel arrays of length ``n``.
    """
    if turning_starts.size == 0:
        a = np.zeros(n, dtype=np.intp)
        b = np.full(n, n - 1, dtype=np.intp)
        return a, b

    pivots = np.arange(n, dtype=np.intp)
    idx_end = np.searchsorted(turning_starts, pivots, side="right")
    idx_start = idx_end - 1

    a = np.where(
        idx_start < 0,
        np.intp(0),
        turning_starts[np.clip(idx_start, 0, len(turning_starts) - 1)],
    ).astype(np.intp)
    b = np.where(
        idx_end >= len(turning_starts),
        np.intp(n - 1),
        turning_starts[np.clip(idx_end, 0, len(turning_starts) - 1)],
    ).astype(np.intp)
    return a, b


def build_in_turn_mask(
    turning_starts: npt.NDArray[np.intp],
    a_idx: npt.NDArray[np.intp],
    b_idx: npt.NDArray[np.intp],
    n: int,
) -> npt.NDArray[np.bool_]:
    """Mark samples that sit before the first turn or after the last as
    "outside any segment" -- treated like in_turn for masking purposes.

    Within identified segments, samples are straight by construction
    (turns are point-events in this algorithm, not intervals).  The
    ``in_turn`` flag we expose is therefore "no valid enclosing segment":
    True for the head and tail of the flight where ortho is undefined.
    """
    in_turn = np.zeros(n, dtype=np.bool_)
    if turning_starts.size == 0:
        in_turn[:] = True
        return in_turn
    in_turn[: turning_starts[0]] = True
    last_start = int(turning_starts[-1])
    in_turn[last_start:] = True
    in_turn |= a_idx == b_idx
    return in_turn
