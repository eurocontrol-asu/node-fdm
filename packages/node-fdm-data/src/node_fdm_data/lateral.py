"""Lateral trajectory computations for Neural ODE control inputs.

Detects start-of-turn points via smoothed angular rate + peak detection
with backtrack, then computes a great-circle reference track from each
point to the end of its straight segment.  This serves as the **lateral
control input** for the flight dynamics model -- the Neural ODE learns
aircraft response to navigation commands, analogous to how
``fdm_alt_target_m`` / ``fdm_tas_target_ms`` serve as longitudinal commands.

Algorithm ported verbatim from the legacy production module
``opensky_v2/src/node_fdm/data/lateral_computations.py``:

1. Savitzky-Golay smoothing of the track signal.
2. ``scipy.signal.find_peaks`` on absolute rotation rate.
3. Backtrack from each peak until rotation rate falls below a noise
   floor -- locates the **start** of every turn.
4. ``searchsorted`` to assign each sample to its enclosing straight
   segment ``[A, B)`` and read endpoint coordinates.
5. Great-circle bearing from current position C to segment end B
   gives ``track_ortho`` (the FMS-equivalent lateral target).

Inside turns the bearing is undefined while transitioning between legs;
:func:`augment_lateral` back-fills those samples with the *next* straight
segment's bearing (then forward-fills any leftover tail) so that
``fdm_track_ortho_deg`` is finite on every sample of a non-degenerate
flight and ``fdm_track_sel_known`` is True everywhere.

Example::

    from node_fdm_data.lateral import augment_lateral

    result = augment_lateral(flight_df)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.signal import find_peaks, savgol_filter

from node_fdm_data.lateral_segments import build_in_turn_mask, segment_bounds
from node_fdm_data.smoothing import bilateral_1d

__all__ = [
    "augment_lateral",
    "detect_turn_intervals",
    "detect_turning_starts",
    "orthodromic_bearing",
]

R_EARTH_M: float = 6_371_000.0
"""Mean Earth radius in metres."""

_SAVGOL_WINDOW: int = 9
_SAVGOL_POLY: int = 3
_PEAK_DISTANCE: int = 10
_HALF_TURN_DEG: float = 180.0
_FULL_TURN_DEG: float = 360.0


# ---------------------------------------------------------------------------
# Low-level geometry
# ---------------------------------------------------------------------------


def orthodromic_bearing(
    phi1: npt.ArrayLike,
    lam1: npt.ArrayLike,
    phi2: npt.ArrayLike,
    lam2: npt.ArrayLike,
) -> np.ndarray:
    """Initial great-circle bearing from (phi1, lam1) to (phi2, lam2).

    All inputs and output in **radians**.  Result is in ``[0, 2 pi)``.
    """
    phi1, lam1 = np.asarray(phi1), np.asarray(lam1)
    phi2, lam2 = np.asarray(phi2), np.asarray(lam2)
    d_lam = lam2 - lam1
    y = np.sin(d_lam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(d_lam)
    bearing = np.arctan2(y, x)
    return np.asarray((bearing + 2 * np.pi) % (2 * np.pi))


# ---------------------------------------------------------------------------
# Turn detection (V3: Savgol + bilateral + symmetric-threshold intervals)
# ---------------------------------------------------------------------------


def detect_turn_intervals(  # noqa: PLR0913
    track_deg: npt.NDArray[np.floating[Any]],
    *,
    dt: float = 4.0,
    rate_threshold: float = 0.05,
    bilateral_sigma_s: float = 8.0,
    bilateral_sigma_r: float = 0.01,
    bilateral_passes: int = 2,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.intp],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Detect ``(start, end)`` index intervals for every turn in a track signal.

    V3 algorithm: smooth the track with Savitzky-Golay, derive
    ``|d_savgol(track)/dt|``, then apply two passes of
    :func:`bilateral_1d` to flatten sub-threshold noise while preserving
    jumps.  ``find_peaks`` locates turn centres; a single
    ``rate_threshold`` is then walked back AND forward from each peak
    to define the symmetric ``[start, end]`` bracket.  Overlapping
    intervals are merged.

    Args:
        track_deg: Track angle in degrees (may wrap at 0/360).
        dt: Sampling interval in seconds (constant grid).
        rate_threshold: Single threshold used for both peak height and
            the symmetric walk-back/walk-forward boundaries.

    Returns:
        ``(starts, ends, abs_rate_raw, abs_rate_bilat)`` where ``starts``
        and ``ends`` are sorted, merged ``np.intp`` index arrays
        (``starts <= ends``, strictly increasing), ``abs_rate_raw`` is
        the pre-smoothing absolute rotation rate, and ``abs_rate_bilat``
        is the bilateral-smoothed rate used for peak detection.
    """
    n = len(track_deg)
    if n < _SAVGOL_WINDOW:
        empty = np.empty(0, dtype=np.intp)
        zeros = np.zeros(n, dtype=np.float64)
        return empty, empty, zeros, zeros

    track_filled, all_nan = _forward_fill_track(track_deg)
    if all_nan:
        empty = np.empty(0, dtype=np.intp)
        zeros = np.zeros(n, dtype=np.float64)
        return empty, empty, zeros, zeros

    smoothed = savgol_filter(track_filled, _SAVGOL_WINDOW, _SAVGOL_POLY)
    d_track = _unwrap_diff(np.diff(smoothed, prepend=smoothed[0]))
    abs_rate = np.abs(d_track / dt)

    abs_rate_bilat = abs_rate.copy()
    for _ in range(bilateral_passes):
        abs_rate_bilat = bilateral_1d(abs_rate_bilat, bilateral_sigma_s, bilateral_sigma_r)

    peaks, _ = find_peaks(
        abs_rate_bilat,
        height=rate_threshold,
        distance=_PEAK_DISTANCE,
    )
    if peaks.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, abs_rate, abs_rate_bilat

    starts_list = [_walk_back_below(abs_rate_bilat, int(p), rate_threshold) for p in peaks]
    ends_list = [_walk_forward_below(abs_rate_bilat, int(p), rate_threshold) for p in peaks]

    starts_arr = np.asarray(starts_list, dtype=np.intp)
    ends_arr = np.asarray(ends_list, dtype=np.intp)
    order = np.argsort(starts_arr)
    starts_arr = starts_arr[order]
    ends_arr = ends_arr[order]

    merged_s: list[int] = [int(starts_arr[0])]
    merged_e: list[int] = [int(ends_arr[0])]
    for s, e in zip(starts_arr[1:], ends_arr[1:], strict=True):
        if int(s) <= merged_e[-1]:
            merged_e[-1] = max(merged_e[-1], int(e))
        else:
            merged_s.append(int(s))
            merged_e.append(int(e))

    return (
        np.asarray(merged_s, dtype=np.intp),
        np.asarray(merged_e, dtype=np.intp),
        abs_rate,
        abs_rate_bilat,
    )


def _walk_back_below(abs_rate: npt.NDArray[np.float64], peak: int, threshold: float) -> int:
    """Walk left from ``peak`` until rate falls below ``threshold``."""
    i = int(peak)
    while i > 0:
        i -= 1
        if abs_rate[i] < threshold:
            return i + 1
    return 0


def _walk_forward_below(abs_rate: npt.NDArray[np.float64], peak: int, threshold: float) -> int:
    """Walk right from ``peak`` until rate falls below ``threshold``."""
    n = abs_rate.size
    j = int(peak)
    while j < n - 1:
        j += 1
        if abs_rate[j] < threshold:
            return j - 1
    return n - 1


def detect_turning_starts(
    track_deg: npt.NDArray[np.floating[Any]],
    *,
    dt: float = 4.0,
    threshold_deg_per_sec: float = 0.05,
    noise_threshold_deg_per_sec: float = 0.005,
) -> npt.NDArray[np.intp]:
    """Deprecated shim: use :func:`detect_turn_intervals` instead.

    ``noise_threshold_deg_per_sec`` is accepted but ignored: V3 uses a
    single symmetric threshold (``threshold_deg_per_sec``) for both peak
    detection and the start/end walk-back/walk-forward.
    """
    warnings.warn(
        "detect_turning_starts is deprecated; use detect_turn_intervals instead. "
        "noise_threshold_deg_per_sec is ignored under the V3 symmetric-threshold algorithm.",
        DeprecationWarning,
        stacklevel=2,
    )
    starts, _, _, _ = detect_turn_intervals(track_deg, dt=dt, rate_threshold=threshold_deg_per_sec)
    return starts


def _forward_fill_track(
    track_deg: npt.NDArray[np.floating[Any]],
) -> tuple[npt.NDArray[np.float64], bool]:
    """Forward-fill non-finite samples; return ``(filled, all_nan)``.

    Savgol does not tolerate NaN/inf; the short 9-sample window means
    isolated gaps get a near-constant local fill so detected rotation
    rate at those points is ~0 (no spurious turn detection).
    """
    track_filled = np.asarray(track_deg, dtype=np.float64).copy()
    if np.all(np.isfinite(track_filled)):
        return track_filled, False
    bad = ~np.isfinite(track_filled)
    good_idx = np.flatnonzero(~bad)
    if good_idx.size == 0:
        return track_filled, True
    last = track_filled[good_idx[0]]
    for i in range(track_filled.size):
        if bad[i]:
            track_filled[i] = last
        else:
            last = track_filled[i]
    return track_filled, False


def _unwrap_diff(d_track: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Wrap-protect a diff'd track: large jumps come from the 0/360 boundary."""
    d_track = np.where(d_track > _HALF_TURN_DEG, d_track - _FULL_TURN_DEG, d_track)
    d_track = np.where(d_track < -_HALF_TURN_DEG, d_track + _FULL_TURN_DEG, d_track)
    return d_track


def _bfill_then_ffill(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Back-fill NaN with the next finite value, then forward-fill any leftovers."""
    n = arr.size
    valid = ~np.isnan(arr)
    if not valid.any() or valid.all():
        return arr.copy()

    out = arr.copy()

    sentinel = n
    idx_right = np.where(valid, np.arange(n), sentinel)
    next_valid = np.minimum.accumulate(idx_right[::-1])[::-1]
    bfill_mask = ~valid & (next_valid < sentinel)
    out[bfill_mask] = arr[next_valid[bfill_mask]]

    valid2 = ~np.isnan(out)
    if valid2.all():
        return out
    idx_left = np.where(valid2, np.arange(n), -1)
    prev_valid = np.maximum.accumulate(idx_left)
    ffill_mask = ~valid2 & (prev_valid >= 0)
    out[ffill_mask] = out[prev_valid[ffill_mask]]
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_lateral(  # noqa: PLR0913
    df: pl.DataFrame,
    *,
    dt: float = 4.0,
    rate_threshold: float = 0.05,
    bilateral_sigma_s: float = 8.0,
    bilateral_sigma_r: float = 0.01,
    bilateral_passes: int = 2,
) -> pl.DataFrame:
    """Augment a flight DataFrame with a lateral reference track.

    Adds three columns:

    - ``fdm_in_turn`` (bool) -- True iff the sample sits inside a
      detected turn interval ``[s_k, e_k]``.  Head and tail of the
      flight are straight by construction and therefore False.
    - ``fdm_track_ortho_deg`` (deg) -- great-circle bearing from the
      current position to the next straight-segment end ``B``.  Inside
      a turn, the value is back-filled with the *next* straight
      segment's bearing so the FMS-equivalent target is defined
      everywhere.
    - ``fdm_track_sel_known`` (bool) -- equal to
      ``np.isfinite(fdm_track_ortho_deg)``.  After bfill+ffill this is
      True on every sample of a non-degenerate flight.

    Args:
        df: Single-flight eager DataFrame with ``latitude``,
            ``longitude``, and ``track`` columns (degrees).  Sampling is
            assumed uniform at ``dt`` seconds.
        dt: Sampling interval in seconds.
        rate_threshold: Peak detection threshold (deg/s) — see
            :func:`detect_turn_intervals`.
        bilateral_sigma_s: Spatial sigma of the bilateral smoother.
        bilateral_sigma_r: Range sigma of the bilateral smoother.
        bilateral_passes: Number of bilateral smoothing passes.

    Returns:
        DataFrame with the three lateral columns appended.
    """
    n = df.height
    if n < _SAVGOL_WINDOW:
        return df.with_columns(
            pl.lit(True).alias("fdm_in_turn"),
            pl.lit(None, dtype=pl.Float64).alias("fdm_track_ortho_deg"),
            pl.lit(False).alias("fdm_track_sel_known"),
        )

    track_raw = df["track"].to_numpy().astype(np.float64)
    lat = df["latitude"].to_numpy().astype(np.float64)
    lon = df["longitude"].to_numpy().astype(np.float64)

    starts, ends, _, _ = detect_turn_intervals(
        track_raw,
        dt=dt,
        rate_threshold=rate_threshold,
        bilateral_sigma_s=bilateral_sigma_s,
        bilateral_sigma_r=bilateral_sigma_r,
        bilateral_passes=bilateral_passes,
    )

    _, b_idx = segment_bounds(starts, ends, n)
    in_turn = build_in_turn_mask(starts, ends, n)

    phi_c = np.radians(lat)
    lam_c = np.radians(lon)
    phi_b = np.radians(lat[b_idx])
    lam_b = np.radians(lon[b_idx])

    ortho_rad = orthodromic_bearing(phi_c, lam_c, phi_b, lam_b)
    ortho_deg = np.degrees(ortho_rad)
    ortho_deg[in_turn] = np.nan
    ortho_deg = _bfill_then_ffill(ortho_deg)

    known = np.isfinite(ortho_deg)

    return df.with_columns(
        pl.Series("fdm_in_turn", in_turn),
        pl.Series("fdm_track_ortho_deg", ortho_deg),
        pl.Series("fdm_track_sel_known", known),
    )
