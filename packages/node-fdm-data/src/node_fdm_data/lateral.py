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

The reference track is naturally NaN before the first detected turn and
after the last (no enclosing segment).  Inside turns it is also set to
NaN -- the FMS target is undefined while transitioning between legs.

Example::

    from node_fdm_data.lateral import augment_lateral

    result = augment_lateral(flight_df)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.signal import find_peaks, savgol_filter

from node_fdm_data.lateral_segments import build_in_turn_mask, segment_bounds

__all__ = [
    "augment_lateral",
    "detect_turning_starts",
    "orthodromic_bearing",
]

R_EARTH_M: float = 6_371_000.0
"""Mean Earth radius in metres."""

_SAVGOL_WINDOW: int = 9
_SAVGOL_POLY: int = 3
_PEAK_DISTANCE: int = 10
_DEFAULT_THRESHOLD: float = 0.05
_DEFAULT_NOISE_FLOOR: float = 0.005
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
# Turn detection (Savgol + find_peaks + backtrack, legacy production algo)
# ---------------------------------------------------------------------------


def detect_turning_starts(
    track_deg: npt.NDArray[np.floating[Any]],
    *,
    dt: float = 4.0,
    threshold_deg_per_sec: float = _DEFAULT_THRESHOLD,
    noise_threshold_deg_per_sec: float = _DEFAULT_NOISE_FLOOR,
) -> npt.NDArray[np.intp]:
    """Detect the **start index** of every turn in a track signal.

    Mirrors ``detect_start_of_turning_points`` from the legacy production
    module.  Smooths the track with Savitzky-Golay, computes rotation
    rate, picks peaks above ``threshold_deg_per_sec`` then walks
    backwards from each peak until the rate falls below
    ``noise_threshold_deg_per_sec``.

    Args:
        track_deg: Track angle in degrees (may wrap at 0/360).
        dt: Sampling interval in seconds (constant grid).
        threshold_deg_per_sec: Peak height for ``find_peaks``.
        noise_threshold_deg_per_sec: Backtrack stops when rate drops
            below this -- defines the practical "start of turn".

    Returns:
        Sorted unique array of indices marking turn-starts.  Empty if
        the signal is shorter than the Savgol window or contains no
        peaks.
    """
    n = len(track_deg)
    if n < _SAVGOL_WINDOW:
        return np.empty(0, dtype=np.intp)

    track_filled, all_nan = _forward_fill_track(track_deg)
    if all_nan:
        return np.empty(0, dtype=np.intp)

    smoothed = savgol_filter(track_filled, _SAVGOL_WINDOW, _SAVGOL_POLY)
    d_track = _unwrap_diff(np.diff(smoothed, prepend=smoothed[0]))
    abs_rate = np.abs(d_track / dt)

    peaks, _ = find_peaks(
        abs_rate,
        height=threshold_deg_per_sec,
        distance=_PEAK_DISTANCE,
    )

    starts = _backtrack_starts(abs_rate, peaks, noise_threshold_deg_per_sec)
    if not starts:
        return np.empty(0, dtype=np.intp)
    return np.unique(np.asarray(starts, dtype=np.intp))


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


def _backtrack_starts(
    abs_rate: npt.NDArray[np.float64],
    peaks: npt.NDArray[np.intp],
    noise_threshold_deg_per_sec: float,
) -> list[int]:
    """Walk back from each peak until rate drops below the noise floor."""
    starts: list[int] = []
    for peak in peaks:
        i = int(peak)
        while i > 0:
            i -= 1
            if abs_rate[i] < noise_threshold_deg_per_sec:
                starts.append(i + 1)
                break
            if i == 0:
                starts.append(0)
                break
    return starts


# ---------------------------------------------------------------------------
# Segment endpoint assignment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_lateral(
    df: pl.DataFrame,
    *,
    dt: float = 4.0,
    threshold_deg_per_sec: float = _DEFAULT_THRESHOLD,
    noise_threshold_deg_per_sec: float = _DEFAULT_NOISE_FLOOR,
) -> pl.DataFrame:
    """Augment a flight DataFrame with a lateral reference track.

    Adds three columns:

    - ``in_turn`` (bool) -- True when the sample has no enclosing
      straight segment (head/tail of flight, or degenerate cases).
    - ``track_ortho`` (deg) -- great-circle bearing from the current
      position to the segment end ``B``.  This is the FMS-equivalent
      lateral target.  NaN inside ``in_turn``.
    - ``track_sel_known`` (bool) -- True iff ``track_ortho`` is finite
      (analogous to ``fdm_tas_target_known``).

    Args:
        df: Single-flight eager DataFrame with ``latitude``,
            ``longitude``, and ``track`` columns (degrees).  Sampling is
            assumed uniform at ``dt`` seconds.
        dt: Sampling interval in seconds.
        threshold_deg_per_sec: Peak detection threshold.
        noise_threshold_deg_per_sec: Backtrack stops below this rate.

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

    turning_starts = detect_turning_starts(
        track_raw,
        dt=dt,
        threshold_deg_per_sec=threshold_deg_per_sec,
        noise_threshold_deg_per_sec=noise_threshold_deg_per_sec,
    )

    a_idx, b_idx = segment_bounds(turning_starts, n)
    in_turn = build_in_turn_mask(turning_starts, a_idx, b_idx, n)

    phi_c = np.radians(lat)
    lam_c = np.radians(lon)
    phi_b = np.radians(lat[b_idx])
    lam_b = np.radians(lon[b_idx])

    ortho_rad = orthodromic_bearing(phi_c, lam_c, phi_b, lam_b)
    ortho_deg = np.degrees(ortho_rad)
    ortho_deg[in_turn] = np.nan

    known = ~in_turn & ~np.isnan(ortho_deg)

    return df.with_columns(
        pl.Series("fdm_in_turn", in_turn),
        pl.Series("fdm_track_ortho_deg", ortho_deg),
        pl.Series("fdm_track_sel_known", known),
    )
