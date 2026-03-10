"""Lateral computations — turning point detection and theoretical track.

Ports ``lateral_computations.py`` from the legacy ``opensky_v2`` branch to
Polars.  Signal-processing helpers remain in numpy/scipy; DataFrame
operations use Polars throughout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.signal import find_peaks, savgol_filter

__all__ = [
    "augment_with_segments",
    "compute_lateral_track",
    "compute_theoretical_track",
    "detect_turning_points",
    "find_segment_bounds",
]

# ---------------------------------------------------------------------------
# Signal helpers (numpy/scipy — no Polars equivalent)
# ---------------------------------------------------------------------------

_WINDOW_LENGTH = 9
_POLYORDER = 3
_WRAP_THRESHOLD = 180
_DT_MIN = 1e-6
_MIN_POINTS_FOR_LATERAL = 2


def detect_turning_points(
    track_deg: npt.NDArray[np.floating[Any]],
    time_s: npt.NDArray[np.floating[Any]],
    *,
    threshold_deg_per_sec: float = 0.05,
    noise_threshold_deg_per_sec: float = 0.005,
) -> npt.NDArray[np.intp]:
    """Detect start-of-turn indices from a track-angle series.

    Uses a Savitzky-Golay filter to smooth the track signal, computes the
    rotation rate (°/s), then walks backwards from each peak to find the
    onset index where the rotation rate drops below the noise threshold.

    Parameters
    ----------
    track_deg:
        Unwrapped track angle in degrees.
    time_s:
        Timestamps in seconds (same length as *track_deg*).
    threshold_deg_per_sec:
        Minimum peak rotation rate to consider a turn (°/s).
    noise_threshold_deg_per_sec:
        Rotation rate below which the signal is considered noise (°/s).

    Returns
    -------
    Sorted, unique array of turn-onset indices.
    """
    if len(track_deg) < _WINDOW_LENGTH:
        return np.array([], dtype=np.intp)

    smoothed = savgol_filter(track_deg, _WINDOW_LENGTH, _POLYORDER)

    d_track = np.diff(smoothed, prepend=smoothed[0])
    d_track = np.where(d_track > _WRAP_THRESHOLD, d_track - 360, d_track)
    d_track = np.where(d_track < -_WRAP_THRESHOLD, d_track + 360, d_track)

    dt = np.diff(time_s, prepend=time_s[0] if len(time_s) > 0 else 1.0)
    dt = np.where(dt < _DT_MIN, _DT_MIN, dt)

    abs_rate = np.abs(d_track / dt)

    peaks, _ = find_peaks(abs_rate, height=threshold_deg_per_sec, distance=10)

    starts: list[int] = []
    for peak in peaks:
        idx = peak
        while idx > 0:
            idx -= 1
            if abs_rate[idx] < noise_threshold_deg_per_sec:
                starts.append(idx + 1)
                break
        else:
            starts.append(0)

    return np.unique(np.asarray(starts, dtype=np.intp))


# ---------------------------------------------------------------------------
# Segment helpers (pure numpy)
# ---------------------------------------------------------------------------


def find_segment_bounds(
    turning_indices: npt.NDArray[np.intp],
    pivot: int,
    total_length: int,
) -> tuple[int, int]:
    """Return the segment ``(start, end)`` that contains *pivot*.

    *turning_indices* must be sorted.  The returned bounds are inclusive
    row indices into the original DataFrame.
    """
    idx_end = int(np.searchsorted(turning_indices, pivot, side="right"))
    idx_start = idx_end - 1

    start = int(turning_indices[idx_start]) if idx_start >= 0 else 0
    end = int(turning_indices[idx_end]) if idx_end < len(turning_indices) else total_length - 1
    return start, end


# ---------------------------------------------------------------------------
# DataFrame helpers (Polars)
# ---------------------------------------------------------------------------


def augment_with_segments(
    df: pl.DataFrame,
    turning_indices: npt.NDArray[np.intp],
) -> pl.DataFrame:
    """Add ``lat_A / lon_A / lat_B / lon_B`` columns from segment endpoints.

    Each row is assigned to the segment delimited by the nearest turning
    indices.  The endpoint coordinates are looked up via ``pl.Series.gather``.
    """
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("lat_A"),
            pl.lit(None, dtype=pl.Float64).alias("lon_A"),
            pl.lit(None, dtype=pl.Float64).alias("lat_B"),
            pl.lit(None, dtype=pl.Float64).alias("lon_B"),
        )

    n = len(df)
    starts = np.empty(n, dtype=np.intp)
    ends = np.empty(n, dtype=np.intp)
    for i in range(n):
        s, e = find_segment_bounds(turning_indices, i, n)
        starts[i] = s
        ends[i] = e

    lat = df["latitude"]
    lon = df["longitude"]

    return df.with_columns(
        lat.gather(starts).alias("lat_A"),
        lon.gather(starts).alias("lon_A"),
        lat.gather(ends).alias("lat_B"),
        lon.gather(ends).alias("lon_B"),
    )


def compute_theoretical_track(df: pl.DataFrame) -> pl.Series:
    """Compute orthodromic (great-circle) bearing toward segment endpoint B.

    Expects columns ``latitude``, ``longitude``, ``lat_B``, ``lon_B`` in
    degrees.  Returns a ``pl.Series`` of bearings in degrees [0, 360).
    """
    lat_c = np.radians(df["latitude"].to_numpy())
    lon_c = np.radians(df["longitude"].to_numpy())
    lat_b = np.radians(df["lat_B"].to_numpy())
    lon_b = np.radians(df["lon_B"].to_numpy())

    d_lon = lon_b - lon_c
    y = np.sin(d_lon) * np.cos(lat_b)
    x = np.cos(lat_c) * np.sin(lat_b) - np.sin(lat_c) * np.cos(lat_b) * np.cos(d_lon)

    bearing_deg = (np.degrees(np.arctan2(y, x)) + 360) % 360
    return pl.Series("track_sel", bearing_deg)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def compute_lateral_track(df: pl.DataFrame) -> pl.DataFrame:
    """Full lateral-track pipeline: detect turns → segment → compute bearings.

    Requires columns ``track`` (degrees), ``timestamp`` (datetime or numeric),
    ``latitude``, and ``longitude``.
    """
    if df.is_empty() or len(df) < _MIN_POINTS_FOR_LATERAL:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias("track_sel"))

    track_raw = df["track"].to_numpy().astype(np.float64)
    track_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(track_raw)))

    ts_col = df["timestamp"]
    if ts_col.dtype in (pl.Datetime, pl.Date, pl.Duration):
        time_s = (
            ((ts_col.cast(pl.Int64) - ts_col.cast(pl.Int64).min()) / 1_000_000)
            .to_numpy()
            .astype(np.float64)
        )
    else:
        time_s = ts_col.to_numpy().astype(np.float64)
        time_s = time_s - time_s[0]

    turning_idx = detect_turning_points(track_unwrapped, time_s)
    df = augment_with_segments(df, turning_idx)
    track_sel = compute_theoretical_track(df)
    return df.with_columns(track_sel)
