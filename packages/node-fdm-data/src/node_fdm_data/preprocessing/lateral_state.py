"""Lateral state preprocessing: track cleaning, drift, wind std, heading coalesce.

Helpers used by the data pipeline to build the lateral channel inputs:

- :func:`clean_track_with_medfilt` — median-filter + Savgol smooth on raw GPS track
  (kills isolated spikes <= 2 samples before they leak into Savgol output).
- :func:`compute_drift_from_wind` — signed drift from the ERA5 wind triangle.
- :func:`compute_wind_std` — rolling std of wind magnitude (diagnostic only,
  exposed via ``fdm_wind_std_ms``).
- :func:`coalesce_heading` — primary BDS heading + declination, fallback to
  ``track_clean - drift`` (drift computed from track as a heading proxy)
  whenever BDS is missing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

__all__ = [
    "clean_track_with_medfilt",
    "coalesce_heading",
    "compute_drift_from_wind",
    "compute_wind_std",
]

_SAVGOL_WINDOW: int = 9
_SAVGOL_POLY: int = 3
_FULL_TURN_DEG: float = 360.0
_MIN_STD_SAMPLES: int = 2


def _wrap_unsigned_deg(x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Wrap to ``[0, 360)``."""
    return np.asarray(x, dtype=np.float64) % _FULL_TURN_DEG


def clean_track_with_medfilt(
    track_deg: npt.NDArray[np.floating],
    *,
    medfilt_window: int = 5,
) -> npt.NDArray[np.float64]:
    """Median-filter then unwrap-aware Savgol smooth on a track signal.

    The median filter (window 5 → ~20 s at 4 s sampling) kills isolated GPS
    spikes of <= 2 samples before Savgol smooths them into intermediate
    (wrong) values.  Unwrap is applied in radians so the smoother does not
    fight the 0/360 boundary.

    Args:
        track_deg: Raw track angle in degrees, may contain NaN.
        medfilt_window: Median filter window in samples (must be odd).

    Returns:
        Cleaned track in degrees, wrapped to ``[0, 360)``.
    """
    arr = np.asarray(track_deg, dtype=np.float64)
    n = arr.size
    if n < _SAVGOL_WINDOW:
        return _wrap_unsigned_deg(arr)

    # Forward+backward fill NaN so medfilt and unwrap see a finite signal.
    filled = arr.copy()
    if not np.all(np.isfinite(filled)):
        bad = ~np.isfinite(filled)
        good_idx = np.flatnonzero(~bad)
        if good_idx.size == 0:
            return np.full(n, np.nan, dtype=np.float64)
        last = filled[good_idx[0]]
        for i in range(n):
            if bad[i]:
                filled[i] = last
            else:
                last = filled[i]

    medfiltered = median_filter(filled, size=medfilt_window, mode="nearest")
    unwrapped_rad = np.unwrap(np.radians(medfiltered))
    smoothed_rad = savgol_filter(unwrapped_rad, _SAVGOL_WINDOW, _SAVGOL_POLY)
    return _wrap_unsigned_deg(np.degrees(smoothed_rad))


def compute_drift_from_wind(
    heading_deg: npt.NDArray[np.floating],
    tas_ms: npt.NDArray[np.floating],
    u_wind_ms: npt.NDArray[np.floating],
    v_wind_ms: npt.NDArray[np.floating],
) -> npt.NDArray[np.float64]:
    """Wind-triangle drift in degrees (signed).

    drift = ``atan2(V_wind_cross, TAS + V_wind_along)``

    where ``V_wind_cross`` is the component of wind perpendicular to
    heading (positive = wind from the left, pushes track to the right of
    heading).  Aviation convention: ``u`` = east-component, ``v`` =
    north-component.  Heading 0 = north, 90 = east.

    Args:
        heading_deg: Heading in degrees, shape ``(n,)``.
        tas_ms: True airspeed in m/s, shape ``(n,)``.
        u_wind_ms: East-wind component in m/s, shape ``(n,)``.
        v_wind_ms: North-wind component in m/s, shape ``(n,)``.

    Returns:
        Drift in degrees, signed, shape ``(n,)``.
    """
    psi = np.radians(np.asarray(heading_deg, dtype=np.float64))
    u = np.asarray(u_wind_ms, dtype=np.float64)
    v = np.asarray(v_wind_ms, dtype=np.float64)
    tas = np.asarray(tas_ms, dtype=np.float64)
    along = u * np.sin(psi) + v * np.cos(psi)
    cross = u * np.cos(psi) - v * np.sin(psi)
    return np.degrees(np.arctan2(cross, tas + along))


def compute_wind_std(
    u_wind_ms: npt.NDArray[np.floating],
    v_wind_ms: npt.NDArray[np.floating],
    *,
    window_samples: int = 8,
) -> npt.NDArray[np.float64]:
    """Rolling standard deviation of wind magnitude.

    Diagnostic of ERA5 vertical-interpolation stability — high std means
    the ambient wind estimate is jittery (typically off-cruise).  Exposed
    via ``fdm_wind_std_ms`` for downstream analysis.  No longer used as a
    gate inside :func:`coalesce_heading`.

    Default window is 8 samples = 32 s at 4 s sampling.  Returned series
    has the same length as the input; NaN samples count as zero deviation
    locally.

    Args:
        u_wind_ms: East-wind component, shape ``(n,)``.
        v_wind_ms: North-wind component, shape ``(n,)``.
        window_samples: Rolling window length in samples.

    Returns:
        Per-sample wind-magnitude rolling std, shape ``(n,)``.
    """
    u = np.asarray(u_wind_ms, dtype=np.float64)
    v = np.asarray(v_wind_ms, dtype=np.float64)
    mag = np.hypot(u, v)
    n = mag.size
    if n == 0:
        return mag.astype(np.float64)
    w = max(2, int(window_samples))
    out = np.empty(n, dtype=np.float64)
    half = w // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = mag[lo:hi]
        finite = window[np.isfinite(window)]
        out[i] = float(np.std(finite)) if finite.size >= _MIN_STD_SAMPLES else 0.0
    return out


def coalesce_heading(  # noqa: PLR0913 — six primary signals, all required
    bds_hdg_deg: npt.NDArray[np.floating],
    declination_deg: npt.NDArray[np.floating],
    track_clean_deg: npt.NDArray[np.floating],
    tas_ms: npt.NDArray[np.floating],
    u_wind_ms: npt.NDArray[np.floating],
    v_wind_ms: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Build the final heading signal and known-flag.

    Coalesce strategy:

    1. **Primary**: ``bds_hdg + declination`` (true heading) wherever BDS
       and declination are finite.
    2. **Fallback**: ``track_clean - drift_from_track``, where
       ``drift_from_track`` is the wind-triangle drift computed by feeding
       the cleaned GPS track to :func:`compute_drift_from_wind` as a
       heading proxy.  Substituting ``track`` for the unknown true heading
       in the drift formula is a second-order error (a few 0.1° at typical
       cruise drift of 3-5°), well below the heading noise floor.
    3. ``heading_known = isfinite(heading)``.

    Args:
        bds_hdg_deg: BDS magnetic heading in degrees.  May be NaN.
        declination_deg: Magnetic declination in degrees (added to
            magnetic heading to get true heading).
        track_clean_deg: Cleaned GPS track in degrees.
        tas_ms: True airspeed in m/s.
        u_wind_ms: East-wind component in m/s.
        v_wind_ms: North-wind component in m/s.

    Returns:
        Tuple ``(heading_deg, heading_known)``:

        - ``heading_deg``: shape ``(n,)``, wrapped to ``[0, 360)``.  NaN
          where neither the primary nor the fallback can be evaluated.
        - ``heading_known``: shape ``(n,)``, bool — finite-mask of
          ``heading_deg``.
    """
    bds = np.asarray(bds_hdg_deg, dtype=np.float64)
    decl = np.asarray(declination_deg, dtype=np.float64)
    track = np.asarray(track_clean_deg, dtype=np.float64)

    primary = _wrap_unsigned_deg(bds + decl)
    drift_from_track = compute_drift_from_wind(track, tas_ms, u_wind_ms, v_wind_ms)
    fallback = _wrap_unsigned_deg(track - drift_from_track)

    heading = np.where(np.isfinite(primary), primary, fallback)
    known = np.isfinite(heading)
    return heading, known
