"""Lateral state preprocessing: track cleaning, drift, wind std, heading coalesce.

Helpers used by the data pipeline to build the lateral channel inputs:

- :func:`clean_track_with_medfilt` — median-filter + Savgol smooth on raw GPS track
  (kills isolated spikes <= 2 samples before they leak into Savgol output).
- :func:`compute_drift_from_wind` — signed drift from the ERA5 wind triangle.
- :func:`compute_wind_std` — rolling std of wind magnitude (gate for fallback
  reliability — high std means ERA5 vertical interpolation is unstable).
- :func:`coalesce_heading` — primary BDS heading + declination, fallback to
  ``track - drift`` when BDS missing and wind is stable.
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


def _wrap_signed_deg(x: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """Wrap to ``[-180, 180]``."""
    return ((np.asarray(x, dtype=np.float64) + 180.0) % 360.0) - 180.0


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

    Used as a gate for fallback heading reliability: when ERA5 vertical
    interpolation is unstable (typically off-cruise), the wind triangle
    drift estimate becomes noisy and this rolling std flags it.

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


def coalesce_heading(  # noqa: PLR0913 — five primary signals + threshold kwarg
    bds_hdg_deg: npt.NDArray[np.floating],
    declination_deg: npt.NDArray[np.floating],
    track_clean_deg: npt.NDArray[np.floating],
    drift_deg: npt.NDArray[np.floating],
    wind_std_ms: npt.NDArray[np.floating],
    *,
    wind_std_threshold: float = 5.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Build the final heading signal and known-flag.

    Coalesce strategy:

    1. **Primary**: ``bds_hdg + declination`` (true heading) when BDS is
       finite.
    2. **Fallback**: ``track_clean - drift`` when BDS is missing **and**
       ``wind_std <= threshold`` (ERA5 fallback reliable).
    3. ``known = False`` when both sources are unavailable, including when
       ``wind_std > threshold`` knocks out the fallback.

    Args:
        bds_hdg_deg: BDS magnetic heading in degrees.  May be NaN.
        declination_deg: Magnetic declination in degrees (added to
            magnetic heading to get true heading).
        track_clean_deg: Cleaned GPS track in degrees.
        drift_deg: Wind-triangle drift in degrees (signed).
        wind_std_ms: Rolling std of wind magnitude.  Samples above
            ``wind_std_threshold`` are deemed unreliable for fallback.
        wind_std_threshold: Gate threshold in m/s.

    Returns:
        Tuple ``(heading_deg, heading_known)``:

        - ``heading_deg``: shape ``(n,)``, wrapped to ``[0, 360)``.  NaN
          where neither source is available.
        - ``heading_known``: shape ``(n,)``, bool.
    """
    bds = np.asarray(bds_hdg_deg, dtype=np.float64)
    decl = np.asarray(declination_deg, dtype=np.float64)
    track = np.asarray(track_clean_deg, dtype=np.float64)
    drift = np.asarray(drift_deg, dtype=np.float64)
    wstd = np.asarray(wind_std_ms, dtype=np.float64)

    primary = _wrap_unsigned_deg(bds + decl)
    fallback = _wrap_unsigned_deg(track - drift)

    bds_ok = np.isfinite(bds) & np.isfinite(decl)
    fb_ok = (
        ~bds_ok
        & np.isfinite(track)
        & np.isfinite(drift)
        & np.isfinite(wstd)
        & (wstd <= wind_std_threshold)
    )

    heading = np.full_like(primary, np.nan)
    heading[bds_ok] = primary[bds_ok]
    heading[fb_ok] = fallback[fb_ok]

    known = bds_ok | fb_ok
    return heading, known
