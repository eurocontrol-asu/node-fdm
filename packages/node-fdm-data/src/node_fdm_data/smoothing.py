"""1D smoothing primitives shared across the data layer.

Pure numpy/scipy functions — no Polars, no schema, no I/O. Extracted
from :mod:`node_fdm_data.segments` so plateau detectors and any other
consumer can share the same edge-preserving / low-pass building blocks.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

__all__ = ["bilateral_1d", "butter_lowpass", "interpolate_nans"]


def bilateral_1d(y: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """1D bilateral filter (Tomasi & Manduchi 1998).

    Spatial kernel ``sigma_s`` (samples) flattens homogeneous zones,
    range kernel ``sigma_r`` (y-units) preserves jumps.
    """
    n = len(y)
    half = int(np.ceil(3 * sigma_s))
    out = np.empty_like(y)
    spatial = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_s) ** 2)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        ys = y[a:b]
        sp_w = spatial[a - (i - half) : b - (i - half)]
        rng_w = np.exp(-0.5 * ((ys - y[i]) / sigma_r) ** 2)
        w = sp_w * rng_w
        out[i] = float(np.sum(w * ys) / np.sum(w))
    return out


def interpolate_nans(y: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in ``y``.

    Returns a new ``float64`` array. Behaviour:

    * ≥ 2 valid samples → linear interpolation across NaN gaps;
      leading/trailing NaNs are filled with the nearest endpoint.
    * exactly 1 valid sample → that value propagates to every row.
    * 0 valid samples → array of zeros (same shape as ``y``).
    """
    out = np.asarray(y, dtype=np.float64).copy()
    nan_mask = np.isnan(out)
    valid = ~nan_mask
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.zeros_like(out)
    if n_valid == 1:
        out[nan_mask] = out[valid][0]
        return out
    out[nan_mask] = np.interp(
        np.flatnonzero(nan_mask),
        np.flatnonzero(valid),
        out[valid],
    )
    return out


def butter_lowpass(
    y: np.ndarray,
    cutoff_s: float,
    dt: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter via ``filtfilt``.

    NaNs are linearly interpolated up front. Returns ``y`` unchanged when
    the signal is too short for the requested filter order.
    """
    y_arr = np.asarray(y, dtype=np.float64).copy()
    if len(y_arr) < 2 * order:
        return y_arr
    if np.isnan(y_arr).any():
        y_arr = interpolate_nans(y_arr)
    nyq = 0.5 / dt
    wn = min(0.99, (1.0 / cutoff_s) / nyq)
    b, a = butter(order, wn, btype="low")
    return np.asarray(filtfilt(b, a, y_arr), dtype=np.float64)
