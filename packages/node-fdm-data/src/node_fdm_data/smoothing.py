"""1D smoothing primitives shared across the data layer.

Pure numpy/scipy functions — no Polars, no schema, no I/O. Extracted
from :mod:`node_fdm_data.segments` so plateau detectors and any other
consumer can share the same edge-preserving / low-pass building blocks.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt

__all__ = ["bilateral_1d", "butter_lowpass", "interpolate_nans"]


def bilateral_1d(y: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """1D bilateral filter (Tomasi & Manduchi 1998).

    Spatial kernel ``sigma_s`` (samples) flattens homogeneous zones,
    range kernel ``sigma_r`` (y-units) preserves jumps.
    """
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    n = len(y_arr)
    half = int(np.ceil(3 * sigma_s))
    w = 2 * half + 1

    pad = np.zeros(n + 2 * half, dtype=np.float64)
    pad[half : half + n] = y_arr
    windows = sliding_window_view(pad, w)

    rel = np.arange(w)
    abs_idx = np.arange(n)[:, None] - half + rel[None, :]
    valid = (abs_idx >= 0) & (abs_idx < n)

    spatial = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_s) ** 2)
    diff = windows - y_arr[:, None]
    rng_w = np.exp(-0.5 * (diff / sigma_r) ** 2)
    weights = spatial[None, :] * rng_w * valid

    num = np.sum(weights * windows, axis=1)
    den = np.sum(weights, axis=1)
    return np.asarray(num / den, dtype=np.float64)


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
