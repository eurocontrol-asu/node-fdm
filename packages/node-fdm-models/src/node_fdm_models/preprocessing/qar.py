"""QAR preprocessing pipeline.

Signal processing (Butterworth filter, moving-average smoothing,
mode stabilisation) and engine parameter reduction.  Signal-level
functions operate on NumPy arrays; the ``flight_processing`` wrapper
handles the Polars ↔ NumPy bridge.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.signal import butter, filtfilt

__all__ = [
    "filter_noise",
    "flight_processing",
    "mode_stabilize",
    "reduce_engine",
    "smooth_strong",
]

# ---------------------------------------------------------------------------
# Signal-level helpers (NumPy)
# ---------------------------------------------------------------------------

_ENGINE_LOW_THR: float = 5.0
"""N1 threshold below which an engine is considered shut down."""

_ENGINE_DIFF_THR: float = 5.0
"""N1 difference threshold for asymmetric-engine detection."""


def filter_noise(
    values: np.ndarray[Any, np.dtype[Any]],
    fs: float = 1.0,
    cutoff: float = 0.04,
    order: int = 4,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Low-pass Butterworth filter.

    Args:
        values: Input signal.
        fs: Sampling frequency (Hz).
        cutoff: Cut-off frequency (Hz).
        order: Filter order.

    Returns:
        Filtered signal as a NumPy array.
    """
    nyq = fs / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    result: np.ndarray[Any, np.dtype[Any]] = filtfilt(b, a, values)
    return result


def smooth_strong(
    values: np.ndarray[Any, np.dtype[Any]],
    window_size: int = 100,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Moving-average smoothing with reflection padding.

    Args:
        values: Input signal.
        window_size: Width of the averaging window.

    Returns:
        Smoothed signal (same length as input).
    """
    x = np.asarray(values, dtype=float)
    half = window_size // 2
    x_padded = np.pad(x, pad_width=(half, half - 1), mode="reflect")
    kernel = np.ones(window_size) / window_size
    result: np.ndarray[Any, np.dtype[Any]] = np.convolve(x_padded, kernel, mode="valid")
    return result


def mode_stabilize(
    values: np.ndarray[Any, np.dtype[Any]],
    min_duration: int = 10,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Suppress transient categorical state changes.

    A new state is only accepted after it persists for at least
    *min_duration* consecutive samples.

    Args:
        values: Input categorical signal.
        min_duration: Minimum dwell time before accepting a new state.

    Returns:
        Stabilised signal as a NumPy array.
    """
    result = np.array(values, copy=True)
    current = result[0]
    count = 0
    for i in range(1, len(result)):
        if result[i] == current:
            count = 0
        else:
            count += 1
            if count < min_duration:
                result[i] = current
            else:
                current = result[i]
                count = 0
    return result


# ---------------------------------------------------------------------------
# Polars-level helpers
# ---------------------------------------------------------------------------


def reduce_engine(
    df: pl.LazyFrame,
    col_left: str,
    col_right: str,
    *,
    output_col: str = "n1",
) -> pl.LazyFrame:
    """Reduce symmetric engine parameters to a single column.

    When both engines are running normally (N1 ≥ threshold and difference
    ≤ threshold), the result is the **average**.  Otherwise, the
    **maximum** is used to represent the active engine.

    Args:
        df: Input LazyFrame with engine columns.
        col_left: Column name for the left-engine metric.
        col_right: Column name for the right-engine metric.
        output_col: Name of the output column (default ``"n1"``).

    Returns:
        LazyFrame with the reduced engine column appended.
    """
    is_low = (pl.col(col_left) < _ENGINE_LOW_THR) | (pl.col(col_right) < _ENGINE_LOW_THR)
    is_diff = (pl.col(col_left) - pl.col(col_right)).abs() > _ENGINE_DIFF_THR
    is_asymmetric = is_low & is_diff

    avg = (pl.col(col_left) + pl.col(col_right)) / 2
    mx = pl.max_horizontal(col_left, col_right)

    return df.with_columns(
        pl.when(is_asymmetric).then(mx).otherwise(avg).alias(output_col),
    )


def flight_processing(
    df: pl.LazyFrame,
    *,
    step: int = 4,
) -> pl.LazyFrame:
    """Full QAR preprocessing pipeline.

    The pipeline applies:
    1. TAS null-fill from groundspeed
    2. Back/forward-fill remaining nulls
    3. Remove on-ground rows
    4. Downsample by *step*

    .. note::

       Heavy signal processing (Butterworth, smoothing) is architecture-
       specific.  Use :func:`filter_noise`, :func:`smooth_strong`, and
       :func:`mode_stabilize` individually as needed before calling this
       function for the final cleanup.

    Args:
        df: LazyFrame containing raw QAR columns.
        step: Downsampling factor applied after processing.

    Returns:
        Cleaned and downsampled LazyFrame.
    """
    lf = df.with_columns(
        pl.col("SPD__TAS").fill_null(pl.col("SPD__GND")),
    )

    lf = lf.fill_null(strategy="backward").fill_null(strategy="forward")

    # Remove on-ground samples if the column exists
    lf = lf.filter(pl.col("ENG__ON_GROUND") == 0)

    # Downsample
    lf = lf.with_row_index("__row_idx").filter(pl.col("__row_idx") % step == 0).drop("__row_idx")

    return lf
