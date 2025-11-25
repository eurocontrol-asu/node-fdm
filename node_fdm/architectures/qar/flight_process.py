#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-processing utilities for QAR flight data, including smoothing and filtering."""

from typing import Any, Callable, Sequence, Union

import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt

from utils.data.column import Column
from utils.physics.constants import (
    gamma_ratio,
    R,
)

from node_fdm.architectures.qar.columns import (
    col_alt_diff,
    col_spd_diff,
    col_alt_sel,
    col_spd_sel,
    col_cas,
    col_alt,
    col_gs,
    col_mach,
    col_temp,
    col_tas,
    col_gamma,
    col_on_ground,
    col_n1_left,
    col_n1_right,
    col_ff_left,
    col_ff_right,
    col_reduce_engine,
    col_ff,
    col_n1,
    col_vz_sel,
    col_runway_elev,
    col_fma_2,
    col_dist_to_thr,
    col_mass,
)


def mode_stabilize(series: pd.Series, min_duration: int = 10) -> pd.Series:
    """Stabilize a categorical series by enforcing a minimum dwell time per state.

    Args:
        series: Input categorical series to smooth.
        min_duration: Minimum consecutive occurrences required before accepting a new mode.

    Returns:
        Series with transient changes suppressed.
    """
    stable_series = series.copy()
    current_mode = stable_series.iloc[0]
    count = 0
    for i in range(1, len(stable_series)):
        if stable_series.iloc[i] == current_mode:
            count = 0
        else:
            count += 1
            if count < min_duration:
                stable_series.iloc[i] = current_mode
            else:
                current_mode = stable_series.iloc[i]
                count = 0
    return stable_series


def one_reduce_eng_value(col_left: Column, col_right: Column) -> Callable[[Any], Any]:
    """Return a reducer that averages symmetric engine values unless an engine is flagged.

    Args:
        col_left: Column identifier for the left engine metric.
        col_right: Column identifier for the right engine metric.

    Returns:
        Callable that maps a row-like object to a single engine value.
    """

    def one_reduce_value_curry(el: Any) -> Any:
        if el[col_reduce_engine]:
            return max(el[col_left], el[col_right])
        else:
            return (el[col_left] + el[col_right]) / 2

    return one_reduce_value_curry


def reduce_engine_vectorized(
    df: pd.DataFrame,
    col_left: Column,
    col_right: Column,
    col_reduce_engine: Column,
) -> pd.Series:
    """Combine symmetric engine signals into a single representative series.

    Args:
        df: DataFrame containing engine measurements.
        col_left: Column identifier for the left engine metric.
        col_right: Column identifier for the right engine metric.
        col_reduce_engine: Column indicating when to rely on the higher engine value.

    Returns:
        Series representing the reduced engine metric.
    """
    mask = df[col_reduce_engine] != 0
    result = pd.Series(index=df.index, dtype=float)
    result[mask] = df[[col_left, col_right]].loc[mask].max(axis=1)
    result[~mask] = df[[col_left, col_right]].loc[~mask].mean(axis=1)

    return result


def engine_process(df: pd.DataFrame) -> pd.DataFrame:
    """Flag asymmetric engine states and aggregate engine-related columns.

    Args:
        df: DataFrame containing engine measurements.

    Returns:
        DataFrame with additional reduced-engine columns.
    """

    reduce = (df[col_n1_left] < 5) | (df[col_n1_right] < 5)
    diff = np.abs(df[col_n1_left] - df[col_n1_right]) > 5
    df[col_reduce_engine] = (reduce * diff).astype(int)
    df[col_ff] = df[col_ff_left] + df[col_ff_right]
    df[col_n1] = reduce_engine_vectorized(
        df, col_n1_left, col_n1_right, col_reduce_engine
    )
    return df


def smooth_strong(
    x: Union[pd.Series, Sequence[float], np.ndarray],
    window_size: int = 100,
) -> np.ndarray:
    """Apply a strong moving-average smoothing window to a sequence.

    Args:
        x: Sequence of numeric values to smooth.
        window_size: Width of the averaging window.

    Returns:
        Smoothed NumPy array.
    """

    x = np.asarray(x, dtype=float)
    half = window_size // 2

    x_padded = np.pad(x, pad_width=(half, half - 1), mode="reflect")

    kernel = np.ones(window_size) / window_size
    y = np.convolve(x_padded, kernel, mode="valid")

    return y


def filter_noise(
    df_col: Union[pd.Series, Sequence[float], np.ndarray],
    fs: float = 1.0,
    cutoff: float = 0.04,
    order: int = 4,
) -> np.ndarray:
    """Filter high-frequency noise from a signal using a low-pass Butterworth filter.

    Args:
        df_col: Input signal.
        fs: Sampling frequency of the signal.
        cutoff: Cutoff frequency for the low-pass filter.
        order: Order of the Butterworth filter.

    Returns:
        Filtered signal as a NumPy array.
    """

    b, a = butter(order, cutoff / (fs / 2), btype="low")
    signal_filtre = filtfilt(b, a, df_col)
    return signal_filtre


def flight_processing(df: pd.DataFrame, step: int = 4) -> pd.DataFrame:
    """Prepare flight data for training by smoothing, filtering, and resampling.

    Args:
        df: Raw flight data.
        step: Downsampling step applied after processing.

    Returns:
        Processed and downsampled DataFrame.
    """

    df[col_tas] = df[col_tas].fillna(df[col_gs])
    speed_of_sound = np.sqrt(gamma_ratio * R * df[col_temp])
    df[col_mach] = df[col_tas] / speed_of_sound

    df[col_gamma] = np.arcsin(
        df[col_gs] * np.sin(df[col_gamma]) / df[col_tas]
    )  # Compute Gamma_air with no vertical wind assumption
    df[col_gamma] = df[col_gamma].fillna(0)

    smooth_gamma_rad = smooth_strong(df[col_gamma])
    smooth_vz_sel = np.tan(smooth_gamma_rad) * df[col_gs]
    df[col_vz_sel] = mode_stabilize(df[col_vz_sel])
    df[col_vz_sel] = (
        df.fma_col_2_category.isin([6, 11, 12, 13, 14, 15]) * smooth_vz_sel
        + df.fma_col_2_category.isin([10]) * df[col_vz_sel]
    )

    alt_cond1 = df[col_alt].diff(5).fillna(0) > 500
    alt_cond2 = df[col_alt].diff(-5).fillna(0) > 500
    prod_cond = alt_cond1 * alt_cond2
    prod_cond = prod_cond.apply(lambda el: np.nan if el == 1 else 1)
    df[col_alt] = df[col_alt] * prod_cond
    df[col_alt] = df[col_alt].where(df[col_on_ground] == 0).bfill().ffill()

    df[col_runway_elev] = df[col_alt].where(df[col_on_ground] == 1).bfill()
    df[col_alt_sel] = np.where(
        df[col_fma_2].isin([12, 13, 14, 15]), df[col_runway_elev], df[col_alt_sel]
    )
    df[col_alt_sel] = np.where(
        df[col_dist_to_thr] < 3, df[col_runway_elev], df[col_alt_sel]
    )

    df = engine_process(df)

    df[col_mass.derivative] = -df[col_ff]

    df[col_alt_diff] = df[col_alt_sel] - df[col_alt]
    df[col_spd_diff] = df[col_spd_sel] - df[col_cas]

    df = df.bfill().ffill()
    df = df[df[col_on_ground] == 0]
    df = df.iloc[::step]
    return df


def segment_filtering(f: Any, start_idx: int, seq_len: int) -> bool:
    """Stub for segment filtering to keep compatibility with processing pipelines."""

    return True
