#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-processing utilities for OpenSky 2025 flight data."""

import pandas as pd

from node_fdm.architectures.opensky_2025.columns import (
    col_alt_diff,
    col_alt_sel,
    col_alt,
    col_dist,
)

LOW_THR = 200  # meters
UPPER_THR = 3000  # meters


def flight_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare OpenSky flight data by computing altitude differences.

    Args:
        df: Input DataFrame containing flight measurements.

    Returns:
        DataFrame with altitude difference column added.
    """
    df[col_alt_diff] = df[col_alt_sel] - df[col_alt]

    return df


def segment_filtering(f: pd.DataFrame, start_idx: int, seq_len: int) -> bool:
    """Check whether a segment meets distance variation thresholds.

    Args:
        f: DataFrame containing flight measurements.
        start_idx: Starting index of the segment to evaluate.
        seq_len: Length of the segment to evaluate.

    Returns:
        True if the segment stays within distance thresholds, otherwise False.
    """
    dist_diff = f[col_dist].diff(1)
    seg = dist_diff.iloc[start_idx : start_idx + seq_len]
    condition = len(seg[(seg < LOW_THR) | (seg > UPPER_THR)]) == 0
    return condition


selected_param_config = {
    "mach": {"tol": 0.002, "min_len": 120, "alt_threshold": 20000, "use_alt": True},
    "cas": {
        "tol": 1.0,
        "min_len": 60,
        "use_alt": False,
        "smooth_window": 15,
        "smooth_method": "savgol",
    },
    "vz": {
        "tol": 25,
        "min_len": 30,
        "use_alt": False,
        "min_abs_value": 50,
        "smooth_window": 15,
        "smooth_method": "savgol",
    },
    "add_alt": False,
}
