"""Segment-based selected parameter estimation.

Detects quasi-constant segments in time series and assigns the
segment mean as the "selected" value for that region.  Used by
the processing pipeline to estimate pilot-selected Mach, CAS,
vertical speed, altitude, and flight-path angle.

All functions operate on **collected** ``pl.DataFrame`` (not
LazyFrame) since segment detection is inherently row-iterative.

Example::

    from node_fdm_data.segments import build_selected_params

    processed = build_selected_params(flight_df, config.model_dump())
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

__all__ = [
    "add_segment_column",
    "build_selected_params",
    "detect_constant_segments",
]


def detect_constant_segments(
    values: np.ndarray,
    *,
    tol: float = 0.002,
    min_len: int = 5,
    alt_values: np.ndarray | None = None,
    alt_threshold: float = 20_000,
    use_alt: bool = True,
    min_abs_value: float | None = None,
    smooth_window: int | None = None,
    smooth_method: str = "rolling",
) -> list[dict[str, Any]]:
    """Detect segments where a variable is quasi-constant.

    Scans the diff of *values* and groups consecutive points whose
    absolute diff is below *tol*.  Segments shorter than *min_len*
    are discarded.

    Args:
        values: 1-D array of the variable to analyse.
        tol: Maximum absolute diff to consider a point stable.
        min_len: Minimum segment length (points) to keep.
        alt_values: Altitude array (feet) for altitude gating.
        alt_threshold: Minimum altitude to accept segments when
            *use_alt* is True.
        use_alt: Whether to apply the altitude gate.
        min_abs_value: If set, discard segments whose mean
            absolute value is below this threshold.
        smooth_window: Optional smoothing window length. ``None``
            disables smoothing.
        smooth_method: ``"rolling"`` or ``"savgol"``.

    Returns:
        List of segment dicts, each with keys ``start_idx``,
        ``end_idx``, ``var_mean``.
    """
    y = np.asarray(values, dtype=np.float64)
    nan_mask = np.isnan(y)

    # Interpolate NaNs before smoothing (savgol can't handle them)
    if nan_mask.any():
        valid = ~nan_mask
        if valid.sum() >= 2:  # noqa: PLR2004
            y[nan_mask] = np.interp(
                np.flatnonzero(nan_mask),
                np.flatnonzero(valid),
                y[valid],
            )
        elif valid.sum() == 1:
            y[nan_mask] = y[valid][0]
        else:
            return []  # all NaN → no segments

    # Smooth if requested
    if smooth_window is not None and smooth_window > 1:
        if smooth_method == "savgol":
            win = min(smooth_window, len(y) - (len(y) % 2 == 0))
            if win >= 3:  # noqa: PLR2004
                y = savgol_filter(y, window_length=win, polyorder=2, mode="interp")
        else:  # rolling
            y = np.convolve(y, np.ones(smooth_window) / smooth_window, mode="same")

    # Restore NaN positions — they break segment continuity
    if nan_mask.any():
        y[nan_mask] = np.nan

    # Altitude array
    alt: np.ndarray | None = None
    if use_alt:
        if alt_values is None:
            msg = "alt_values required when use_alt=True"
            raise ValueError(msg)
        alt = np.asarray(alt_values, dtype=np.float64)

    # Detect stable points
    dy = np.abs(np.diff(y))
    stable = np.concatenate(([False], dy < tol))

    segments: list[dict[str, Any]] = []
    start: int | None = None

    for i, s in enumerate(stable):
        cond_alt = (not use_alt) or (alt is not None and alt[i] > alt_threshold)
        cond_abs = (min_abs_value is None) or (np.abs(y[i]) > min_abs_value)
        cond = s and cond_alt and cond_abs

        if cond and start is None:
            start = i
        elif (not cond) and start is not None:
            if i - start >= min_len:
                segments.append(
                    {
                        "start_idx": start,
                        "end_idx": i - 1,
                        "var_mean": float(np.mean(y[start:i])),
                    }
                )
            start = None

    # Close trailing segment
    if start is not None and len(y) - start >= min_len:
        segments.append(
            {
                "start_idx": start,
                "end_idx": len(y) - 1,
                "var_mean": float(np.mean(y[start:])),
            }
        )

    return segments


def add_segment_column(
    df: pl.DataFrame,
    segments: list[dict[str, Any]],
    col_name: str,
    *,
    fill_value: float | None = None,
) -> pl.DataFrame:
    """Add a column populated with segment mean values.

    Points outside detected segments receive *fill_value* (default
    ``NaN``).

    Args:
        df: Input DataFrame.
        segments: Segment dicts from :func:`detect_constant_segments`.
        col_name: Name of the new column.
        fill_value: Value for non-segment rows. ``None`` → ``NaN``.

    Returns:
        DataFrame with the added column.
    """
    n = len(df)
    arr = np.full(n, fill_value if fill_value is not None else np.nan, dtype=np.float64)
    for seg in segments:
        arr[seg["start_idx"] : seg["end_idx"] + 1] = seg["var_mean"]
    return df.with_columns(pl.Series(col_name, arr))


def build_selected_params(
    df: pl.DataFrame,
    config: dict[str, Any],
) -> pl.DataFrame:
    """Build selected-parameter columns from segment detection.

    Analyses Mach, CAS, vertical rate, gamma (optional), and
    altitude (optional) to produce ``fdm_mach_sel``, ``fdm_cas_sel_kt``,
    ``fdm_vz_sel_ftmin``, ``fdm_gamma_sel_rad``, ``fdm_alt_sel_ft``
    columns.

    Args:
        df: Single-flight DataFrame (sorted by time).
        config: Selected-parameter config dict with keys
            ``mach``, ``cas``, ``vz``, and optionally ``gamma``,
            ``alt``.  Each value is a dict of kwargs for
            :func:`detect_constant_segments`.

    Returns:
        DataFrame with selected-parameter columns added.
    """
    # --- Altitude array (shared across detectors) ---
    alt_col = "raw_alt_ft" if "raw_alt_ft" in df.columns else "altitude"
    alt_arr = df[alt_col].to_numpy()

    # --- Mach selected ---
    mach_col = "era_mach" if "era_mach" in df.columns else "Mach"
    if mach_col in df.columns:
        mach_cfg = config.get("mach", {})
        mach_segs = detect_constant_segments(
            df[mach_col].to_numpy(),
            alt_values=alt_arr,
            **mach_cfg,
        )
        df = add_segment_column(df, mach_segs, "fdm_mach_sel")
    else:
        mach_segs = []

    # --- CAS selected (mask Mach-constant regions first) ---
    cas_col = "bds_ias_kt" if "bds_ias_kt" in df.columns else "CAS"
    if cas_col in df.columns:
        cas_arr = df[cas_col].to_numpy().copy()
        # Null out CAS in Mach-constant regions
        for seg in mach_segs:
            cas_arr[seg["start_idx"] : seg["end_idx"] + 1] = np.nan
        cas_cfg = config.get("cas", {})
        cas_segs = detect_constant_segments(cas_arr, **cas_cfg)
        df = add_segment_column(df, cas_segs, "fdm_cas_sel_kt")

    # --- Vz selected ---
    vz_col = "raw_vz_ftmin" if "raw_vz_ftmin" in df.columns else "vertical_rate"
    if vz_col in df.columns:
        vz_cfg = config.get("vz", {})
        vz_segs = detect_constant_segments(
            df[vz_col].to_numpy(),
            **vz_cfg,
        )
        df = add_segment_column(df, vz_segs, "fdm_vz_sel_ftmin")
    else:
        vz_segs = []

    # --- Gamma selected (optional, mask Vz-constant regions) ---
    gamma_cfg = config.get("gamma")
    if gamma_cfg is not None and "fdm_gamma_rad" in df.columns:
        gamma_arr = df["fdm_gamma_rad"].to_numpy().copy()
        for seg in vz_segs:
            gamma_arr[seg["start_idx"] : seg["end_idx"] + 1] = np.nan
        gamma_segs = detect_constant_segments(gamma_arr, **gamma_cfg)
        df = add_segment_column(df, gamma_segs, "fdm_gamma_sel_rad")

    # --- Altitude selected (optional, backfill) ---
    alt_cfg = config.get("alt")
    if alt_cfg is not None:
        alt_segs = detect_constant_segments(
            alt_arr,
            **alt_cfg,
        )
        df = add_segment_column(df, alt_segs, "fdm_alt_sel_ft")
        # Backfill: last point inherits actual altitude, then bfill
        mcp = df["fdm_alt_sel_ft"].to_list()
        mcp[-1] = float(alt_arr[-1])
        # Backward fill
        for i in range(len(mcp) - 2, -1, -1):
            if np.isnan(mcp[i]):
                mcp[i] = mcp[i + 1]
        df = df.with_columns(pl.Series("fdm_mcp_alt_sel_ft", mcp))

    # --- Fill NaN with 0.0 for segment-detected columns (legacy parity) ---
    # Non-segment timesteps → 0.0 means "no active selection" for the model.
    sel_cols = [
        c
        for c in ("fdm_mach_sel", "fdm_cas_sel_kt", "fdm_vz_sel_ftmin", "fdm_gamma_sel_rad")
        if c in df.columns
    ]
    if sel_cols:
        df = df.with_columns(pl.col(c).fill_nan(0.0).fill_null(0.0) for c in sel_cols)

    return df
