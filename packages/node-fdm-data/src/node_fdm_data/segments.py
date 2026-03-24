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

from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas

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
    y = np.array(values, dtype=np.float64, copy=True)
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


def build_selected_params(  # noqa: PLR0915
    df: pl.DataFrame,
    config: dict[str, Any],
) -> pl.DataFrame:
    """Build selected-parameter columns from segment detection.

    Analyses Mach, CAS, TAS (optional), vertical rate, gamma (optional),
    and altitude (optional) to produce ``fdm_mach_sel``,
    ``fdm_cas_sel_kt``, ``fdm_tas_sel_kt``, ``fdm_vz_sel_ftmin``,
    ``fdm_gamma_sel_rad``, ``fdm_alt_sel_ft`` columns.

    TAS plateaus are masked in regions where Mach or CAS plateaus have
    already been detected, avoiding double-counting.

    Level-flight indicator:

    * ``fdm_gamma_from_alt_rad`` — 0.0 where ``fdm_alt_sel_ft`` is
      detected (altitude plateau / ALT HLD), NaN elsewhere.

    Target columns (backward-filled with last-point anchor):

    * ``fdm_alt_target_ft`` — from altitude segments
    * ``fdm_cas_target_kt`` — from CAS segments
    * ``fdm_tas_target_kt`` — unified TAS target built from
      Mach→TAS (highest priority), CAS→TAS, and TAS_sel segments.

    Args:
        df: Single-flight DataFrame (sorted by time).
        config: Selected-parameter config dict with keys
            ``mach``, ``cas``, ``vz``, and optionally ``tas``,
            ``gamma``, ``alt``.  Each value is a dict of kwargs
            for :func:`detect_constant_segments`.

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
    else:
        cas_segs = []

    # --- TAS selected (optional, mask Mach- and CAS-constant regions) ---
    tas_cfg = config.get("tas")
    tas_col = "era_tas_kt"
    if tas_cfg is not None and tas_col in df.columns:
        tas_arr = df[tas_col].to_numpy().copy()
        for seg in mach_segs:
            tas_arr[seg["start_idx"] : seg["end_idx"] + 1] = np.nan
        for seg in cas_segs:
            tas_arr[seg["start_idx"] : seg["end_idx"] + 1] = np.nan
        tas_segs = detect_constant_segments(tas_arr, **tas_cfg)
        df = add_segment_column(df, tas_segs, "fdm_tas_sel_kt")

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

    # --- Altitude selected (detect level segments on raw_alt_ft) ---
    alt_cfg = config.get("alt")
    if alt_cfg is not None and alt_col in df.columns:
        alt_segs = detect_constant_segments(alt_arr, **alt_cfg)
        df = add_segment_column(df, alt_segs, "fdm_alt_sel_ft")

    # --- Gamma from altitude hold (gamma=0 where alt plateau detected) ---
    if "fdm_alt_sel_ft" in df.columns:
        alt_sel = df["fdm_alt_sel_ft"].to_numpy()
        gamma_from_alt = np.where(np.isnan(alt_sel), np.nan, 0.0)
        df = df.with_columns(pl.Series("fdm_gamma_from_alt_rad", gamma_from_alt))

    # --- MCP altitude backfill (independent of alt segments) ---
    mcp_col = "bds_mcp_sel_alt_ft"
    if mcp_col in df.columns:
        df = df.with_columns(
            pl.col(mcp_col)
            .fill_nan(None)
            .forward_fill()
            .backward_fill()
            .fill_null(pl.lit(float("nan")))
            .alias("fdm_mcp_alt_sel_ft")
        )

    # --- FMS altitude backfill ---
    fms_col = "bds_fms_sel_alt_ft"
    if fms_col in df.columns:
        df = df.with_columns(
            pl.col(fms_col)
            .fill_nan(None)
            .forward_fill()
            .backward_fill()
            .fill_null(pl.lit(float("nan")))
            .alias("fdm_fms_alt_sel_ft")
        )

    # --- Target columns (bfill with last-point anchor) ---
    # fdm_alt_target_ft: "which altitude is the aircraft heading towards?"
    # Anchor last row to actual altitude, then backward-fill from segments.
    if "fdm_alt_sel_ft" in df.columns and alt_col in df.columns:
        last_alt = df[alt_col][-1]
        df = df.with_columns(
            pl.col("fdm_alt_sel_ft").fill_nan(None).alias("_alt_target_tmp"),
        )
        # Set last row to actual altitude, then bfill
        n = len(df)
        target = df["_alt_target_tmp"].to_list()
        target[n - 1] = last_alt
        df = df.with_columns(
            pl.Series("_alt_target_tmp", target).backward_fill().alias("fdm_alt_target_ft"),
        ).drop("_alt_target_tmp")

    # fdm_cas_target_kt: "which CAS is the aircraft heading towards?"
    cas_src = "era_cas_kt" if "era_cas_kt" in df.columns else "bds_ias_kt"
    if "fdm_cas_sel_kt" in df.columns and cas_src in df.columns:
        last_cas = df[cas_src][-1]
        df = df.with_columns(
            pl.col("fdm_cas_sel_kt").fill_nan(None).alias("_cas_target_tmp"),
        )
        n = len(df)
        target = df["_cas_target_tmp"].to_list()
        target[n - 1] = last_cas
        df = df.with_columns(
            pl.Series("_cas_target_tmp", target).backward_fill().alias("fdm_cas_target_kt"),
        ).drop("_cas_target_tmp")

    # fdm_tas_target_kt: "which TAS is the aircraft heading towards?"
    # Combines Mach→TAS (priority), CAS→TAS, TAS_sel, then bfill.
    tas_src = "era_tas_kt"
    if tas_src in df.columns:
        n = len(df)
        tas_target = np.full(n, np.nan)
        alt_m = alt_arr * 0.3048  # ft → m
        _ms_to_kt = 1.0 / 0.514444

        # Layer 1 (lowest priority): TAS_sel segments
        if "fdm_tas_sel_kt" in df.columns:
            sel = df["fdm_tas_sel_kt"].to_numpy()
            mask = ~np.isnan(sel)
            tas_target[mask] = sel[mask]

        # Layer 2: CAS→TAS (overrides TAS_sel)
        if "fdm_cas_sel_kt" in df.columns:
            cas_sel = df["fdm_cas_sel_kt"].to_numpy()
            mask = ~np.isnan(cas_sel)
            if mask.any():
                cas_ms = cas_sel[mask] * 0.514444
                tas_ms = cas_to_tas(cas_ms, alt_m[mask])
                tas_target[mask] = np.asarray(tas_ms) * _ms_to_kt

        # Layer 3 (highest priority): Mach→TAS
        if "fdm_mach_sel" in df.columns:
            mach_sel = df["fdm_mach_sel"].to_numpy()
            mask = ~np.isnan(mach_sel)
            if mask.any():
                tas_ms = mach_to_tas(mach_sel[mask], alt_m[mask])
                tas_target[mask] = np.asarray(tas_ms) * _ms_to_kt

        # Anchor last row to actual TAS, then backward-fill
        tas_target[n - 1] = df[tas_src][-1]
        df = df.with_columns(
            pl.Series("fdm_tas_target_kt", tas_target)
            .fill_nan(None)
            .backward_fill()
            .alias("fdm_tas_target_kt"),
        )

    return df
