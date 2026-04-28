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
from pydantic import BaseModel
from scipy.signal import savgol_filter

from node_fdm_data.physics.speed import cas_to_tas, mach_to_tas, vz_to_gamma

__all__ = [
    "GammaFilterConfig",
    "add_segment_column",
    "build_selected_params",
    "detect_constant_segments",
]


class GammaFilterConfig(BaseModel):
    """Configuration for gamma segment detection with sensible defaults.

    Adds ``min_abs_value`` (default 0.005 rad) to filter near-zero
    gamma plateaus during cruise that are not meaningful targets.
    """

    tol: float = 0.002
    min_len: int = 15
    use_alt: bool = False
    min_abs_value: float = 0.005
    smooth_window: int = 5
    smooth_method: str = "savgol"


def _interpolate_nans(y: np.ndarray, nan_mask: np.ndarray) -> np.ndarray | None:
    valid = ~nan_mask
    n_valid = int(valid.sum())
    if n_valid >= 2:  # noqa: PLR2004
        y[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(valid),
            y[valid],
        )
        return y
    if n_valid == 1:
        y[nan_mask] = y[valid][0]
        return y
    return None


def _smooth(y: np.ndarray, window: int, method: str) -> np.ndarray:
    if method == "savgol":
        win = min(window, len(y) - (len(y) % 2 == 0))
        if win >= 3:  # noqa: PLR2004
            smoothed: np.ndarray = savgol_filter(y, window_length=win, polyorder=2, mode="interp")
            return smoothed
        return y
    return np.convolve(y, np.ones(window) / window, mode="same")


def _make_segment(start: int, end_idx: int, y: np.ndarray) -> dict[str, Any]:
    return {
        "start_idx": start,
        "end_idx": end_idx,
        "var_mean": float(np.mean(y[start : end_idx + 1])),
    }


def _prepare_values(
    values: np.ndarray,
    smooth_window: int | None,
    smooth_method: str,
) -> np.ndarray | None:
    y = np.array(values, dtype=np.float64, copy=True)
    nan_mask = np.isnan(y)
    has_nan = bool(nan_mask.any())
    if has_nan:
        interpolated = _interpolate_nans(y, nan_mask)
        if interpolated is None:
            return None
        y = interpolated
    if smooth_window is not None and smooth_window > 1:
        y = _smooth(y, smooth_window, smooth_method)
    if has_nan:
        y[nan_mask] = np.nan
    return y


def _resolve_alt(alt_values: np.ndarray | None, *, use_alt: bool) -> np.ndarray | None:
    if not use_alt:
        return None
    if alt_values is None:
        msg = "alt_values required when use_alt=True"
        raise ValueError(msg)
    return np.asarray(alt_values, dtype=np.float64)


def _point_passes(
    i: int,
    s: bool,
    y: np.ndarray,
    alt: np.ndarray | None,
    alt_threshold: float,
    min_abs_value: float | None,
) -> bool:
    if not s:
        return False
    if alt is not None and alt[i] <= alt_threshold:
        return False
    if min_abs_value is not None and np.abs(y[i]) <= min_abs_value:
        return False
    return True


def _collect_segments(
    y: np.ndarray,
    stable: np.ndarray,
    alt: np.ndarray | None,
    alt_threshold: float,
    min_abs_value: float | None,
    min_len: int,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    start: int | None = None
    for i, s in enumerate(stable):
        if _point_passes(i, bool(s), y, alt, alt_threshold, min_abs_value):
            if start is None:
                start = i
            continue
        if start is not None:
            if i - start >= min_len:
                segments.append(_make_segment(start, i - 1, y))
            start = None
    if start is not None and len(y) - start >= min_len:
        segments.append(_make_segment(start, len(y) - 1, y))
    return segments


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
    y = _prepare_values(values, smooth_window, smooth_method)
    if y is None:
        return []
    alt = _resolve_alt(alt_values, use_alt=use_alt)
    dy = np.abs(np.diff(y))
    stable = np.concatenate(([False], dy < tol))
    return _collect_segments(y, stable, alt, alt_threshold, min_abs_value, min_len)


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


_KT_TO_MS = 0.514444
_MS_TO_KT = 1.0 / 0.514444
_FT_MIN_TO_MS = 0.3048 / 60
_FT_TO_M = 0.3048


def _mask_segments(arr: np.ndarray, segments: list[dict[str, Any]]) -> None:
    for seg in segments:
        arr[seg["start_idx"] : seg["end_idx"] + 1] = np.nan


def _resolve_col(df: pl.DataFrame, primary: str, fallback: str) -> str:
    return primary if primary in df.columns else fallback


def _detect_with_alt(
    df: pl.DataFrame,
    src_col: str,
    out_col: str,
    cfg: dict[str, Any],
    alt_arr: np.ndarray,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    if src_col not in df.columns:
        return df, []
    segs = detect_constant_segments(df[src_col].to_numpy(), alt_values=alt_arr, **cfg)
    return add_segment_column(df, segs, out_col), segs


def _detect_masked(
    df: pl.DataFrame,
    src_col: str,
    out_col: str,
    cfg: dict[str, Any],
    masks: list[list[dict[str, Any]]],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    if src_col not in df.columns:
        return df, []
    arr = df[src_col].to_numpy().copy()
    for mask_segs in masks:
        _mask_segments(arr, mask_segs)
    segs = detect_constant_segments(arr, **cfg)
    return add_segment_column(df, segs, out_col), segs


def _backfill_alias(df: pl.DataFrame, src: str, alias: str) -> pl.DataFrame:
    return df.with_columns(
        pl.col(src)
        .fill_nan(None)
        .forward_fill()
        .backward_fill()
        .fill_null(pl.lit(float("nan")))
        .alias(alias)
    )


def _last_valid(df: pl.DataFrame, col: str, default: float = 0.0) -> float:
    series = df[col].drop_nulls().drop_nans()
    return float(series[-1]) if len(series) > 0 else default


def _anchored_target(
    df: pl.DataFrame,
    sel_col: str,
    src_col: str,
    out_col: str,
) -> pl.DataFrame:
    if sel_col not in df.columns or src_col not in df.columns:
        return df
    last = _last_valid(df, src_col)
    target = df[sel_col].fill_nan(None).to_list()
    target[len(df) - 1] = last
    return df.with_columns(
        pl.Series(out_col, target).cast(pl.Float64).backward_fill().alias(out_col),
    )


def _gamma_from_alt(
    df: pl.DataFrame,
    alt_segs: list[dict[str, Any]],
    alt_cfg: dict[str, Any] | None,
    relax: int,
) -> pl.DataFrame:
    if "fdm_alt_sel_ft" not in df.columns:
        return df
    alt_sel = df["fdm_alt_sel_ft"].to_numpy()
    gamma_from_alt = np.where(np.isnan(alt_sel), np.nan, 0.0)
    if relax > 0 and alt_cfg is not None:
        for seg in alt_segs:
            end = min(seg["start_idx"] + relax, seg["end_idx"] + 1)
            gamma_from_alt[seg["start_idx"] : end] = np.nan
    return df.with_columns(pl.Series("fdm_gamma_from_alt_rad", gamma_from_alt))


def _build_tas_target(df: pl.DataFrame, alt_arr: np.ndarray, tas_src: str) -> pl.DataFrame:
    if tas_src not in df.columns:
        return df
    n = len(df)
    tas_target = np.full(n, np.nan)
    alt_m = alt_arr * _FT_TO_M

    if "fdm_tas_sel_kt" in df.columns:
        sel = df["fdm_tas_sel_kt"].to_numpy()
        mask = ~np.isnan(sel)
        tas_target[mask] = sel[mask]

    if "fdm_cas_sel_kt" in df.columns:
        cas_sel = df["fdm_cas_sel_kt"].to_numpy()
        mask = ~np.isnan(cas_sel)
        if mask.any():
            tas_ms = cas_to_tas(cas_sel[mask] * _KT_TO_MS, alt_m[mask])
            tas_target[mask] = np.asarray(tas_ms) * _MS_TO_KT

    if "fdm_mach_sel" in df.columns:
        mach_sel = df["fdm_mach_sel"].to_numpy()
        mask = ~np.isnan(mach_sel)
        if mask.any():
            tas_ms = mach_to_tas(mach_sel[mask], alt_m[mask])
            tas_target[mask] = np.asarray(tas_ms) * _MS_TO_KT

    tas_target[n - 1] = _last_valid(df, tas_src)
    return df.with_columns(
        pl.Series("fdm_tas_target_kt", tas_target)
        .fill_nan(None)
        .backward_fill()
        .alias("fdm_tas_target_kt"),
    )


def _gamma_layer_from_alt(df: pl.DataFrame, target: np.ndarray) -> None:
    if "fdm_gamma_from_alt_rad" not in df.columns:
        return
    gfa = df["fdm_gamma_from_alt_rad"].to_numpy()
    mask = ~np.isnan(gfa)
    if mask.any():
        target[mask] = 0.0


def _gamma_layer_sel(df: pl.DataFrame, target: np.ndarray) -> None:
    if "fdm_gamma_sel_rad" not in df.columns:
        return
    gamma_sel = df["fdm_gamma_sel_rad"].to_numpy()
    mask = ~np.isnan(gamma_sel)
    if mask.any():
        target[mask] = gamma_sel[mask]


def _gamma_layer_vz(df: pl.DataFrame, target: np.ndarray, tas_col: str) -> None:
    if "fdm_vz_sel_ftmin" not in df.columns or tas_col not in df.columns:
        return
    vz_sel = df["fdm_vz_sel_ftmin"].to_numpy()
    tas_arr_ms = df[tas_col].to_numpy() * _KT_TO_MS
    mask = ~np.isnan(vz_sel)
    if mask.any():
        target[mask] = vz_to_gamma(vz_sel[mask] * _FT_MIN_TO_MS, tas_arr_ms[mask])


def _build_gamma_target(df: pl.DataFrame, tas_col: str) -> pl.DataFrame:
    if "fdm_gamma_rad" not in df.columns:
        return df
    gamma_target = np.full(len(df), np.nan)
    _gamma_layer_from_alt(df, gamma_target)
    _gamma_layer_sel(df, gamma_target)
    _gamma_layer_vz(df, gamma_target, tas_col)
    gamma_known = (~np.isnan(gamma_target)).astype(np.float64)
    gamma_filled = np.where(np.isnan(gamma_target), 0.0, gamma_target)
    return df.with_columns(
        pl.Series("fdm_gamma_target_rad", gamma_filled),
        pl.Series("fdm_gamma_target_known", gamma_known),
    )


def _detect_gamma_sel(df: pl.DataFrame, gamma_cfg: dict[str, Any] | None) -> pl.DataFrame:
    if gamma_cfg is None or "fdm_gamma_rad" not in df.columns:
        return df
    gcfg = GammaFilterConfig(**gamma_cfg)
    gamma_segs = detect_constant_segments(
        df["fdm_gamma_rad"].to_numpy().copy(), **gcfg.model_dump()
    )
    return add_segment_column(df, gamma_segs, "fdm_gamma_sel_rad")


def _detect_alt_sel(
    df: pl.DataFrame,
    alt_cfg: dict[str, Any] | None,
    alt_col: str,
    alt_arr: np.ndarray,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    if alt_cfg is None or alt_col not in df.columns:
        return df, []
    segs = detect_constant_segments(alt_arr, **alt_cfg)
    return add_segment_column(df, segs, "fdm_alt_sel_ft"), segs


def build_selected_params(
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
    * ``fdm_gamma_target_rad`` — unified gamma target built from
      vz→gamma (highest priority), gamma_sel, and gamma_from_alt=0
      (lowest priority, ALT HLD).  NaN-preserving: gaps stay NaN.

    Args:
        df: Single-flight DataFrame (sorted by time).
        config: Selected-parameter config dict with keys
            ``mach``, ``cas``, ``vz``, and optionally ``tas``,
            ``gamma``, ``alt``.  Each value is a dict of kwargs
            for :func:`detect_constant_segments`.

    Returns:
        DataFrame with selected-parameter columns added.
    """
    alt_col = _resolve_col(df, "raw_alt_ft", "altitude")
    alt_arr = df[alt_col].to_numpy()
    tas_col = "era_tas_kt"

    df, mach_segs = _detect_with_alt(
        df,
        _resolve_col(df, "era_mach", "Mach"),
        "fdm_mach_sel",
        config.get("mach", {}),
        alt_arr,
    )
    df, cas_segs = _detect_masked(
        df,
        _resolve_col(df, "bds_ias_kt", "CAS"),
        "fdm_cas_sel_kt",
        config.get("cas", {}),
        [mach_segs],
    )
    tas_cfg = config.get("tas")
    if tas_cfg is not None:
        df, _ = _detect_masked(
            df,
            tas_col,
            "fdm_tas_sel_kt",
            tas_cfg,
            [mach_segs, cas_segs],
        )
    df, _ = _detect_with_alt(
        df,
        _resolve_col(df, "raw_vz_ftmin", "vertical_rate"),
        "fdm_vz_sel_ftmin",
        config.get("vz", {}),
        alt_arr,
    )
    df = _detect_gamma_sel(df, config.get("gamma"))
    alt_cfg = config.get("alt")
    df, alt_segs = _detect_alt_sel(df, alt_cfg, alt_col, alt_arr)

    df = _gamma_from_alt(df, alt_segs, alt_cfg, int(config.get("alt_hold_relax", 15)))

    if "bds_mcp_sel_alt_ft" in df.columns:
        df = _backfill_alias(df, "bds_mcp_sel_alt_ft", "fdm_mcp_alt_sel_ft")
    if "bds_fms_sel_alt_ft" in df.columns:
        df = _backfill_alias(df, "bds_fms_sel_alt_ft", "fdm_fms_alt_sel_ft")

    df = _anchored_target(df, "fdm_alt_sel_ft", alt_col, "fdm_alt_target_ft")
    df = _anchored_target(
        df,
        "fdm_cas_sel_kt",
        _resolve_col(df, "era_cas_kt", "bds_ias_kt"),
        "fdm_cas_target_kt",
    )
    df = _build_tas_target(df, alt_arr, tas_col)
    return _build_gamma_target(df, tas_col)
