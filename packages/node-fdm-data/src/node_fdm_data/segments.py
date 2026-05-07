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

from node_fdm_data.physics.isa import isa_temperature
from node_fdm_data.physics.speed import (
    cas_to_tas_real,
    mach_to_tas_real,
    tas_to_cas_real,
    vz_to_gamma,
)
from node_fdm_data.smoothing import bilateral_1d, butter_lowpass, interpolate_nans

__all__ = [
    "GammaFilterConfig",
    "add_segment_column",
    "build_selected_params",
    "detect_alt_hold_from_vz",
    "detect_cas_plateaus_bilat",
    "detect_constant_segments",
    "detect_gamma_plateaus_from_bilat",
    "detect_mach_plateaus_bilat",
    "detect_vz_plateaus_from_bilat",
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


def _smooth_vz_bilateral(
    vz: np.ndarray,
    sigma_s: float,
    sigma_r: float,
    n_passes: int,
) -> np.ndarray | None:
    """Interpolate NaNs then apply ``n_passes`` of 1D bilateral smoothing.

    Returns ``None`` if ``vz`` is fully NaN (no signal to recover).
    """
    nan_mask = np.isnan(vz)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return None
        vz = interpolate_nans(vz)
    for _ in range(max(0, int(n_passes))):
        vz = bilateral_1d(vz, sigma_s, sigma_r)
    return vz


def _below_tol_mask(values: np.ndarray, tol: float) -> np.ndarray:
    """Boolean mask of samples whose absolute value is strictly below ``tol``."""
    mask: np.ndarray = np.abs(values) < tol
    return mask


def detect_alt_hold_from_vz(
    vz_ftmin: np.ndarray,
    alt_ft: np.ndarray,
    *,
    sigma_s: float = 6.0,
    sigma_r: float = 350.0,
    n_passes: int = 2,
    tol_ftmin: float = 400.0,
    min_len: int = 6,
) -> list[dict[str, Any]]:
    """Detect altitude-hold plateaus from vertical speed.

    Linearly interpolates NaNs in ``vz_ftmin`` (returns ``[]`` if fully
    NaN), applies ``n_passes`` of a 1D bilateral filter, then keeps
    runs where ``|vz_bilat| < tol_ftmin`` of length ``>= min_len``.
    Each segment's ``var_mean`` is the mean of ``alt_ft`` over the run
    so the result can feed :func:`add_segment_column` directly into
    ``fdm_alt_sel_ft``.
    """
    vz = np.asarray(vz_ftmin, dtype=np.float64).copy()
    alt = np.asarray(alt_ft, dtype=np.float64)
    vz_bilat = _smooth_vz_bilateral(vz, sigma_s, sigma_r, n_passes)
    if vz_bilat is None:
        return []
    mask = _below_tol_mask(vz_bilat, tol_ftmin)
    segments: list[dict[str, Any]] = []
    start: int | None = None
    n = len(mask)
    for i in range(n):
        if mask[i]:
            if start is None:
                start = i
            continue
        if start is not None:
            if i - start >= min_len:
                segments.append(_make_segment(start, i - 1, alt))
            start = None
    if start is not None and n - start >= min_len:
        segments.append(_make_segment(start, n - 1, alt))
    return segments


def _bilateral_plateau_runs(
    y_bilat: np.ndarray,
    flat: np.ndarray,
    flat_tol: float,
    min_len: int,
    abs_min: float | None,
) -> list[dict[str, Any]]:
    """Walk runs of ``flat=True`` and accept those satisfying tolerances."""
    segments: list[dict[str, Any]] = []
    n = len(flat)
    start: int | None = None

    def _try_emit(s: int, e_inclusive: int) -> None:
        if e_inclusive - s + 1 < min_len:
            return
        chunk = y_bilat[s : e_inclusive + 1]
        if (chunk.max() - chunk.min()) > 2.0 * flat_tol:
            return
        mean = float(np.mean(chunk))
        if abs_min is not None and abs(mean) < abs_min:
            return
        segments.append({"start_idx": s, "end_idx": e_inclusive, "var_mean": mean})

    for i in range(n):
        if flat[i]:
            if start is None:
                start = i
            continue
        if start is not None:
            _try_emit(start, i - 1)
            start = None
    if start is not None:
        _try_emit(start, n - 1)
    return segments


def detect_gamma_plateaus_from_bilat(
    gamma_raw: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    sigma_s: float,
    sigma_r: float,
    slope_tol: float,
    flat_tol: float,
    abs_min: float,
    min_len: int,
) -> list[dict[str, Any]]:
    """Detect gamma plateaus from a bilateral-smoothed gamma signal.

    The exclusion mask (True = exclude) lets the caller enforce cascade
    priority (e.g. mask out alt-hold regions before searching for gamma
    plateaus). Returns segment dicts ``{start_idx, end_idx, var_mean}``
    where ``var_mean`` is the mean of the bilateral gamma over the run.
    """
    g = np.asarray(gamma_raw, dtype=np.float64).copy()
    nan_mask = np.isnan(g)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return []
        g = interpolate_nans(g)
    g_bilat = bilateral_1d(bilateral_1d(g, sigma_s, sigma_r), sigma_s, sigma_r)
    dgamma = np.abs(np.diff(g_bilat, prepend=g_bilat[0]))
    excl = np.asarray(exclusion_mask, dtype=bool)
    flat = (dgamma < slope_tol) & (~excl)
    return _bilateral_plateau_runs(g_bilat, flat, flat_tol, min_len, abs_min)


def _mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Walk a boolean mask and return inclusive [start, end] runs of True."""
    runs: list[tuple[int, int]] = []
    n = len(mask)
    start: int | None = None
    for i in range(n):
        if mask[i]:
            if start is None:
                start = i
            continue
        if start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    return runs


def _passes_alt_gate(
    start: int,
    end: int,
    alt_segs: list[tuple[int, int]],
    alt_mask: np.ndarray,
) -> bool:
    """Endpoint inside any alt plateau OR plateau ``[start, end]`` fully
    contains an alt plateau."""
    if alt_mask[start] or alt_mask[min(end, len(alt_mask) - 1)]:
        return True
    return any(start <= a and b <= end for a, b in alt_segs)


def detect_mach_plateaus_bilat(
    mach_raw: np.ndarray,
    alt_plateau_mask: np.ndarray,
    *,
    sigma_s: float,
    sigma_r: float,
    n_passes: int = 2,
    slope_tol: float,
    flat_tol: float,
    min_len: int,
) -> list[dict[str, Any]]:
    """Detect Mach plateaus from a bilateral-smoothed Mach signal.

    Plateaus are kept only when their endpoint sits inside an altitude
    plateau (per ``alt_plateau_mask``) OR they fully contain at least one
    altitude plateau. ``var_mean`` is the mean of the **raw** Mach over
    the run (not the bilateral-smoothed value).
    """
    raw = np.asarray(mach_raw, dtype=np.float64)
    work = raw.copy()
    nan_mask = np.isnan(work)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return []
        work = interpolate_nans(work)
    smooth = work
    for _ in range(max(0, int(n_passes))):
        smooth = bilateral_1d(smooth, sigma_s, sigma_r)
    dmach = np.abs(np.diff(smooth, prepend=smooth[0]))
    flat = dmach < slope_tol
    alt_mask = np.asarray(alt_plateau_mask, dtype=bool)
    alt_segs = _mask_to_segments(alt_mask)
    segments: list[dict[str, Any]] = []
    n = len(smooth)
    i = 0
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flat[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            seg = smooth[i : j + 1]
            if (seg.max() - seg.min()) <= flat_tol and _passes_alt_gate(i, j, alt_segs, alt_mask):
                segments.append(
                    {
                        "start_idx": i,
                        "end_idx": j,
                        "var_mean": float(np.nanmean(raw[i : j + 1])),
                    }
                )
        i = j + 1
    return segments


def detect_cas_plateaus_bilat(
    cas_raw: np.ndarray,
    mach_mask: np.ndarray,
    *,
    cutoff_s: float,
    sigma_s: float,
    sigma_r: float,
    n_passes: int = 2,
    slope_tol: float,
    flat_tol: float,
    min_len: int,
) -> list[dict[str, Any]]:
    """Detect CAS plateaus from a Butterworth+bilateral-smoothed CAS signal.

    Samples flagged in ``mach_mask`` are excluded so the detector does not
    fire inside Mach-plateau zones. ``var_mean`` is the mean of the raw
    CAS over the run.
    """
    raw = np.asarray(cas_raw, dtype=np.float64)
    work = raw.copy()
    nan_mask = np.isnan(work)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return []
        work = interpolate_nans(work)
    work = butter_lowpass(work, cutoff_s)
    smooth = work
    for _ in range(max(0, int(n_passes))):
        smooth = bilateral_1d(smooth, sigma_s, sigma_r)
    dcas = np.abs(np.diff(smooth, prepend=smooth[0]))
    excl = np.asarray(mach_mask, dtype=bool)
    flat = (dcas < slope_tol) & (~excl)
    segments: list[dict[str, Any]] = []
    n = len(smooth)
    i = 0
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flat[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            seg = smooth[i : j + 1]
            if (seg.max() - seg.min()) <= flat_tol:
                segments.append(
                    {
                        "start_idx": i,
                        "end_idx": j,
                        "var_mean": float(np.nanmean(raw[i : j + 1])),
                    }
                )
        i = j + 1
    return segments


def detect_vz_plateaus_from_bilat(
    vz_ftmin: np.ndarray,
    exclusion_mask: np.ndarray,
    *,
    sigma_s: float,
    sigma_r: float,
    slope_tol: float,
    flat_tol: float,
    min_len: int,
) -> list[dict[str, Any]]:
    """Detect vz plateaus from a bilateral-smoothed vz signal."""
    vz = np.asarray(vz_ftmin, dtype=np.float64).copy()
    nan_mask = np.isnan(vz)
    if nan_mask.any():
        if int((~nan_mask).sum()) == 0:
            return []
        vz = interpolate_nans(vz)
    vz_bilat = bilateral_1d(bilateral_1d(vz, sigma_s, sigma_r), sigma_s, sigma_r)
    dvz = np.abs(np.diff(vz_bilat, prepend=vz_bilat[0]))
    excl = np.asarray(exclusion_mask, dtype=bool)
    flat = (dvz < slope_tol) & (~excl)
    return _bilateral_plateau_runs(vz_bilat, flat, flat_tol, min_len, abs_min=None)


def _prepare_values(
    values: np.ndarray,
    smooth_window: int | None,
    smooth_method: str,
) -> np.ndarray | None:
    y = np.array(values, dtype=np.float64, copy=True)
    nan_mask = np.isnan(y)
    has_nan = bool(nan_mask.any())
    if has_nan:
        if int((~nan_mask).sum()) == 0:
            return None
        y = interpolate_nans(y)
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


_CFG_ALIASES = {"min_length": "min_len", "tolerance": "tol"}


def _normalize_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map ticket-spec aliases (``min_length``/``tolerance``) to the canonical kwargs."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        out[_CFG_ALIASES.get(key, key)] = value
    return out


def _detect_mach_in_plateau(
    df: pl.DataFrame,
    src_col: str,
    out_col: str,
    cfg: dict[str, Any],
    alt_arr: np.ndarray,
    plateau_mask: np.ndarray,
    min_mach_value: float,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Detect Mach plateaus restricted to altitude-plateau rows only.

    Dispatches on ``cfg['mode']``:
    - ``"bilateral_mach"`` (AXM-1689) — bilateral-smoothed detector with
      altitude-plateau gate.
    - ``"savgol_mach"`` — legacy detector: NaN out non-plateau rows then
      run :func:`detect_constant_segments`.
    """
    if src_col not in df.columns:
        return df, []
    cfg_norm = _normalize_cfg(cfg)
    mode = cfg_norm.pop("mode", "savgol_mach")
    if mode == "bilateral_mach":
        kwargs = {k: cfg_norm[k] for k in _MACH_BILATERAL_KEYS if k in cfg_norm}
        raw = df[src_col].to_numpy().astype(np.float64, copy=True)
        segs = detect_mach_plateaus_bilat(raw, plateau_mask, **kwargs)
        segs = [s for s in segs if s["var_mean"] >= min_mach_value]
        return add_segment_column(df, segs, out_col), segs
    arr = df[src_col].to_numpy().astype(np.float64, copy=True)
    arr[~plateau_mask] = np.nan
    legacy_cfg = {k: v for k, v in cfg_norm.items() if k not in _MACH_BILATERAL_KEYS}
    segs = detect_constant_segments(arr, alt_values=alt_arr, **legacy_cfg)
    segs = [s for s in segs if s["var_mean"] >= min_mach_value]
    return add_segment_column(df, segs, out_col), segs


def _detect_cas_dispatch(
    df: pl.DataFrame,
    src_col: str,
    out_col: str,
    cfg: dict[str, Any],
    mach_segs: list[dict[str, Any]],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Detect CAS plateaus, dispatching on ``cfg['mode']``.

    - ``"bilateral_cas"`` (AXM-1689) — Butterworth+bilateral detector.
    - ``"savgol_cas"`` — legacy :func:`_detect_masked` path.
    """
    if src_col not in df.columns:
        return df, []
    cfg_norm = _normalize_cfg(cfg)
    mode = cfg_norm.pop("mode", "savgol_cas")
    if mode == "bilateral_cas":
        n = len(df)
        mach_mask = np.zeros(n, dtype=bool)
        for seg in mach_segs:
            mach_mask[seg["start_idx"] : seg["end_idx"] + 1] = True
        kwargs = {k: cfg_norm[k] for k in _CAS_BILATERAL_KEYS if k in cfg_norm}
        raw = df[src_col].to_numpy().astype(np.float64, copy=True)
        segs = detect_cas_plateaus_bilat(raw, mach_mask, **kwargs)
        return add_segment_column(df, segs, out_col), segs
    legacy_cfg = {k: v for k, v in cfg_norm.items() if k not in _CAS_BILATERAL_KEYS}
    return _detect_masked(df, src_col, out_col, legacy_cfg, [mach_segs])


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
    norm = _normalize_cfg(cfg)
    norm.setdefault("use_alt", False)
    segs = detect_constant_segments(arr, **norm)
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
    if sel_col not in df.columns or src_col not in df.columns or len(df) == 0:
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


def _resolve_temp_k(df: pl.DataFrame, alt_arr: np.ndarray) -> np.ndarray:
    """Return the static-temperature array (K) used for speed conversions.

    Uses ``era_temp_K`` when present (real reanalysis temperature); falls
    back to ISA ``isa_temperature(alt_m)`` otherwise.
    """
    if "era_temp_K" in df.columns:
        return df["era_temp_K"].to_numpy().astype(np.float64)
    alt_m = np.asarray(alt_arr, dtype=np.float64) * _FT_TO_M
    return np.asarray(isa_temperature(alt_m), dtype=np.float64)


def _build_tas_target(df: pl.DataFrame, alt_arr: np.ndarray) -> pl.DataFrame:
    """Build ``fdm_tas_target_kt`` from FMS envelope and emit ``fdm_tas_target_known``.

    On rows covered by both Mach and CAS segments, target is
    ``min(mach_to_tas_real, cas_to_tas_real)``; on single-source rows it
    is that single segment's TAS; rows covered by neither stay NaN
    (no global backward-fill).  ``fdm_tas_target_known`` is a boolean
    column that is True iff the target is non-NaN.
    """
    n = len(df)
    target = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return df.with_columns(
            pl.Series("fdm_tas_target_kt", target),
            pl.Series("fdm_tas_target_known", np.zeros(0, dtype=bool)),
        )

    alt_m = np.asarray(alt_arr, dtype=np.float64) * _FT_TO_M
    temp_k = _resolve_temp_k(df, alt_arr)

    tas_mach = np.full(n, np.nan)
    if "fdm_mach_sel" in df.columns:
        mach_sel = df["fdm_mach_sel"].to_numpy()
        mask = ~np.isnan(mach_sel)
        if mask.any():
            tas_ms = mach_to_tas_real(mach_sel[mask], temp_k[mask])
            tas_mach[mask] = np.asarray(tas_ms) * _MS_TO_KT

    tas_cas = np.full(n, np.nan)
    if "fdm_cas_sel_kt" in df.columns:
        cas_sel = df["fdm_cas_sel_kt"].to_numpy()
        mask = ~np.isnan(cas_sel)
        if mask.any():
            tas_ms = cas_to_tas_real(cas_sel[mask] * _KT_TO_MS, alt_m[mask], temp_k[mask])
            tas_cas[mask] = np.asarray(tas_ms) * _MS_TO_KT

    have_mach = ~np.isnan(tas_mach)
    have_cas = ~np.isnan(tas_cas)
    overlap = have_mach & have_cas
    only_mach = have_mach & ~have_cas
    only_cas = ~have_mach & have_cas
    target[overlap] = np.minimum(tas_mach[overlap], tas_cas[overlap])
    target[only_mach] = tas_mach[only_mach]
    target[only_cas] = tas_cas[only_cas]

    if "fdm_tas_sel_kt" in df.columns:
        sel = df["fdm_tas_sel_kt"].to_numpy()
        gap = np.isnan(target) & ~np.isnan(sel)
        target[gap] = sel[gap]

    known = ~np.isnan(target)
    df = df.with_columns(
        pl.Series("fdm_tas_target_kt", target),
        pl.Series("fdm_tas_target_known", known),
    )
    return df.with_columns(
        pl.col("fdm_tas_target_kt")
        .fill_nan(None)
        .forward_fill()
        .backward_fill()
        .fill_null(pl.lit(float("nan")))
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


_GAMMA_BILATERAL_KEYS = {
    "sigma_s",
    "sigma_r",
    "slope_tol",
    "flat_tol",
    "abs_min",
    "min_len",
}

_VZ_BILATERAL_KEYS = {
    "sigma_s",
    "sigma_r",
    "slope_tol",
    "flat_tol",
    "min_len",
}


def _gamma_raw_from_vz_tas(df: pl.DataFrame) -> np.ndarray | None:
    if "raw_vz_ftmin" not in df.columns:
        return None
    tas_col = "fdm_tas_from_cas_kt" if "fdm_tas_from_cas_kt" in df.columns else None
    if tas_col is None and "fdm_tas_kt" in df.columns:
        tas_col = "fdm_tas_kt"
    if tas_col is None:
        return None
    vz_ms = df["raw_vz_ftmin"].to_numpy().astype(np.float64) * _FT_MIN_TO_MS
    tas_ms = df[tas_col].to_numpy().astype(np.float64) * _KT_TO_MS
    return np.arcsin(np.clip(vz_ms / np.clip(tas_ms, 1e-6, None), -1.0, 1.0))


def _alt_hold_mask(df: pl.DataFrame) -> np.ndarray:
    if "fdm_alt_sel_ft" in df.columns:
        return np.asarray(~np.isnan(df["fdm_alt_sel_ft"].to_numpy()), dtype=bool)
    return np.zeros(len(df), dtype=bool)


def _detect_gamma_sel(df: pl.DataFrame, gamma_cfg: dict[str, Any] | None) -> pl.DataFrame:
    if gamma_cfg is None:
        return df
    cfg = _normalize_cfg(gamma_cfg)
    mode = cfg.pop("mode", "savgol_gamma")
    if mode == "bilateral_gamma":
        gamma_raw = _gamma_raw_from_vz_tas(df)
        if gamma_raw is None:
            return df
        kwargs = {k: cfg[k] for k in _GAMMA_BILATERAL_KEYS if k in cfg}
        segs = detect_gamma_plateaus_from_bilat(gamma_raw, _alt_hold_mask(df), **kwargs)
        return add_segment_column(df, segs, "fdm_gamma_sel_rad")
    if "fdm_gamma_rad" not in df.columns:
        return df
    legacy_cfg = {k: v for k, v in cfg.items() if k not in _GAMMA_BILATERAL_KEYS - {"min_len"}}
    gcfg = GammaFilterConfig(**legacy_cfg)
    gamma_segs = detect_constant_segments(
        df["fdm_gamma_rad"].to_numpy().copy(), **gcfg.model_dump()
    )
    return add_segment_column(df, gamma_segs, "fdm_gamma_sel_rad")


def _detect_vz_sel(
    df: pl.DataFrame,
    vz_cfg: dict[str, Any],
    vz_col: str,
    alt_arr: np.ndarray,
) -> pl.DataFrame:
    if vz_col not in df.columns:
        return df
    cfg = _normalize_cfg(vz_cfg)
    mode = cfg.pop("mode", "savgol_vz")
    if mode == "bilateral_vz":
        alt_hold = _alt_hold_mask(df)
        gamma_mask = (
            ~np.isnan(df["fdm_gamma_sel_rad"].to_numpy())
            if "fdm_gamma_sel_rad" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        exclusion = alt_hold | gamma_mask
        kwargs = {k: cfg[k] for k in _VZ_BILATERAL_KEYS if k in cfg}
        segs = detect_vz_plateaus_from_bilat(df[vz_col].to_numpy(), exclusion, **kwargs)
        return add_segment_column(df, segs, "fdm_vz_sel_ftmin")
    legacy_cfg = {k: v for k, v in cfg.items() if k not in _VZ_BILATERAL_KEYS - {"min_len"}}
    segs = detect_constant_segments(df[vz_col].to_numpy(), alt_values=alt_arr, **legacy_cfg)
    return add_segment_column(df, segs, "fdm_vz_sel_ftmin")


_BILATERAL_KEYS = {"sigma_s", "sigma_r", "n_passes", "tol_ftmin", "min_len"}

_MACH_BILATERAL_KEYS = {
    "sigma_s",
    "sigma_r",
    "n_passes",
    "slope_tol",
    "flat_tol",
    "min_len",
}

_CAS_BILATERAL_KEYS = {
    "cutoff_s",
    "sigma_s",
    "sigma_r",
    "n_passes",
    "slope_tol",
    "flat_tol",
    "min_len",
}


def _detect_alt_sel(
    df: pl.DataFrame,
    alt_cfg: dict[str, Any] | None,
    alt_col: str,
    alt_arr: np.ndarray,
    vz_col: str | None = None,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Detect altitude-hold segments, dispatching on ``alt_cfg['mode']``.

    - ``"bilateral_vz"`` — derive alt-hold from vertical speed via
      :func:`detect_alt_hold_from_vz` (requires ``vz_col``).
    - ``"savgol_alt"`` (default) — legacy :func:`detect_constant_segments`
      on the altitude column with bilateral-only keys stripped.
    """
    if alt_cfg is None or alt_col not in df.columns:
        return df, []
    cfg = _normalize_cfg(alt_cfg)
    mode = cfg.pop("mode", "savgol_alt")
    if mode == "bilateral_vz":
        if vz_col is None or vz_col not in df.columns:
            return df, []
        vz_arr = df[vz_col].to_numpy()
        bilateral_kwargs = {k: cfg[k] for k in _BILATERAL_KEYS if k in cfg}
        segs = detect_alt_hold_from_vz(vz_arr, alt_arr, **bilateral_kwargs)
        return add_segment_column(df, segs, "fdm_alt_sel_ft"), segs
    # Legacy savgol_alt path — drop bilateral-only keys.
    legacy_cfg = {k: v for k, v in cfg.items() if k not in _BILATERAL_KEYS}
    legacy_cfg.setdefault("use_alt", False)
    segs = detect_constant_segments(alt_arr, **legacy_cfg)
    return add_segment_column(df, segs, "fdm_alt_sel_ft"), segs


def _optimize_transition_cas(
    tas_real_kt: np.ndarray,
    mach_const: float,
    alt_m: np.ndarray,
    temp_k: np.ndarray,
    *,
    search_window: int = 60,
) -> float:
    """Find the constant CAS (kt) minimising envelope-vs-real TAS error.

    Cost ``= Sum (min(mach_to_tas_real(M, T), cas_to_tas_real(CAS, h, T)) - TAS_real)^2``
    is evaluated over a brute search 200..350 kt @ 1 kt then refined +/-2 kt @ 0.1 kt.
    Returns ``nan`` if no valid samples are present.
    """
    valid = ~np.isnan(tas_real_kt) & ~np.isnan(alt_m) & ~np.isnan(temp_k)
    if int(valid.sum()) < 2:  # noqa: PLR2004
        return float("nan")
    tas_real_ms = tas_real_kt[valid] * _KT_TO_MS
    h = alt_m[valid]
    t = temp_k[valid]

    tas_mach_ms = np.asarray(mach_to_tas_real(mach_const, t), dtype=np.float64)

    def grid_cost(grid_kt: np.ndarray) -> np.ndarray:
        cas_ms = grid_kt[:, None] * _KT_TO_MS
        tas_cas_ms = np.asarray(cas_to_tas_real(cas_ms, h[None, :], t[None, :]), dtype=np.float64)
        env = np.minimum(tas_mach_ms[None, :], tas_cas_ms)
        diff = env - tas_real_ms[None, :]
        return np.asarray(np.nansum(diff * diff, axis=1), dtype=np.float64)

    coarse = np.arange(200.0, 351.0, 1.0)
    coarse_costs = grid_cost(coarse)
    if not np.any(np.isfinite(coarse_costs)):
        return float("nan")
    best = float(coarse[int(np.argmin(coarse_costs))])
    fine = np.arange(best - 2.0, best + 2.0 + 0.05, 0.1)
    fine_costs = grid_cost(fine)
    return float(fine[int(np.argmin(fine_costs))])


def _mach_value_in_alt_seg(
    mach_segs: list[dict[str, Any]],
    alt_seg: dict[str, Any],
) -> float | None:
    """Return the mean Mach of any Mach segment overlapping *alt_seg*, else None."""
    a0, a1 = alt_seg["start_idx"], alt_seg["end_idx"]
    for ms in mach_segs:
        m0, m1 = ms["start_idx"], ms["end_idx"]
        if m0 <= a1 and m1 >= a0:
            return float(ms["var_mean"])
    return None


def _walk_apply_cas(
    cas_sel: np.ndarray,
    cas_real: np.ndarray,
    cas_opt: float,
    *,
    boundary_idx: int,
    direction: int,
    deviation_kt: float,
    win_start: int,
    win_end: int,
) -> None:
    """Walk from *boundary_idx* outward, marking ``cas_sel`` with *cas_opt*.

    Walking stops on the first row where ``|cas_real - cas_opt| > deviation_kt``;
    rows beyond the stop point keep their previous value (NaN after reset).
    NaN ``cas_real`` rows are skipped without terminating the walk.
    """
    if direction < 0:
        rng = range(boundary_idx - 1, win_start - 1, -1)
    else:
        rng = range(boundary_idx, win_end)
    for i in rng:
        r = cas_real[i]
        if np.isnan(r):
            continue
        if abs(r - cas_opt) > deviation_kt:
            break
        cas_sel[i] = cas_opt


def _climb_window(
    first_alt: dict[str, Any], search_window: int, margin: int
) -> tuple[int, int, int] | None:
    """Return ``(boundary, win_start, win_end)`` for the climb side, or None."""
    if first_alt["start_idx"] < margin:
        return None
    boundary = first_alt["start_idx"]
    win_start = max(0, boundary - search_window)
    return boundary, win_start, boundary


def _descent_window(
    last_alt: dict[str, Any], n: int, search_window: int, margin: int
) -> tuple[int, int, int] | None:
    """Return ``(boundary, win_start, win_end)`` for the descent side, or None."""
    if (n - 1 - last_alt["end_idx"]) < margin:
        return None
    boundary = last_alt["end_idx"] + 1
    win_end = min(n, boundary + search_window)
    return boundary, boundary, win_end


def _apply_one_transition_window(
    *,
    cas_sel: np.ndarray,
    cas_real: np.ndarray,
    tas_real: np.ndarray,
    alt_m: np.ndarray,
    temp_k: np.ndarray,
    mach_const: float,
    boundary_idx: int,
    win_start: int,
    win_end: int,
    direction: int,
    deviation_kt: float,
    search_window: int,
) -> None:
    """Clear-window, optimise crossover CAS, and walk-apply on a single side."""
    if direction < 0:
        cas_sel[:boundary_idx] = np.nan
    else:
        cas_sel[boundary_idx:] = np.nan
    cas_opt = _optimize_transition_cas(
        tas_real[win_start:win_end],
        mach_const,
        alt_m[win_start:win_end],
        temp_k[win_start:win_end],
        search_window=search_window,
    )
    if not np.isfinite(cas_opt):
        return
    _walk_apply_cas(
        cas_sel,
        cas_real,
        cas_opt,
        boundary_idx=boundary_idx,
        direction=direction,
        deviation_kt=deviation_kt,
        win_start=win_start,
        win_end=win_end,
    )


def _apply_transition_optimisation(
    df: pl.DataFrame,
    alt_segs: list[dict[str, Any]],
    mach_segs: list[dict[str, Any]],
    alt_arr: np.ndarray,
    cas_src_col: str,
    config: dict[str, Any],
) -> pl.DataFrame:
    """Replace climb/descent CAS in ``fdm_cas_sel_kt`` by the optimised crossover CAS.

    Inside each transition window the previous CAS-detection values are
    cleared, then re-emitted only on rows that match the optimised CAS
    within the deviation cutoff (walking outward from the cruise boundary).
    Skipped if no Mach plateau, no altitude plateau, no real TAS, or the
    plateau is within ``transition_margin`` of a flight edge.
    """
    if "fdm_cas_sel_kt" not in df.columns or not alt_segs or not mach_segs:
        return df
    n = len(df)
    if n == 0 or "fdm_tas_from_cas_kt" not in df.columns:
        return df
    tas_real = df["fdm_tas_from_cas_kt"].to_numpy().astype(np.float64)
    if not np.any(~np.isnan(tas_real)):
        return df
    if cas_src_col not in df.columns:
        return df

    deviation_kt = float(config.get("cas_deviation_kt", 5.0))
    margin = int(config.get("transition_margin", 30))
    search_window = int(config.get("cas_search_window", 60))

    alt_m = np.asarray(alt_arr, dtype=np.float64) * _FT_TO_M
    temp_k = _resolve_temp_k(df, alt_arr)
    cas_real = df[cas_src_col].to_numpy().astype(np.float64)
    cas_sel = df["fdm_cas_sel_kt"].to_numpy().astype(np.float64).copy()

    first_alt, last_alt = alt_segs[0], alt_segs[-1]

    sides: list[tuple[tuple[int, int, int] | None, dict[str, Any], int]] = [
        (_climb_window(first_alt, search_window, margin), first_alt, -1),
        (_descent_window(last_alt, n, search_window, margin), last_alt, +1),
    ]
    for window, alt_seg, direction in sides:
        if window is None:
            continue
        mach_const = _mach_value_in_alt_seg(mach_segs, alt_seg)
        if mach_const is None:
            continue
        boundary, win_start, win_end = window
        _apply_one_transition_window(
            cas_sel=cas_sel,
            cas_real=cas_real,
            tas_real=tas_real,
            alt_m=alt_m,
            temp_k=temp_k,
            mach_const=mach_const,
            boundary_idx=boundary,
            win_start=win_start,
            win_end=win_end,
            direction=direction,
            deviation_kt=deviation_kt,
            search_window=search_window,
        )

    return df.with_columns(pl.Series("fdm_cas_sel_kt", cas_sel))


def _load_speed_column(df: pl.DataFrame, col: str, n: int) -> np.ndarray:
    """Return ``col`` as a writable float64 array, or NaN-filled length ``n``."""
    if col in df.columns:
        return df[col].to_numpy().astype(np.float64, copy=True)
    return np.full(n, np.nan)


def _apply_mach_plateau(
    seg: dict[str, Any],
    temp_k: np.ndarray,
    alt_m: np.ndarray,
    cas_sel: np.ndarray,
    tas_sel: np.ndarray,
) -> None:
    """Fill CAS/TAS arrays in place from a constant-Mach plateau and local atmosphere."""
    s, e = seg["start_idx"], seg["end_idx"] + 1
    mach_const = float(seg["var_mean"])
    t_loc = temp_k[s:e]
    h_loc = alt_m[s:e]
    tas_ms = np.asarray(mach_to_tas_real(np.full_like(t_loc, mach_const), t_loc), dtype=np.float64)
    cas_ms = np.asarray(tas_to_cas_real(tas_ms, h_loc, t_loc), dtype=np.float64)
    cas_sel[s:e] = cas_ms * _MS_TO_KT
    tas_sel[s:e] = tas_ms * _MS_TO_KT


def _apply_cas_plateau(
    seg: dict[str, Any],
    temp_k: np.ndarray,
    alt_m: np.ndarray,
    mach_sel: np.ndarray,
    tas_sel: np.ndarray,
) -> None:
    """Fill Mach/TAS arrays in place from a constant-CAS plateau and local atmosphere."""
    s, e = seg["start_idx"], seg["end_idx"] + 1
    cas_const_ms = float(seg["var_mean"]) * _KT_TO_MS
    t_loc = temp_k[s:e]
    h_loc = alt_m[s:e]
    cas_arr = np.full_like(t_loc, cas_const_ms)
    tas_ms = np.asarray(cas_to_tas_real(cas_arr, h_loc, t_loc), dtype=np.float64)
    a_local = np.sqrt(1.4 * 287.05287 * t_loc)
    mach_sel[s:e] = tas_ms / a_local
    tas_sel[s:e] = tas_ms * _MS_TO_KT


def _propagate_speed_plateaus(
    df: pl.DataFrame,
    mach_segs: list[dict[str, Any]],
    cas_segs: list[dict[str, Any]],
    alt_arr: np.ndarray,
) -> pl.DataFrame:
    """Pointwise-propagate speed plateaus across Mach/CAS/TAS columns.

    Within each Mach plateau the constant Mach combined with the local
    static temperature yields a varying TAS and CAS. Within each CAS
    plateau the constant CAS combined with local altitude+temperature
    yields a varying TAS and Mach. Existing per-segment values written
    by :func:`add_segment_column` are overwritten with the pointwise
    series so downstream consumers (``_build_tas_target``) see the full
    physical envelope (AXM-1689).
    """
    n = len(df)
    if n == 0 or (not mach_segs and not cas_segs):
        return df
    alt_m = np.asarray(alt_arr, dtype=np.float64) * _FT_TO_M
    temp_k = _resolve_temp_k(df, alt_arr)

    mach_sel = _load_speed_column(df, "fdm_mach_sel", n)
    cas_sel = _load_speed_column(df, "fdm_cas_sel_kt", n)
    tas_sel = _load_speed_column(df, "fdm_tas_sel_kt", n)

    for seg in mach_segs:
        _apply_mach_plateau(seg, temp_k, alt_m, cas_sel, tas_sel)
    for seg in cas_segs:
        _apply_cas_plateau(seg, temp_k, alt_m, mach_sel, tas_sel)

    columns = _collect_propagated_columns(
        df, mach_sel, cas_sel, tas_sel, has_any_seg=bool(mach_segs or cas_segs)
    )
    return df.with_columns(*columns)


def _collect_propagated_columns(
    df: pl.DataFrame,
    mach_sel: np.ndarray,
    cas_sel: np.ndarray,
    tas_sel: np.ndarray,
    *,
    has_any_seg: bool,
) -> list[pl.Series]:
    """Build the propagated speed columns to merge back into the frame."""
    columns: list[pl.Series] = []
    if "fdm_mach_sel" in df.columns or has_any_seg:
        columns.append(pl.Series("fdm_mach_sel", mach_sel))
    if "fdm_cas_sel_kt" in df.columns or has_any_seg:
        columns.append(pl.Series("fdm_cas_sel_kt", cas_sel))
    columns.append(pl.Series("fdm_tas_sel_kt", tas_sel))
    return columns


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
    tas_col = "fdm_tas_from_cas_kt"
    cas_src_col = "bds_ias_kt_clean"

    # 1. Altitude plateaus FIRST — Mach detection is restricted to these rows.
    alt_cfg = config.get("alt")
    vz_col = _resolve_col(df, "raw_vz_ftmin", "vertical_rate")
    df, alt_segs = _detect_alt_sel(df, alt_cfg, alt_col, alt_arr, vz_col=vz_col)
    if alt_cfg is None:
        # Backwards compatible: without an alt config we cannot derive the
        # plateau mask, so Mach detection falls back to its altitude-gated
        # behaviour without plateau restriction.
        plateau_mask = np.ones(len(df), dtype=bool)
    else:
        plateau_mask = np.zeros(len(df), dtype=bool)
        for seg in alt_segs:
            plateau_mask[seg["start_idx"] : seg["end_idx"] + 1] = True

    min_mach_value = float(config.get("mach_min_value", 0.5))
    df, mach_segs = _detect_mach_in_plateau(
        df,
        "bds_mach_clean",
        "fdm_mach_sel",
        config.get("mach", {}),
        alt_arr,
        plateau_mask,
        min_mach_value,
    )
    df, cas_segs = _detect_cas_dispatch(
        df,
        cas_src_col,
        "fdm_cas_sel_kt",
        config.get("cas", {}),
        mach_segs,
    )
    df = _propagate_speed_plateaus(df, mach_segs, cas_segs, alt_arr)
    tas_cfg = config.get("tas")
    if tas_cfg is not None:
        df, _ = _detect_masked(
            df,
            tas_col,
            "fdm_tas_sel_kt",
            tas_cfg,
            [mach_segs, cas_segs],
        )
    df = _detect_gamma_sel(df, config.get("gamma"))
    df = _detect_vz_sel(
        df,
        config.get("vz", {}),
        _resolve_col(df, "raw_vz_ftmin", "vertical_rate"),
        alt_arr,
    )

    df = _apply_transition_optimisation(
        df,
        alt_segs,
        mach_segs,
        alt_arr,
        cas_src_col,
        config,
    )

    df = _gamma_from_alt(df, alt_segs, alt_cfg, int(config.get("alt_hold_relax", 15)))

    if "bds_mcp_alt_sel_ft" in df.columns:
        df = _backfill_alias(df, "bds_mcp_alt_sel_ft", "fdm_mcp_alt_sel_ft")
    if "bds_fms_alt_sel_ft" in df.columns:
        df = _backfill_alias(df, "bds_fms_alt_sel_ft", "fdm_fms_alt_sel_ft")

    df = _anchored_target(df, "fdm_alt_sel_ft", alt_col, "fdm_alt_target_ft")
    df = _anchored_target(
        df,
        "fdm_cas_sel_kt",
        "bds_ias_kt_clean",
        "fdm_cas_target_kt",
    )
    df = _build_tas_target(df, alt_arr)
    return _build_gamma_target(df, tas_col)
