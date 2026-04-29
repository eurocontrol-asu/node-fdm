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
    vz_to_gamma,
)

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


_CFG_ALIASES = {"min_length": "min_len", "tolerance": "tol"}


def _normalize_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map ticket-spec aliases (``min_length``/``tolerance``) to the canonical kwargs."""
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        out[_CFG_ALIASES.get(key, key)] = value
    return out


def _detect_with_alt(
    df: pl.DataFrame,
    src_col: str,
    out_col: str,
    cfg: dict[str, Any],
    alt_arr: np.ndarray,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    if src_col not in df.columns:
        return df, []
    segs = detect_constant_segments(
        df[src_col].to_numpy(), alt_values=alt_arr, **_normalize_cfg(cfg)
    )
    return add_segment_column(df, segs, out_col), segs


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

    Mach values outside the altitude-plateau mask are set to NaN before
    detection so detected segments cannot extend beyond cruise.  Detected
    segments whose mean Mach is below ``min_mach_value`` are dropped
    (aberrant/ghost-data guard).
    """
    if src_col not in df.columns:
        return df, []
    arr = df[src_col].to_numpy().astype(np.float64, copy=True)
    arr[~plateau_mask] = np.nan
    segs = detect_constant_segments(arr, alt_values=alt_arr, **_normalize_cfg(cfg))
    segs = [s for s in segs if s["var_mean"] >= min_mach_value]
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


def _build_tas_target(df: pl.DataFrame, alt_arr: np.ndarray, tas_src: str) -> pl.DataFrame:
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
    return df.with_columns(
        pl.Series("fdm_tas_target_kt", target),
        pl.Series("fdm_tas_target_known", known),
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
    gcfg = GammaFilterConfig(**_normalize_cfg(gamma_cfg))
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
    cfg = _normalize_cfg(alt_cfg)
    cfg.setdefault("use_alt", False)
    segs = detect_constant_segments(alt_arr, **cfg)
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

    def cost(cas_kt: float) -> float:
        tas_cas_ms = np.asarray(cas_to_tas_real(cas_kt * _KT_TO_MS, h, t), dtype=np.float64)
        env = np.minimum(tas_mach_ms, tas_cas_ms)
        diff = env - tas_real_ms
        return float(np.nansum(diff * diff))

    coarse = np.arange(200.0, 351.0, 1.0)
    coarse_costs = np.array([cost(c) for c in coarse])
    if not np.any(np.isfinite(coarse_costs)):
        return float("nan")
    best = float(coarse[int(np.argmin(coarse_costs))])
    fine = np.arange(best - 2.0, best + 2.0 + 0.05, 0.1)
    fine_costs = np.array([cost(c) for c in fine])
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
    if n == 0 or "bds_tas_from_cas_kt" not in df.columns:
        return df
    tas_real = df["bds_tas_from_cas_kt"].to_numpy().astype(np.float64)
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

    # Climb window
    if first_alt["start_idx"] >= margin:
        mach_const = _mach_value_in_alt_seg(mach_segs, first_alt)
        if mach_const is not None:
            boundary = first_alt["start_idx"]
            win_start = max(0, boundary - search_window)
            cas_sel[:boundary] = np.nan
            cas_opt = _optimize_transition_cas(
                tas_real[win_start:boundary],
                mach_const,
                alt_m[win_start:boundary],
                temp_k[win_start:boundary],
                search_window=search_window,
            )
            if np.isfinite(cas_opt):
                _walk_apply_cas(
                    cas_sel,
                    cas_real,
                    cas_opt,
                    boundary_idx=boundary,
                    direction=-1,
                    deviation_kt=deviation_kt,
                    win_start=win_start,
                    win_end=boundary,
                )

    # Descent window
    if (n - 1 - last_alt["end_idx"]) >= margin:
        mach_const = _mach_value_in_alt_seg(mach_segs, last_alt)
        if mach_const is not None:
            boundary = last_alt["end_idx"] + 1
            win_end = min(n, boundary + search_window)
            cas_sel[boundary:] = np.nan
            cas_opt = _optimize_transition_cas(
                tas_real[boundary:win_end],
                mach_const,
                alt_m[boundary:win_end],
                temp_k[boundary:win_end],
                search_window=search_window,
            )
            if np.isfinite(cas_opt):
                _walk_apply_cas(
                    cas_sel,
                    cas_real,
                    cas_opt,
                    boundary_idx=boundary,
                    direction=+1,
                    deviation_kt=deviation_kt,
                    win_start=boundary,
                    win_end=win_end,
                )

    return df.with_columns(pl.Series("fdm_cas_sel_kt", cas_sel))


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
    tas_col = "bds_tas_from_cas_kt"
    cas_src_col = "bds_ias_kt_clean"

    # 1. Altitude plateaus FIRST — Mach detection is restricted to these rows.
    alt_cfg = config.get("alt")
    df, alt_segs = _detect_alt_sel(df, alt_cfg, alt_col, alt_arr)
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
    df, cas_segs = _detect_masked(
        df,
        cas_src_col,
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

    df = _apply_transition_optimisation(
        df,
        alt_segs,
        mach_segs,
        alt_arr,
        cas_src_col,
        config,
    )

    df = _gamma_from_alt(df, alt_segs, alt_cfg, int(config.get("alt_hold_relax", 15)))

    if "bds_mcp_sel_alt_ft" in df.columns:
        df = _backfill_alias(df, "bds_mcp_sel_alt_ft", "fdm_mcp_alt_sel_ft")
    if "bds_fms_sel_alt_ft" in df.columns:
        df = _backfill_alias(df, "bds_fms_sel_alt_ft", "fdm_fms_alt_sel_ft")

    df = _anchored_target(df, "fdm_alt_sel_ft", alt_col, "fdm_alt_target_ft")
    df = _anchored_target(
        df,
        "fdm_cas_sel_kt",
        "bds_ias_kt_clean",
        "fdm_cas_target_kt",
    )
    df = _build_tas_target(df, alt_arr, tas_col)
    return _build_gamma_target(df, tas_col)
