"""BDS speed cleaning — multi-pass Hampel + ERA-deviation cap + short-gap interp.

Removes spikes and systematic bias from BDS (Mode-S) airspeed signals.
The orchestrator :func:`clean_speeds` operates on 1-D NumPy arrays;
:func:`clean_bds_speeds` is a Polars wrapper that produces
``bds_*_clean`` columns.

.. note::

   ``bds_tas_kt`` is mislabeled upstream — it actually contains TAS in
   **m/s**.  Cleaning runs on the raw m/s values; the unit fix stays in
   :func:`~node_fdm_data.preprocessing.merge.merge_bds_era5`.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["clean_bds_speeds", "clean_speeds"]

_HAMPEL_SCALE: float = 1.4826  # Gaussian consistency factor

# (bds_col, era_col, era_dev_max, use_era_fill)
# tas: era_tas_kt is in kt while bds_tas_kt is m/s — disable ERA cap & fill.
_BDS_SPEC: list[tuple[str, str, float | None, bool]] = [
    ("bds_mach", "era_mach", 0.05, True),
    ("bds_ias_kt", "era_cas_kt", 20.0, True),
    ("bds_tas_kt", "era_tas_kt", None, False),
]


def _hampel_filter(x: np.ndarray, *, window: int, k: float) -> np.ndarray:
    """Single-pass Hampel filter — flag outliers as NaN.

    For each point, compute median and MAD over a centered window of
    half-size *window*.  Flag the point if
    ``|x − median| > k * 1.4826 * MAD``.  When MAD is zero (constant
    window) any value differing from the median is treated as an
    outlier.

    Args:
        x: 1-D NaN-aware array.
        window: half-window size; full window is ``2 * window + 1``.
        k: number of MADs for the outlier threshold.

    Returns:
        Copy of *x* with outliers replaced by NaN.
    """
    n = len(x)
    out = x.copy()
    for i in range(n):
        if np.isnan(x[i]):
            continue
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        w = x[lo:hi]
        w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        if mad == 0.0:
            if x[i] != med:
                out[i] = np.nan
            continue
        if abs(x[i] - med) > k * _HAMPEL_SCALE * mad:
            out[i] = np.nan
    return out


def _interpolate_short_gaps(x: np.ndarray, *, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length ``<= max_gap`` between anchors.

    Runs longer than *max_gap* and edge runs without two anchors are
    left untouched.
    """
    out = x.copy()
    n = len(out)
    isnan = np.isnan(out)
    i = 0
    while i < n:
        if not isnan[i]:
            i += 1
            continue
        j = i
        while j < n and isnan[j]:
            j += 1
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < n and not isnan[i - 1] and not isnan[j]:
            lo, hi = float(out[i - 1]), float(out[j])
            for p in range(i, j):
                out[p] = lo + (hi - lo) * (p - i + 1) / (gap_len + 1)
        i = j
    return out


def clean_speeds(
    values: np.ndarray,
    era: np.ndarray,
    *,
    window: int,
    k: float,
    era_dev_max: float | None,
    n_passes: int,
    interp_max_gap: int,
) -> np.ndarray:
    """Clean a BDS speed signal: multi-pass Hampel + ERA cap + short-gap fill.

    Strategy:

    1. ``n_passes`` of Hampel — handles isolated spikes and clusters
       where a single-pass median is biased by neighbouring outliers.
    2. ERA-deviation cap (when ``era_dev_max`` is set) — catches
       systematic bias that the median misses.
    3. Linear interpolation of NaN runs of length ``<= interp_max_gap``.

    Long NaN gaps are left untouched for the caller (typically an ERA
    fill stage) to handle.

    Args:
        values: 1-D NaN-aware BDS signal.
        era: 1-D NaN-aware ERA signal aligned with *values*.
        window: Hampel half-window size.
        k: Hampel threshold (number of MADs).
        era_dev_max: max allowed ``|cleaned − era|``; ``None`` disables.
        n_passes: number of Hampel passes.
        interp_max_gap: max NaN run length to fill via interpolation.

    Returns:
        Cleaned signal as a new array (input is not mutated).
    """
    cleaned = values.astype(np.float64, copy=True)
    for _ in range(n_passes):
        cleaned = _hampel_filter(cleaned, window=window, k=k)
    if era_dev_max is not None:
        valid = ~np.isnan(cleaned) & ~np.isnan(era)
        dev_outlier = valid & (np.abs(cleaned - era) > era_dev_max)
        cleaned[dev_outlier] = np.nan
    if interp_max_gap > 0:
        cleaned = _interpolate_short_gaps(cleaned, max_gap=interp_max_gap)
    return cleaned


def _fill_with_era(values: np.ndarray, era: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in *values* with *era* where ERA is valid."""
    out = values.copy()
    mask = np.isnan(out) & ~np.isnan(era)
    out[mask] = era[mask]
    return out


def _clean_one_column(
    df: pl.DataFrame,
    bds_col: str,
    era_col: str,
    *,
    era_dev_max: float | None,
    use_era_fill: bool,
    window: int,
    k: float,
    n_passes: int,
    interp_max_gap: int,
) -> np.ndarray | None:
    """Clean one BDS column — returns the cleaned array or ``None`` if absent."""
    if bds_col not in df.columns:
        return None
    values = df[bds_col].cast(pl.Float64).to_numpy()
    era = (
        df[era_col].cast(pl.Float64).to_numpy()
        if era_col in df.columns
        else np.full(len(values), np.nan)
    )
    cleaned = clean_speeds(
        values,
        era,
        window=window,
        k=k,
        era_dev_max=era_dev_max,
        n_passes=n_passes,
        interp_max_gap=interp_max_gap,
    )
    if use_era_fill and era_col in df.columns:
        cleaned = _fill_with_era(cleaned, era)
        cleaned = _hampel_filter(cleaned, window=max(3, window // 2 + 1), k=k)
        cleaned = _interpolate_short_gaps(cleaned, max_gap=interp_max_gap)
    return cleaned


def clean_bds_speeds(
    df: pl.DataFrame,
    *,
    window: int = 7,
    k: float = 3.0,
    era_dev_max_mach: float = 0.05,
    era_dev_max_ias: float = 20.0,
    n_passes: int = 3,
    interp_max_gap: int = 10,
) -> pl.DataFrame:
    """Add ``bds_*_clean`` columns for mach, IAS and TAS to *df*.

    Per-column behaviour:

    * ``bds_mach``  → cap at ``era_dev_max_mach`` against ``era_mach``,
      ERA-fill long gaps.
    * ``bds_ias_kt`` → cap at ``era_dev_max_ias`` against ``era_cas_kt``,
      ERA-fill long gaps.
    * ``bds_tas_kt`` → no ERA cap, no ERA fill (units differ upstream).

    Existing ``bds_*_clean`` columns are dropped before recomputation,
    making the function idempotent.  Missing input columns are silently
    skipped.

    Args:
        df: DataFrame with ``bds_*`` and (optionally) ``era_*`` columns.
        window: Hampel half-window size.
        k: Hampel threshold (number of MADs).
        era_dev_max_mach: ERA-deviation cap for ``bds_mach``.
        era_dev_max_ias: ERA-deviation cap for ``bds_ias_kt``.
        n_passes: number of Hampel passes.
        interp_max_gap: max NaN run length to fill via interpolation.

    Returns:
        New DataFrame with ``bds_*_clean`` columns added.
    """
    existing = [c for c in df.columns if c.endswith("_clean") and c.startswith("bds_")]
    if existing:
        df = df.drop(existing)

    overrides = {
        "bds_mach": era_dev_max_mach,
        "bds_ias_kt": era_dev_max_ias,
    }
    new_columns: list[pl.Series] = []
    for bds_col, era_col, default_era_dev, use_era_fill in _BDS_SPEC:
        era_dev_max = overrides.get(bds_col, default_era_dev)
        cleaned = _clean_one_column(
            df,
            bds_col,
            era_col,
            era_dev_max=era_dev_max,
            use_era_fill=use_era_fill,
            window=window,
            k=k,
            n_passes=n_passes,
            interp_max_gap=interp_max_gap,
        )
        if cleaned is not None:
            new_columns.append(pl.Series(f"{bds_col}_clean", cleaned))

    if not new_columns:
        return df
    return df.with_columns(new_columns)
