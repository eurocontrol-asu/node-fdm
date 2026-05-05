"""BDS speed cleaning — frozen-run + Hampel + V-shape + zigzag-region + ERA fill.

Removes spikes, drift and zigzag artefacts from BDS (Mode-S) airspeed signals
and fills long Mode-S blackouts with cleaned ERA5 reanalysis values.

The orchestrator :func:`clean_speeds` operates on 1-D NumPy arrays;
:func:`clean_bds_speeds` is a Polars wrapper that produces ``bds_*_clean``
columns and an optional ``fdm_tas_from_cas_kt`` derived from cleaned IAS.

.. note::

   ERA fill is disabled for the TAS channel: ERA TAS is derived from
   wind + groundspeed and does not coincide point-wise with the Mode-S
   measured TAS, so filling Mode-S blackouts from ERA would inject a
   different physical quantity into the same column.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from node_fdm_data.physics.speed import cas_to_tas_real

__all__ = ["clean_bds_speeds", "clean_speeds"]

_HAMPEL_SCALE: float = 1.4826  # Gaussian consistency factor
_HAMPEL_MIN_WINDOW: int = 3  # min non-NaN points in window to compute MAD
_ZIGZAG_MIN_LEN: int = 2  # minimum array length to compute deltas
_ZIGZAG_MIN_VALID: int = 4  # min non-NaN deltas in window to estimate density
_KT_TO_MS: float = 0.5144444444444445
_MS_TO_KT: float = 1.0 / _KT_TO_MS
_FT_TO_M: float = 0.3048

# (bds_col, era_col, use_era_fill, frozen_min_run_len)
# tas: ERA fill disabled — ERA TAS (derived from wind+GS) is not the same
# physical quantity as Mode-S measured TAS.
_BDS_SPEC: list[tuple[str, str, bool, int]] = [
    ("bds_mach", "era_mach", True, 20),
    ("bds_ias_kt", "era_cas_kt", True, 20),
    ("bds_tas_kt", "era_tas_kt", False, 6),
]


def _flag_frozen_runs(x: np.ndarray, *, min_run_len: int) -> np.ndarray:
    """Replace runs of strictly identical consecutive non-NaN values with NaN.

    A run is a maximal sequence of indices where ``x[i] == x[i-1]``
    (non-NaN). Runs of length ``>= min_run_len`` are flagged as
    frozen-signal artifacts (sensor stuck on a value while the aircraft
    state evolves). Shorter runs (legitimate quantization plateaus on
    ``IAS HOLD`` / ``MACH HOLD``) are preserved.

    Args:
        x: 1-D NaN-aware array.
        min_run_len: minimum run length to flag as frozen.  Must be ``>= 2``.

    Returns:
        Copy of *x* with frozen runs replaced by NaN.
    """
    out = x.copy()
    n = len(out)
    i = 0
    while i < n:
        if np.isnan(out[i]):
            i += 1
            continue
        j = i + 1
        while j < n and not np.isnan(out[j]) and out[j] == out[i]:
            j += 1
        if j - i >= min_run_len:
            out[i:j] = np.nan
        i = j
    return out


def _hampel_filter(x: np.ndarray, *, window: int, k: float) -> np.ndarray:
    """Single-pass Hampel filter — flag outliers as NaN.

    For each point, compute median and MAD over a centered window of
    half-size *window*.  Flag the point if
    ``|x - median| > k * 1.4826 * MAD``.  When MAD is zero (constant
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
        if len(w) < _HAMPEL_MIN_WINDOW:
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


def _flag_point_jumps(x: np.ndarray, *, max_jump: float) -> np.ndarray:
    """Flag isolated V-shape spikes as NaN.

    A point ``x[i]`` is flagged when both adjacent deltas exceed
    ``max_jump`` in absolute value AND go in opposite directions
    (V-shape, not gradient).  Catches spike-then-recover artefacts that
    Hampel misses when the local gradient inflates the MAD.

    Args:
        x: 1-D NaN-aware array.
        max_jump: minimum absolute delta on each side to flag the point.

    Returns:
        Copy of *x* with V-shape spikes replaced by NaN.
    """
    out = x.copy()
    n = len(out)
    for i in range(1, n - 1):
        if np.isnan(x[i]) or np.isnan(x[i - 1]) or np.isnan(x[i + 1]):
            continue
        d_prev = x[i] - x[i - 1]
        d_next = x[i + 1] - x[i]
        if abs(d_prev) > max_jump and abs(d_next) > max_jump and d_prev * d_next < 0.0:
            out[i] = np.nan
    return out


def _flag_zigzag_region(
    x: np.ndarray, *, half_window: int, jump_min: float, density_min: float
) -> np.ndarray:
    """NaN-out entire regions with a high density of large 1-sample deltas.

    A physically reasonable speed signal has very few large 1-sample
    jumps (transitions are smooth at 4 s sampling).  When the local
    density of ``|delta| > jump_min`` exceeds ``density_min`` over a
    centered window of ``± half_window`` points, the whole window is
    considered corrupt (alternating plateaus, dropouts, BDS frame
    desync) and the centre point is flagged.

    Args:
        x: 1-D NaN-aware array.
        half_window: half-window size for the density estimate.
        jump_min: absolute delta threshold that counts as a "large jump".
        density_min: minimum fraction of large jumps in the window to
            flag the centre point.

    Returns:
        Copy of *x* with detected zigzag regions replaced by NaN.
    """
    out = x.copy()
    n = len(out)
    if n < _ZIGZAG_MIN_LEN:
        return out
    deltas = np.abs(np.diff(x))  # length n-1, NaN-propagating
    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n - 1, i + half_window)
        win = deltas[lo:hi]
        win = win[~np.isnan(win)]
        if win.size < _ZIGZAG_MIN_VALID:
            continue
        if (win > jump_min).mean() >= density_min:
            out[i] = np.nan
    return out


def _on_ground_mask(
    vz_ftmin: np.ndarray,
    alt_ft: np.ndarray,
    *,
    vz_threshold: float,
    alt_threshold: float,
) -> np.ndarray:
    """Boolean mask where the aircraft is considered "on ground".

    A sample is on ground when BOTH ``alt < alt_threshold`` (ft) and
    ``|vz| < vz_threshold`` (ft/min).  NaN inputs are treated as "no
    evidence of being airborne" (NaN → 0).

    Args:
        vz_ftmin: vertical speed in ft/min (NaN allowed).
        alt_ft: pressure altitude in ft (NaN allowed).
        vz_threshold: ft/min threshold below which |vz| counts as still.
        alt_threshold: ft threshold below which alt counts as low.

    Returns:
        Boolean array of shape ``vz_ftmin.shape``.
    """
    abs_vz = np.where(np.isnan(vz_ftmin), 0.0, np.abs(vz_ftmin))
    alt = np.where(np.isnan(alt_ft), 0.0, alt_ft)
    return (alt < alt_threshold) & (abs_vz < vz_threshold)


def _find_next_nan_run(isnan: np.ndarray, start: int) -> tuple[int, int]:
    """Return ``(run_start, run_end_exclusive)`` for the next NaN run from *start*."""
    n = len(isnan)
    i = start
    while i < n and not isnan[i]:
        i += 1
    j = i
    while j < n and isnan[j]:
        j += 1
    return i, j


def _linear_fill(out: np.ndarray, lo_idx: int, hi_idx_exclusive: int) -> None:
    """Fill ``out[lo_idx:hi_idx_exclusive]`` linearly between the two anchors."""
    lo = float(out[lo_idx - 1])
    hi = float(out[hi_idx_exclusive])
    gap_len = hi_idx_exclusive - lo_idx
    for p in range(lo_idx, hi_idx_exclusive):
        out[p] = lo + (hi - lo) * (p - lo_idx + 1) / (gap_len + 1)


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
        i, j = _find_next_nan_run(isnan, i)
        if i >= n:
            break
        gap_len = j - i
        if gap_len <= max_gap and i > 0 and j < n:
            _linear_fill(out, i, j)
        i = j
    return out


def clean_speeds(  # noqa: PLR0913 — config-style orchestrator
    values: np.ndarray,
    *,
    window: int,
    k: float,
    n_passes: int,
    interp_max_gap: int,
    frozen_min_run_len: int | None = None,
    point_jump_max: float | None = None,
    zigzag_jump_min: float | None = None,
    zigzag_half_window: int = 15,
    zigzag_density_min: float = 0.25,
) -> np.ndarray:
    """Clean a 1-D BDS speed signal.

    Strategy:

    0. Frozen-run filter (when ``frozen_min_run_len`` is set).
    1. ``n_passes`` of Hampel — isolated spikes & small clusters.
    2. ``n_passes`` of V-shape filter (when ``point_jump_max`` is set)
       — catches isolated spike-then-recover points where Hampel's MAD
       was inflated by the local gradient.
    3. Zigzag-region filter (when ``zigzag_jump_min`` is set) — NaN-out
       entire windows where the density of large 1-sample deltas exceeds
       ``zigzag_density_min``.
    4. Linear interpolation of NaN runs of length ``<= interp_max_gap``.

    Long NaN gaps are left untouched for the caller (typically an ERA
    fill stage) to handle.

    Args:
        values: 1-D NaN-aware BDS signal.
        window: Hampel half-window size.
        k: Hampel threshold (number of MADs).
        n_passes: number of Hampel (and V-shape) passes.
        interp_max_gap: max NaN run length to fill via interpolation.
        frozen_min_run_len: min length of identical-consecutive-values run
            to flag as frozen-signal artifact; ``None`` disables.
        point_jump_max: V-shape filter threshold; ``None`` disables.
        zigzag_jump_min: large-jump threshold for the region detector;
            ``None`` disables.
        zigzag_half_window: half-window size for the region detector.
        zigzag_density_min: minimum density of large jumps to flag a region.

    Returns:
        Cleaned signal as a new array (input is not mutated).
    """
    raw = values.astype(np.float64, copy=True)
    cleaned = raw.copy()
    if frozen_min_run_len is not None:
        cleaned = _flag_frozen_runs(cleaned, min_run_len=frozen_min_run_len)
    for _ in range(n_passes):
        cleaned = _hampel_filter(cleaned, window=window, k=k)
    if point_jump_max is not None:
        for _ in range(n_passes):
            cleaned = _flag_point_jumps(cleaned, max_jump=point_jump_max)
    if zigzag_jump_min is not None:
        # Run on the RAW signal so NaNs introduced by Hampel don't
        # break delta computation.
        zigzag = _flag_zigzag_region(
            raw,
            half_window=zigzag_half_window,
            jump_min=zigzag_jump_min,
            density_min=zigzag_density_min,
        )
        zigzag_mask = np.isnan(zigzag) & ~np.isnan(raw)
        cleaned[zigzag_mask] = np.nan
    if interp_max_gap > 0:
        cleaned = _interpolate_short_gaps(cleaned, max_gap=interp_max_gap)
    return cleaned


def _clean_era(  # noqa: PLR0913 — config-style helper
    era: np.ndarray,
    *,
    window: int,
    k: float,
    n_passes: int,
    zigzag_jump_min: float | None,
    zigzag_half_window: int,
    zigzag_density_min: float,
) -> np.ndarray:
    """Clean an ERA5 speed signal: multi-pass Hampel + zigzag-region.

    ERA is normally extremely smooth in time, so any region with multiple
    large 1-sample jumps is a grid artefact (missing tile, interpolation
    bug, etc.) and is NaN-ed out as a block, not point-by-point.
    """
    cleaned = era.astype(np.float64, copy=True)
    for _ in range(n_passes):
        cleaned = _hampel_filter(cleaned, window=window, k=k)
    if zigzag_jump_min is not None:
        zigzag = _flag_zigzag_region(
            era,
            half_window=zigzag_half_window,
            jump_min=zigzag_jump_min,
            density_min=zigzag_density_min,
        )
        zigzag_mask = np.isnan(zigzag) & ~np.isnan(era)
        cleaned[zigzag_mask] = np.nan
    return cleaned


def _fill_with_era(values: np.ndarray, era: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in *values* with *era* where ERA is valid."""
    out = values.copy()
    mask = np.isnan(out) & ~np.isnan(era)
    out[mask] = era[mask]
    return out


def _clean_one_column(  # noqa: PLR0913 — config-style helper
    df: pl.DataFrame,
    bds_col: str,
    era_col: str,
    *,
    use_era_fill: bool,
    bds_window: int,
    era_window: int,
    k: float,
    n_passes: int,
    interp_max_gap: int,
    frozen_min_run_len: int | None,
    point_jump_max: float | None,
    zigzag_jump_min: float | None,
    zigzag_half_window: int,
    zigzag_density_min_bds: float,
    zigzag_density_min_era: float,
) -> np.ndarray | None:
    """Clean one BDS column and (optionally) fill from a cleaned ERA column.

    Returns the cleaned 1-D array (kt or m/s, same units as ``bds_col``)
    or ``None`` when the BDS column is missing.
    """
    if bds_col not in df.columns:
        return None
    values = df[bds_col].cast(pl.Float64).to_numpy()
    cleaned = clean_speeds(
        values,
        window=bds_window,
        k=k,
        n_passes=n_passes,
        interp_max_gap=interp_max_gap,
        frozen_min_run_len=frozen_min_run_len,
        point_jump_max=point_jump_max,
        zigzag_jump_min=zigzag_jump_min,
        zigzag_half_window=zigzag_half_window,
        zigzag_density_min=zigzag_density_min_bds,
    )
    if use_era_fill and era_col in df.columns:
        era_raw = df[era_col].cast(pl.Float64).to_numpy()
        era_clean = _clean_era(
            era_raw,
            window=era_window,
            k=k,
            n_passes=n_passes,
            zigzag_jump_min=zigzag_jump_min,
            zigzag_half_window=zigzag_half_window,
            zigzag_density_min=zigzag_density_min_era,
        )
        cleaned = _fill_with_era(cleaned, era_clean)
        cleaned = _hampel_filter(cleaned, window=max(3, bds_window // 2 + 1), k=k)
        cleaned = _interpolate_short_gaps(cleaned, max_gap=interp_max_gap)
    return cleaned


def _build_on_ground_mask(
    df: pl.DataFrame, *, vz_threshold: float, alt_threshold: float
) -> np.ndarray | None:
    """Build the on-ground boolean mask if the required columns are present."""
    if "raw_vz_ftmin" not in df.columns or "raw_alt_ft" not in df.columns:
        return None
    vz = df["raw_vz_ftmin"].cast(pl.Float64).to_numpy()
    alt = df["raw_alt_ft"].cast(pl.Float64).to_numpy()
    return _on_ground_mask(vz, alt, vz_threshold=vz_threshold, alt_threshold=alt_threshold)


def _derive_tas_from_cas(df: pl.DataFrame, ias_clean_kt: np.ndarray) -> np.ndarray | None:
    """Derive TAS (kt) from cleaned IAS via :func:`cas_to_tas_real`.

    Requires ``raw_alt_ft`` and ``era_temp_K`` on *df*; returns ``None``
    otherwise.  IAS is treated as CAS for this conversion (Mode-S BDS50
    reports CAS in the ``IAS`` field at our sampling rate).
    """
    if "raw_alt_ft" not in df.columns or "era_temp_K" not in df.columns:
        return None
    alt_m = df["raw_alt_ft"].cast(pl.Float64).to_numpy() * _FT_TO_M
    temp_k = df["era_temp_K"].cast(pl.Float64).to_numpy()
    cas_ms = ias_clean_kt * _KT_TO_MS
    tas_ms = np.asarray(cas_to_tas_real(cas_ms, alt_m, temp_k), dtype=np.float64)
    return tas_ms * _MS_TO_KT


def _drop_existing_clean_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Drop ``bds_*_clean`` and ``fdm_tas_from_cas_kt`` from *df* if present."""
    existing = [
        c
        for c in df.columns
        if (c.endswith("_clean") and c.startswith("bds_")) or c == "fdm_tas_from_cas_kt"
    ]
    return df.drop(existing) if existing else df


def _apply_on_ground(arr: np.ndarray, on_ground: np.ndarray | None) -> np.ndarray:
    """Return a copy of *arr* with on-ground samples set to NaN, when applicable."""
    if on_ground is None:
        return arr
    out = arr.copy()
    out[on_ground] = np.nan
    return out


def _build_clean_columns(  # noqa: PLR0913 — config-style helper
    df: pl.DataFrame,
    *,
    on_ground: np.ndarray | None,
    bds_window: int,
    era_window: int,
    k: float,
    n_passes: int,
    interp_max_gap: int,
    frozen_min_run_len_mach: int | None,
    frozen_min_run_len_ias: int | None,
    frozen_min_run_len_tas: int | None,
    point_jump_max_mach: float | None,
    point_jump_max_kt: float | None,
    zigzag_jump_min_mach: float | None,
    zigzag_jump_min_kt: float | None,
    zigzag_half_window: int,
    zigzag_density_min_bds: float,
    zigzag_density_min_era: float,
) -> tuple[list[pl.Series], np.ndarray | None]:
    """Iterate over :data:`_BDS_SPEC` and build the cleaned series list."""
    frozen_overrides: dict[str, int | None] = {
        "bds_mach": frozen_min_run_len_mach,
        "bds_ias_kt": frozen_min_run_len_ias,
        "bds_tas_kt": frozen_min_run_len_tas,
    }
    point_jump_overrides: dict[str, float | None] = {
        "bds_mach": point_jump_max_mach,
        "bds_ias_kt": point_jump_max_kt,
        "bds_tas_kt": point_jump_max_kt,
    }
    zigzag_jump_overrides: dict[str, float | None] = {
        "bds_mach": zigzag_jump_min_mach,
        "bds_ias_kt": zigzag_jump_min_kt,
        "bds_tas_kt": zigzag_jump_min_kt,
    }

    new_columns: list[pl.Series] = []
    ias_clean: np.ndarray | None = None
    for bds_col, era_col, use_era_fill, default_frozen in _BDS_SPEC:
        cleaned = _clean_one_column(
            df,
            bds_col,
            era_col,
            use_era_fill=use_era_fill,
            bds_window=bds_window,
            era_window=era_window,
            k=k,
            n_passes=n_passes,
            interp_max_gap=interp_max_gap,
            frozen_min_run_len=frozen_overrides.get(bds_col, default_frozen),
            point_jump_max=point_jump_overrides.get(bds_col),
            zigzag_jump_min=zigzag_jump_overrides.get(bds_col),
            zigzag_half_window=zigzag_half_window,
            zigzag_density_min_bds=zigzag_density_min_bds,
            zigzag_density_min_era=zigzag_density_min_era,
        )
        if cleaned is None:
            continue
        cleaned = _apply_on_ground(cleaned, on_ground)
        new_columns.append(pl.Series(f"{bds_col}_clean", cleaned))
        if bds_col == "bds_ias_kt":
            ias_clean = cleaned
    return new_columns, ias_clean


def _build_tas_from_cas_series(
    df: pl.DataFrame,
    ias_clean: np.ndarray | None,
    *,
    on_ground: np.ndarray | None,
) -> pl.Series | None:
    """Build the ``fdm_tas_from_cas_kt`` series, or ``None`` if not derivable."""
    if ias_clean is None:
        return None
    tas_from_cas = _derive_tas_from_cas(df, ias_clean)
    if tas_from_cas is None:
        return None
    tas_from_cas = _apply_on_ground(tas_from_cas, on_ground)
    return pl.Series("fdm_tas_from_cas_kt", tas_from_cas)


def clean_bds_speeds(  # noqa: PLR0913 — config-style entry point
    df: pl.DataFrame,
    *,
    bds_window: int = 50,
    era_window: int = 15,
    k: float = 3.0,
    n_passes: int = 3,
    interp_max_gap: int = 10,
    frozen_min_run_len_mach: int | None = 20,
    frozen_min_run_len_ias: int | None = 20,
    frozen_min_run_len_tas: int | None = 6,
    point_jump_max_mach: float | None = 0.05,
    point_jump_max_kt: float | None = 20.0,
    zigzag_jump_min_mach: float | None = 0.05,
    zigzag_jump_min_kt: float | None = 20.0,
    zigzag_half_window: int = 15,
    zigzag_density_min_bds: float = 0.25,
    zigzag_density_min_era: float = 0.15,
    on_ground_vz_threshold: float = 200.0,
    on_ground_alt_threshold: float = 1500.0,
) -> pl.DataFrame:
    """Add ``bds_*_clean`` and ``fdm_tas_from_cas_kt`` columns to *df*.

    Per-channel behaviour:

    * ``bds_mach``  → frozen-run + Hampel + V-shape + zigzag region;
      ERA fill from cleaned ``era_mach``; on-ground mask applied.
    * ``bds_ias_kt`` → same pipeline; ERA fill from cleaned ``era_cas_kt``.
    * ``bds_tas_kt`` → same point-wise filters; **no ERA fill** (units
      differ upstream); on-ground mask still applied.

    When ``raw_alt_ft`` and ``era_temp_K`` are present, an extra column
    ``fdm_tas_from_cas_kt`` is appended, derived from ``bds_ias_kt_clean``
    via :func:`~node_fdm_data.physics.speed.cas_to_tas_real`.

    Existing ``bds_*_clean`` and ``fdm_tas_from_cas_kt`` columns are
    dropped before recomputation, making the function idempotent.  Missing
    input columns are silently skipped.

    Args:
        df: DataFrame with ``bds_*`` and (optionally) ``era_*`` /
            ``raw_vz_ftmin`` / ``raw_alt_ft`` / ``era_temp_K`` columns.
        bds_window: Hampel half-window size for BDS signals.
        era_window: Hampel half-window size for ERA signals.
        k: Hampel threshold (number of MADs).
        n_passes: number of Hampel (and V-shape) passes.
        interp_max_gap: max NaN run length to fill via interpolation.
        frozen_min_run_len_mach: frozen-run threshold for ``bds_mach``;
            ``None`` disables.
        frozen_min_run_len_ias: frozen-run threshold for ``bds_ias_kt``;
            ``None`` disables.
        frozen_min_run_len_tas: frozen-run threshold for ``bds_tas_kt``;
            ``None`` disables.
        point_jump_max_mach: V-shape threshold for the Mach channel.
        point_jump_max_kt: V-shape threshold for IAS / TAS channels (kt).
        zigzag_jump_min_mach: zigzag-region jump threshold for Mach.
        zigzag_jump_min_kt: zigzag-region jump threshold for IAS / TAS.
        zigzag_half_window: zigzag-region half-window (shared BDS+ERA).
        zigzag_density_min_bds: zigzag-region density threshold for BDS.
        zigzag_density_min_era: zigzag-region density threshold for ERA.
        on_ground_vz_threshold: ``|vz|`` threshold (ft/min) for the
            on-ground mask.
        on_ground_alt_threshold: altitude threshold (ft) for the
            on-ground mask.

    Returns:
        New DataFrame with cleaned columns appended.
    """
    df = _drop_existing_clean_cols(df)

    on_ground = _build_on_ground_mask(
        df,
        vz_threshold=on_ground_vz_threshold,
        alt_threshold=on_ground_alt_threshold,
    )

    new_columns, ias_clean = _build_clean_columns(
        df,
        on_ground=on_ground,
        bds_window=bds_window,
        era_window=era_window,
        k=k,
        n_passes=n_passes,
        interp_max_gap=interp_max_gap,
        frozen_min_run_len_mach=frozen_min_run_len_mach,
        frozen_min_run_len_ias=frozen_min_run_len_ias,
        frozen_min_run_len_tas=frozen_min_run_len_tas,
        point_jump_max_mach=point_jump_max_mach,
        point_jump_max_kt=point_jump_max_kt,
        zigzag_jump_min_mach=zigzag_jump_min_mach,
        zigzag_jump_min_kt=zigzag_jump_min_kt,
        zigzag_half_window=zigzag_half_window,
        zigzag_density_min_bds=zigzag_density_min_bds,
        zigzag_density_min_era=zigzag_density_min_era,
    )

    tas_series = _build_tas_from_cas_series(df, ias_clean, on_ground=on_ground)
    if tas_series is not None:
        new_columns.append(tas_series)

    if not new_columns:
        return df
    return df.with_columns(new_columns)
