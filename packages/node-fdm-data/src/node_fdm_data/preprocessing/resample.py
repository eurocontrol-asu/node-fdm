"""Gap-aware resampling and interpolation — pipeline v3, étape 1.5.

Resamples each flight to a regular time grid (default 4 s), detecting
sub-segments per column group and interpolating only within them.
Gaps longer than ``max_gap_s`` are left null with explicit boolean
flags ``pre_gap_position``, ``pre_gap_altitude``, ``pre_gap_bds``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import timedelta

import polars as pl

__all__ = [
    "ALTITUDE_COLS",
    "BDS_COLS",
    "COLUMN_GROUPS",
    "POSITION_COLS",
    "detect_subsegments",
    "interpolate_group_by_subsegments",
    "preprocess_flights",
    "resample_flight",
    "smooth_position_subsegments",
]

POSITION_COLS: list[str] = ["raw_lat_deg", "raw_lon_deg"]
ALTITUDE_COLS: list[str] = ["raw_alt_ft", "raw_gs_kt", "raw_track_deg", "raw_vz_ftmin"]
BDS_COLS: list[str] = [
    "bds_mach",
    "bds_tas_kt",
    "bds_ias_kt",
    "bds_hdg_deg",
    "bds_mcp_sel_alt_ft",
    "bds_fms_sel_alt_ft",
]

# (columns_to_interpolate, reference_columns_for_subsegment_detection)
COLUMN_GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "position": (POSITION_COLS, POSITION_COLS),
    "altitude": (ALTITUDE_COLS, ["raw_alt_ft"]),
    "bds": (BDS_COLS, ["bds_mach"]),
}

_CARRY_OVER: set[str] = {"raw_icao24", "raw_callsign"}


def _assign_seg_ids(
    indices: list[int],
    diffs: list[float | None],
    max_gap_s: float,
    n: int,
) -> list[int]:
    seg_ids: list[int] = [-1] * n
    current_seg = 0
    for pos, (idx, diff) in enumerate(zip(indices, diffs, strict=False)):
        if pos > 0 and diff is not None and diff > max_gap_s:
            current_seg += 1
        seg_ids[idx] = current_seg
    return seg_ids


def _drop_singletons(seg_ids: list[int]) -> list[int]:
    counts = Counter(s for s in seg_ids if s >= 0)
    return [s if s < 0 or counts[s] >= 2 else -1 for s in seg_ids]  # noqa: PLR2004


def detect_subsegments(
    df: pl.DataFrame,
    ref_cols: list[str],
    max_gap_s: float,
) -> pl.Series:
    """Detect contiguous sub-segments from non-null runs of reference columns.

    A sub-segment boundary is placed wherever the time gap between
    consecutive non-null rows exceeds *max_gap_s*.  Single-point
    segments are excluded (set to -1).

    Args:
        df: Single-flight DataFrame, sorted by ``raw_timestamp``.
        ref_cols: Columns to check — a row "has data" iff ALL are non-null.
        max_gap_s: Maximum allowed gap (seconds) before splitting.

    Returns:
        Int32 Series (same length as *df*).  Rows with data are assigned
        a sub-segment ID starting at 0; rows without data (or in
        single-point segments) get -1.
    """
    n = len(df)
    present = [c for c in ref_cols if c in df.columns]
    if not present:
        return pl.Series("seg_id", [-1] * n, dtype=pl.Int32)

    has_data = df.select(
        pl.all_horizontal(pl.col(c).is_not_null() for c in present),
    ).to_series()
    if not has_data.any():
        return pl.Series("seg_id", [-1] * n, dtype=pl.Int32)

    indices = has_data.arg_true()
    diffs_s = df["raw_timestamp"].gather(indices).diff().dt.total_seconds()

    seg_ids = _assign_seg_ids(indices.to_list(), diffs_s.to_list(), max_gap_s, n)
    return pl.Series("seg_id", _drop_singletons(seg_ids), dtype=pl.Int32)


@dataclass
class _GridAccumulator:
    columns: list[str]
    ts_to_idx: dict[object, int]
    result: dict[str, list[object]]
    covered: list[bool]

    def write(self, row: dict[str, object]) -> None:
        idx = self.ts_to_idx.get(row["raw_timestamp"])
        if idx is None:
            return
        self.covered[idx] = True
        for c in self.columns:
            self.result[c][idx] = row[c]


def _interpolate_segment(
    seg_data: pl.DataFrame,
    grid_ts: pl.Series,
    columns: list[str],
) -> pl.DataFrame | None:
    seg_start = seg_data["raw_timestamp"].min()
    seg_end = seg_data["raw_timestamp"].max()
    seg_grid_ts = grid_ts.filter((grid_ts >= seg_start) & (grid_ts <= seg_end))
    if len(seg_grid_ts) == 0:
        return None

    grid_only = pl.DataFrame({"raw_timestamp": seg_grid_ts})
    seg_tagged = seg_data.with_columns(pl.lit(0).alias("_ord"))
    grid_tagged = grid_only.with_columns(pl.lit(1).alias("_ord"))
    combined = (
        pl.concat([seg_tagged, grid_tagged], how="diagonal_relaxed")
        .sort("raw_timestamp", "_ord")
        .unique("raw_timestamp", keep="first")
        .drop("_ord")
        .with_columns(pl.col(c).interpolate() for c in columns)
    )
    return combined.join(grid_only, on="raw_timestamp", how="inner")


def _process_segment(
    seg_data: pl.DataFrame,
    grid_ts: pl.Series,
    acc: _GridAccumulator,
) -> None:
    if len(seg_data) == 0:
        return
    if len(seg_data) == 1:
        row: dict[str, object] = {"raw_timestamp": seg_data["raw_timestamp"][0]}
        for c in acc.columns:
            row[c] = seg_data[c][0]
        acc.write(row)
        return

    interpolated = _interpolate_segment(seg_data, grid_ts, acc.columns)
    if interpolated is None:
        return
    for row in interpolated.iter_rows(named=True):
        acc.write(row)


def interpolate_group_by_subsegments(
    original: pl.DataFrame,
    grid_ts: pl.Series,
    columns: list[str],
    seg_ids: pl.Series,
) -> tuple[dict[str, list[object]], pl.Series]:
    """Interpolate columns within sub-segments onto a regular time grid.

    Points on the grid that fall outside any sub-segment remain null.

    Args:
        original: Original single-flight DataFrame (irregular timestamps).
        grid_ts: Regular-grid timestamp Series for the flight.
        columns: Columns to interpolate.
        seg_ids: Sub-segment IDs from :func:`detect_subsegments`.

    Returns:
        Tuple of (*column_values*, *gap_flag*).
        *column_values* maps column name to a list of values aligned
        to *grid_ts*.  *gap_flag* is ``True`` where the grid point
        falls between sub-segments.
    """
    n_grid = len(grid_ts)
    result: dict[str, list[object]] = {c: [None] * n_grid for c in columns}
    covered = [False] * n_grid

    unique_segs = sorted(set(seg_ids.to_list()) - {-1})
    if not unique_segs:
        return result, pl.Series("gap", [True] * n_grid, dtype=pl.Boolean)

    acc = _GridAccumulator(
        columns=columns,
        ts_to_idx={ts: i for i, ts in enumerate(grid_ts.to_list())},
        result=result,
        covered=covered,
    )

    for seg_id in unique_segs:
        seg_data = original.filter(seg_ids == seg_id).select("raw_timestamp", *columns)
        _process_segment(seg_data, grid_ts, acc)

    gap_flag = pl.Series("gap", [not c for c in covered], dtype=pl.Boolean)
    return result, gap_flag


def smooth_position_subsegments(df: pl.DataFrame) -> pl.DataFrame:
    """Apply Savitzky-Golay smoothing on contiguous position sub-segments.

    Uses ``traffic.Flight.filter("aggressive")`` on each run of non-null
    lat/lon in the resampled grid.

    Args:
        df: Resampled flight DataFrame on regular grid.

    Returns:
        DataFrame with smoothed ``raw_lat_deg`` / ``raw_lon_deg``.
    """
    if "raw_lat_deg" not in df.columns or "raw_lon_deg" not in df.columns:
        return df

    has_pos = df.select(
        pl.col("raw_lat_deg").is_not_null() & pl.col("raw_lon_deg").is_not_null(),
    ).to_series()

    if not has_pos.any():
        return df

    try:
        from traffic.core import Flight  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return df

    # Detect runs of non-null position
    changes = has_pos != has_pos.shift()
    run_ids = changes.cum_sum()

    lats = df["raw_lat_deg"].to_list()
    lons = df["raw_lon_deg"].to_list()

    for run_id in run_ids.filter(has_pos).unique().drop_nulls().sort().to_list():
        mask = (run_ids == run_id) & has_pos
        indices = mask.arg_true().to_list()

        if len(indices) < 10:  # noqa: PLR2004
            continue

        select_cols = [
            pl.col("raw_timestamp").alias("timestamp"),
            pl.col("raw_lat_deg").alias("latitude"),
            pl.col("raw_lon_deg").alias("longitude"),
        ]
        if "raw_alt_ft" in df.columns:
            select_cols.append(pl.col("raw_alt_ft").alias("altitude"))

        segment = df.filter(mask).select(select_cols)
        flight = Flight(segment.to_pandas())
        smoothed = flight.filter("aggressive")

        if smoothed is None or len(smoothed) != len(indices):
            continue

        smooth_pd = smoothed.data[["latitude", "longitude"]]
        for i, idx in enumerate(indices):
            lats[idx] = float(smooth_pd.iloc[i]["latitude"])
            lons[idx] = float(smooth_pd.iloc[i]["longitude"])

    return df.with_columns(
        pl.Series("raw_lat_deg", lats, dtype=pl.Float64),
        pl.Series("raw_lon_deg", lons, dtype=pl.Float64),
    )


def resample_flight(
    df: pl.DataFrame,
    rate_s: int = 4,
    max_gap_s: float = 30.0,
    smooth: bool = True,
) -> pl.DataFrame:
    """Resample a single flight to a regular grid with gap-aware interpolation.

    1. Create a regular time grid at *rate_s* seconds.
    2. For each column group (position, altitude, BDS): detect sub-segments,
       interpolate within, leave null between.
    3. Optionally smooth position sub-segments with Savitzky-Golay.
    4. Add gap flags: ``pre_gap_position``, ``pre_gap_altitude``, ``pre_gap_bds``.

    Args:
        df: Single-flight DataFrame, sorted by ``raw_timestamp``.
        rate_s: Grid interval in seconds.
        max_gap_s: Maximum gap for interpolation continuity.
        smooth: Apply Savitzky-Golay on position sub-segments.

    Returns:
        Resampled DataFrame on a regular *rate_s* grid.
    """
    ts_min = df["raw_timestamp"].min()
    ts_max = df["raw_timestamp"].max()
    assert ts_min is not None and ts_max is not None

    grid_ts: pl.Series = pl.datetime_range(
        ts_min,  # type: ignore[arg-type]
        ts_max,  # type: ignore[arg-type]
        interval=timedelta(seconds=rate_s),
        eager=True,
    ).rename("raw_timestamp")  # type: ignore[union-attr]

    result = pl.DataFrame({"raw_timestamp": grid_ts})

    # Carry over constant meta columns and per-flight string identifiers
    carry_cols = [c for c in df.columns if c.startswith("meta_") or c in _CARRY_OVER]
    if carry_cols:
        result = result.with_columns(pl.lit(df[c][0]).alias(c) for c in carry_cols)

    # Process each column group
    for group_name, (interp_cols, ref_cols) in COLUMN_GROUPS.items():
        present_cols = [c for c in interp_cols if c in df.columns]
        if not present_cols:
            result = result.with_columns(pl.lit(True).alias(f"pre_gap_{group_name}"))
            continue

        seg_ids = detect_subsegments(df, ref_cols, max_gap_s)
        col_values, gap_flag = interpolate_group_by_subsegments(
            df,
            grid_ts,
            present_cols,
            seg_ids,
        )

        result = result.with_columns(
            *[pl.Series(c, col_values[c]) for c in present_cols],
            gap_flag.alias(f"pre_gap_{group_name}"),
        )

    # Smooth position sub-segments
    if smooth:
        result = smooth_position_subsegments(result)

    return result


def preprocess_flights(
    df: pl.DataFrame,
    *,
    rate_s: int = 4,
    max_gap_s: float = 30.0,
    min_duration_s: int = 240,
    smooth: bool = True,
) -> pl.DataFrame:
    """Resample all flights with gap-aware interpolation.

    Applies :func:`resample_flight` to each ``meta_flight_id`` group,
    dropping flights shorter than *min_duration_s*.

    Args:
        df: DataFrame with all flights (from Delta Table).
        rate_s: Grid interval in seconds.
        max_gap_s: Maximum gap for sub-segment continuity.
        min_duration_s: Minimum flight duration to keep.
        smooth: Apply Savitzky-Golay on position sub-segments.

    Returns:
        Resampled DataFrame with regular *rate_s* grid per flight.
    """
    flights = df.partition_by("meta_flight_id", maintain_order=True)
    results: list[pl.DataFrame] = []

    for flight_df in flights:
        flight_df = flight_df.sort("raw_timestamp")
        ts = flight_df["raw_timestamp"]
        ts_max_val = ts.max()
        ts_min_val = ts.min()
        if ts_max_val is None or ts_min_val is None:
            continue
        delta = ts_max_val - ts_min_val  # type: ignore[operator]
        duration = delta.total_seconds()  # type: ignore[union-attr]

        if duration < min_duration_s:
            continue

        resampled = resample_flight(
            flight_df,
            rate_s=rate_s,
            max_gap_s=max_gap_s,
            smooth=smooth,
        )
        results.append(resampled)

    if not results:
        return df.clear()

    return pl.concat(results, how="diagonal_relaxed")
