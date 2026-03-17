#!/usr/bin/env python3
"""Validate preprocessing pipeline output (EHS decode + filter + resample).

Checks data structure, temporal regularity, gap-split integrity, and
ADEP/ADES distance completeness after running ``fdm preprocess``.

Run:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/validate_02_preprocessing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> tuple[Path, Path]:
    """Load pipeline config and return (data_dir, preprocess_dir)."""
    sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm-pipeline" / "src"))
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(PROJECT_ROOT / "config.yaml")
    return cfg.paths.data_dir, cfg.paths.resolve("preprocess_dir")


def _status(ok: bool, msg: str) -> bool:
    """Print a status line and return the check result."""
    icon = "✅" if ok else "❌"
    print(f"  {icon} {msg}")
    return ok


def _load_preprocessed(preprocess_dir: Path) -> pl.DataFrame | None:
    """Load all processed parquet files from the preprocess directory."""
    files = sorted(preprocess_dir.glob("processed_*.parquet"))
    if not files:
        _status(False, f"No processed_*.parquet found in {preprocess_dir}")
        return None

    frames = [pl.read_parquet(f) for f in files]
    df = pl.concat(frames)
    _status(True, f"Loaded {len(files)} file(s): {sum(len(f) for f in frames):,} rows total")
    return df


# ---------------------------------------------------------------------------
# T1 — Structure validation
# ---------------------------------------------------------------------------


def validate_structure(df: pl.DataFrame) -> bool:
    """Validate that all required columns are present.

    Args:
        df: Preprocessed dataframe.

    Returns:
        True if all required columns exist.
    """
    required = {
        "flight_id",
        "timestamp",
        "latitude",
        "longitude",
        "altitude",
        "groundspeed",
        "track",
        "typecode",
        "icao24",
        "adep_dist",
        "ades_dist",
    }
    missing = required - set(df.columns)
    ok = _status(not missing, f"Required columns present ({len(required)} expected)")
    if missing:
        _status(False, f"Missing columns: {sorted(missing)}")
    return ok


# ---------------------------------------------------------------------------
# T1b — Temporal ordering (each flight sorted by timestamp)
# ---------------------------------------------------------------------------


def validate_temporal_order(df: pl.DataFrame) -> bool:
    """Check that timestamps are monotonically increasing within each flight.

    Args:
        df: Preprocessed dataframe.

    Returns:
        True if all flights are sorted by timestamp.
    """
    ok = True
    violation_count = 0

    for (fid,), flight in df.group_by("flight_id"):
        ts = flight["timestamp"]
        if not ts.equals(ts.sort()):
            violation_count += 1
            if violation_count <= 5:
                _status(False, f"Flight {fid} is not sorted by timestamp")
            ok = False

    if violation_count > 5:
        _status(False, f"... and {violation_count - 5} more flights unsorted")

    if ok:
        n_flights = df["flight_id"].n_unique()
        _status(True, f"All {n_flights} flights sorted by timestamp")
    return ok


# ---------------------------------------------------------------------------
# T2 — No temporal gaps > 30s (post gap-split)
# ---------------------------------------------------------------------------


def validate_no_gaps(df: pl.DataFrame) -> bool:
    """Check that no flight has a temporal gap exceeding 30 seconds.

    Args:
        df: Preprocessed dataframe with flight_id and timestamp columns.

    Returns:
        True if all flights pass the gap check.
    """
    ok = True
    violation_count = 0

    for (fid,), flight in df.group_by("flight_id"):
        ts = flight.sort("timestamp")["timestamp"]
        dt = ts.diff().drop_nulls().dt.total_seconds().to_numpy()
        max_gap = float(dt.max()) if len(dt) > 0 else 0.0
        if max_gap > 30.0:
            violation_count += 1
            if violation_count <= 5:
                _status(False, f"Gap {max_gap:.1f}s in flight {fid}")
            ok = False

    if violation_count > 5:
        _status(False, f"... and {violation_count - 5} more flights with gaps > 30s")

    if ok:
        n_flights = df["flight_id"].n_unique()
        _status(True, f"No gap > 30s across {n_flights} flights")
    return ok


# ---------------------------------------------------------------------------
# T3 — Sampling rate ≈ 4s
# ---------------------------------------------------------------------------


def validate_sampling_rate(df: pl.DataFrame) -> tuple[bool, np.ndarray]:
    """Check that mean Δt is within [3.5, 5.0] seconds.

    Args:
        df: Preprocessed dataframe.

    Returns:
        Tuple of (check passed, array of all Δt values in seconds).
    """
    deltas: list[float] = []
    for _, flight in df.group_by("flight_id"):
        ts = flight.sort("timestamp")["timestamp"]
        dt = ts.diff().drop_nulls().dt.total_seconds().to_numpy()
        deltas.extend(dt)

    deltas_arr = np.array(deltas)
    mean_dt = float(deltas_arr.mean())
    std_dt = float(deltas_arr.std())
    median_dt = float(np.median(deltas_arr))

    ok = 3.5 < mean_dt < 5.0
    _status(ok, f"Δt mean: {mean_dt:.2f}s, std: {std_dt:.2f}s, median: {median_dt:.2f}s")
    return ok, deltas_arr


# ---------------------------------------------------------------------------
# T4 — ADEP/ADES distances have zero nulls
# ---------------------------------------------------------------------------


def validate_distances(df: pl.DataFrame) -> bool:
    """Check that adep_dist and ades_dist have no nulls per flight.

    Args:
        df: Preprocessed dataframe.

    Returns:
        True if no nulls found.
    """
    ok = True
    for (fid,), flight in df.group_by("flight_id"):
        adep_nulls = flight["adep_dist"].null_count()
        ades_nulls = flight["ades_dist"].null_count()
        if adep_nulls > 0 or ades_nulls > 0:
            _status(
                False, f"Flight {fid}: adep_dist nulls={adep_nulls}, ades_dist nulls={ades_nulls}"
            )
            ok = False

    if ok:
        _status(True, "adep_dist / ades_dist have zero nulls for all flights")
    return ok


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_flight_analysis(df: pl.DataFrame, figure_dir: Path) -> None:
    """Generate a multi-panel flight analysis for a sample flight.

    Produces:
      - Trajectory colored by altitude
      - Trajectory colored by groundspeed
      - Altitude vs time profile
      - Groundspeed vs time profile
      - adep_dist / ades_dist vs time

    Args:
        df: Preprocessed dataframe.
        figure_dir: Directory to save charts.
    """
    import altair as alt

    sample_id = df["flight_id"].unique().sort()[0]
    flight = df.filter(pl.col("flight_id") == sample_id).sort("timestamp")
    flight = flight.with_row_index("t")

    figure_dir.mkdir(parents=True, exist_ok=True)

    base_tooltip = ["flight_id", "timestamp:T", "altitude", "groundspeed", "track"]

    # --- Trajectory colored by altitude ---
    line = (
        alt.Chart(flight)
        .mark_line(strokeWidth=0.5, color="gray")
        .encode(x="longitude:Q", y="latitude:Q", order="timestamp:T")
    )
    pts_alt = (
        alt.Chart(flight)
        .mark_circle(size=20)
        .encode(
            x=alt.X("longitude:Q"),
            y=alt.Y("latitude:Q"),
            color=alt.Color("altitude:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=base_tooltip,
        )
    )
    traj_alt = (line + pts_alt).properties(
        title=f"Trajectory (altitude) — {sample_id}", width=600, height=400
    )

    # --- Trajectory colored by groundspeed ---
    pts_gs = (
        alt.Chart(flight)
        .mark_circle(size=20)
        .encode(
            x=alt.X("longitude:Q"),
            y=alt.Y("latitude:Q"),
            color=alt.Color("groundspeed:Q", scale=alt.Scale(scheme="plasma")),
            tooltip=base_tooltip,
        )
    )
    traj_gs = (line + pts_gs).properties(
        title=f"Trajectory (groundspeed) — {sample_id}", width=600, height=400
    )

    # --- Altitude vs time ---
    alt_profile = (
        alt.Chart(flight)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Sample index"),
            y=alt.Y("altitude:Q", title="Altitude (ft)"),
            tooltip=base_tooltip,
        )
        .properties(title=f"Altitude profile — {sample_id}", width=700, height=250)
    )

    # --- Groundspeed vs time ---
    gs_profile = (
        alt.Chart(flight)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Sample index"),
            y=alt.Y("groundspeed:Q", title="Groundspeed (kt)"),
            tooltip=base_tooltip,
        )
        .properties(title=f"Groundspeed profile — {sample_id}", width=700, height=250)
    )

    # --- ADEP/ADES distances vs time ---
    adep = (
        alt.Chart(flight)
        .mark_line(color="steelblue")
        .encode(
            x=alt.X("t:Q", title="Sample index"),
            y=alt.Y("adep_dist:Q", title="Distance (nm)"),
        )
    )
    ades = alt.Chart(flight).mark_line(color="coral").encode(x="t:Q", y="ades_dist:Q")
    dist_profile = (adep + ades).properties(
        title=f"ADEP (blue) / ADES (coral) distance — {sample_id}",
        width=700,
        height=250,
    )

    # --- Combine all into a single HTML ---
    dashboard = alt.vconcat(
        alt.hconcat(traj_alt, traj_gs),
        alt_profile,
        gs_profile,
        dist_profile,
    ).resolve_scale(color="independent")

    output = figure_dir / "02_flight_analysis.html"
    dashboard.save(output)
    _status(True, f"Flight analysis dashboard saved to {output}")


def plot_dt_distribution(deltas: np.ndarray, figure_dir: Path) -> None:
    """Generate a Δt distribution histogram.

    Args:
        deltas: Array of Δt values in seconds.
        figure_dir: Directory to save the chart.
    """
    import altair as alt

    hist_df = pl.DataFrame({"dt": deltas})

    chart = (
        alt.Chart(hist_df)
        .mark_bar()
        .encode(
            alt.X("dt:Q", bin=alt.Bin(maxbins=50), title="Δt (s)"),
            y="count()",
        )
        .properties(title="Δt distribution (preprocessing output)", width=500)
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "02_dt_distribution.html"
    chart.save(output)
    _status(True, f"Δt distribution chart saved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all preprocessing validation checks."""
    print("=" * 60)
    print("  Preprocessing Validation (02)")
    print("=" * 60)

    data_dir, preprocess_dir = _load_config()

    print("\n--- Loading preprocessed data ---")
    df = _load_preprocessed(preprocess_dir)
    if df is None:
        raise SystemExit(1)

    n_flights = df["flight_id"].n_unique()
    n_points = len(df)

    print("\n--- T1: Structure validation ---")
    t1_ok = validate_structure(df)

    print("\n--- T1b: Temporal ordering ---")
    t1b_ok = validate_temporal_order(df)

    print("\n--- T2: No temporal gaps > 30s ---")
    t2_ok = validate_no_gaps(df)

    print("\n--- T3: Sampling rate ≈ 4s ---")
    t3_ok, deltas = validate_sampling_rate(df)

    print("\n--- T4: ADEP/ADES distance completeness ---")
    t4_ok = validate_distances(df)

    print("\n--- Plots ---")
    figure_dir = data_dir / "figures"
    plot_flight_analysis(df, figure_dir)
    plot_dt_distribution(deltas, figure_dir)

    # Summary
    all_ok = t1_ok and t1b_ok and t2_ok and t3_ok and t4_ok
    print("\n" + "=" * 60)
    icon = "✅" if all_ok else "❌"
    print(f"  {icon} Overall: {'PASS' if all_ok else 'FAIL'}")
    print(f"  📊 {n_flights} flights, {n_points:,} points")
    if n_points > 0:
        # Survival rate: ratio of data retained
        pts_per_flight = n_points / n_flights if n_flights > 0 else 0
        print(f"  📊 ~{pts_per_flight:.0f} points/flight avg")
    print("=" * 60)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
