#!/usr/bin/env python3
"""Validate processing pipeline output (ERA5 + features + split).

Checks feature completeness, physical consistency, distance monotonicity,
dataset split ratios, and per-flight file counts after running ``fdm process``.

Run:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/validate_03_processing.py
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
    """Load pipeline config and return (data_dir, process_dir)."""
    sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm-pipeline" / "src"))
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(PROJECT_ROOT / "config.yaml")
    return cfg.paths.data_dir, cfg.paths.resolve("process_dir")


def _status(ok: bool, msg: str) -> bool:
    """Print a status line and return the check result."""
    icon = "✅" if ok else "❌"
    print(f"  {icon} {msg}")
    return ok


def _load_processed(process_dir: Path) -> pl.DataFrame | None:
    """Load all processed parquet files from the process directory."""
    files = sorted(process_dir.glob("processed_*.parquet"))
    if not files:
        _status(False, f"No processed_*.parquet found in {process_dir}")
        return None

    frames = [pl.read_parquet(f) for f in files]
    df = pl.concat(frames, how="diagonal_relaxed")
    _status(True, f"Loaded {len(files)} file(s): {sum(len(f) for f in frames):,} rows total")
    return df


def _resolve_col(df: pl.DataFrame, *candidates: str) -> str | None:
    """Return the first column name present in *df*, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# T1 — Feature set validation (AC1)
# ---------------------------------------------------------------------------


def validate_feature_set(df: pl.DataFrame) -> bool:
    """Validate that all derived columns are present.

    Checks weather (ERA5), aero (TAS/Mach/CAS with fallback names),
    physics-derived, selected params, lateral dynamics, and distance columns.

    Args:
        df: Processed dataframe.

    Returns:
        True if all required column groups are present.
    """
    ok = True

    # Weather (ERA5)
    weather = {"temperature", "u_component_of_wind", "v_component_of_wind"}
    missing_w = weather - set(df.columns)
    ok = _status(not missing_w, f"Weather columns ({len(weather)} expected)") and ok
    if missing_w:
        _status(False, f"  Missing weather: {sorted(missing_w)}")

    # Aero — accept either raw or SI column names
    aero_checks = [
        (("TAS", "tas_ms"), "TAS/tas_ms"),
        (("Mach", "mach"), "Mach/mach"),
        (("CAS", "cas_ms"), "CAS/cas_ms"),
    ]
    aero_ok = True
    for candidates, label in aero_checks:
        if not any(c in df.columns for c in candidates):
            _status(False, f"  Missing aero column: {label}")
            aero_ok = False
    ok = _status(aero_ok, "Aero columns (TAS, Mach, CAS)") and ok

    # Physics-derived
    physics = {"gamma_air", "long_wind"}
    missing_p = physics - set(df.columns)
    ok = _status(not missing_p, f"Physics columns ({len(physics)} expected)") and ok
    if missing_p:
        _status(False, f"  Missing physics: {sorted(missing_p)}")

    # Selected params
    sel_params = {"mach_sel", "cas_sel", "vz_sel"}
    missing_s = sel_params - set(df.columns)
    ok = _status(not missing_s, f"Selected param columns ({len(sel_params)} expected)") and ok
    if missing_s:
        _status(False, f"  Missing selected params: {sorted(missing_s)}")

    # Lateral dynamics
    lateral = {"track_ortho", "drift_angle"}
    missing_l = lateral - set(df.columns)
    ok = _status(not missing_l, f"Lateral columns ({len(lateral)} expected)") and ok
    if missing_l:
        _status(False, f"  Missing lateral: {sorted(missing_l)}")

    # Distance
    has_dist = "distance_along_track_m" in df.columns
    ok = _status(has_dist, "Distance column (distance_along_track_m)") and ok

    return ok


# ---------------------------------------------------------------------------
# T2 — Physical consistency (AC2)
# ---------------------------------------------------------------------------


def validate_physics(df: pl.DataFrame) -> bool:
    """Check physical bounds on key variables.

    - Mach in [0.05, 1.05]
    - gamma_air in [-0.3, 0.3] rad
    - TAS > 0

    Args:
        df: Processed dataframe.

    Returns:
        True if all physical bounds hold.
    """
    ok = True

    # Mach
    mach_col = _resolve_col(df, "Mach", "mach")
    if mach_col:
        mach = df[mach_col].drop_nulls().to_numpy()
        in_range = bool(np.all((mach > 0.05) & (mach < 1.05)))
        ok = (
            _status(
                in_range,
                f"Mach in [0.05, 1.05] -- actual [{mach.min():.4f}, {mach.max():.4f}]",
            )
            and ok
        )
    else:
        ok = _status(False, "Mach column not found") and ok

    # gamma_air (flight path angle in radians)
    if "gamma_air" in df.columns:
        gamma = df["gamma_air"].drop_nulls().to_numpy()
        in_range = bool(np.all(np.abs(gamma) < 0.3))
        ok = (
            _status(
                in_range,
                f"gamma_air in [-0.3, 0.3] -- actual [{gamma.min():.4f}, {gamma.max():.4f}]",
            )
            and ok
        )
    else:
        ok = _status(False, "gamma_air column not found") and ok

    # TAS > 0
    tas_col = _resolve_col(df, "TAS", "tas_ms")
    if tas_col:
        tas = df[tas_col].drop_nulls().to_numpy()
        all_positive = bool(np.all(tas > 0))
        ok = (
            _status(
                all_positive,
                f"TAS > 0 -- min={tas.min():.2f}",
            )
            and ok
        )
    else:
        ok = _status(False, "TAS column not found") and ok

    return ok


# ---------------------------------------------------------------------------
# T3 — Distance monotonicity (AC2)
# ---------------------------------------------------------------------------


def validate_distance_monotonicity(df: pl.DataFrame) -> bool:
    """Check that distance_along_track_m is monotonically non-decreasing per flight.

    Tolerance of -1.0 m for GPS jitter.

    Args:
        df: Processed dataframe.

    Returns:
        True if monotonicity holds for all flights.
    """
    if "distance_along_track_m" not in df.columns:
        return _status(False, "distance_along_track_m column not found")

    ok = True
    violation_count = 0

    for (fid,), flight in df.group_by("flight_id"):
        d = flight.sort("timestamp")["distance_along_track_m"].to_numpy()
        diffs = np.diff(d)
        if not np.all(diffs >= -1.0):
            violation_count += 1
            if violation_count <= 5:
                min_diff = float(diffs.min())
                _status(False, f"Flight {fid}: min diff = {min_diff:.2f} m")
            ok = False

    if violation_count > 5:
        _status(False, f"... and {violation_count - 5} more flights non-monotone")

    if ok:
        n_flights = df["flight_id"].n_unique()
        _status(True, f"Distance monotone (tol -1.0 m) across {n_flights} flights")
    return ok


# ---------------------------------------------------------------------------
# T4 — Dataset split (AC3)
# ---------------------------------------------------------------------------


def validate_split(process_dir: Path) -> tuple[bool, pl.DataFrame | None]:
    """Validate dataset_split.csv structure and split ratios.

    Args:
        process_dir: Path to the process output directory.

    Returns:
        Tuple of (check passed, split dataframe or None).
    """
    split_csv = process_dir / "dataset_split.csv"
    if not split_csv.exists():
        _status(False, f"dataset_split.csv not found at {split_csv}")
        return False, None

    split_df = pl.read_csv(split_csv)
    ok = True

    # Required columns
    required = {"flight_id", "filepath", "split", "aircraft_type"}
    missing = required - set(split_df.columns)
    ok = _status(not missing, f"Split CSV columns present ({len(required)} expected)") and ok
    if missing:
        _status(False, f"  Missing: {sorted(missing)}")
        return False, split_df

    # All 3 splits present
    splits = set(split_df["split"].unique().to_list())
    expected = {"train", "val", "test"}
    ok = _status(splits == expected, f"All splits present: {sorted(splits)}") and ok

    # Print counts
    counts = split_df.group_by("split").len().sort("split")
    for row in counts.iter_rows(named=True):
        print(f"    {row['split']}: {row['len']} flights")

    # Train ratio ∈ [0.6, 0.8]
    total = len(split_df)
    train_count = len(split_df.filter(pl.col("split") == "train"))
    train_pct = train_count / total if total > 0 else 0.0
    ok = (
        _status(
            0.6 < train_pct < 0.8,
            f"Train ratio: {train_pct:.1%} (expected 60-80%)",
        )
        and ok
    )

    _status(True, f"Split total: {total} flights")
    return ok, split_df


# ---------------------------------------------------------------------------
# T5 — Per-flight files (AC4)
# ---------------------------------------------------------------------------


def validate_flight_files(process_dir: Path, split_df: pl.DataFrame) -> bool:
    """Verify that per-flight parquet files match split CSV rows.

    Args:
        process_dir: Path to the process output directory.
        split_df: Split dataframe from validate_split.

    Returns:
        True if file count matches CSV row count.
    """
    flights_dir = process_dir / "flights"
    if not flights_dir.exists():
        return _status(False, f"flights/ directory not found at {flights_dir}")

    parquets = list(flights_dir.glob("*.parquet"))
    n_files = len(parquets)
    n_rows = len(split_df)
    ok = n_files == n_rows
    return _status(ok, f"Flight files: {n_files} files vs {n_rows} CSV rows")


# ---------------------------------------------------------------------------
# Plots (AC5, AC6)
# ---------------------------------------------------------------------------


def plot_enriched_profile(df: pl.DataFrame, figure_dir: Path) -> None:
    """Generate enriched vertical profile (altitude + gamma_air) for a sample flight.

    Args:
        df: Processed dataframe.
        figure_dir: Directory to save the chart.
    """
    import altair as alt

    sample_id = df["flight_id"].unique().sort()[0]
    flight = df.filter(pl.col("flight_id") == sample_id).sort("timestamp")
    flight = flight.with_row_index("t")

    alt_col = "altitude" if "altitude" in flight.columns else "altitude_ft"

    base = alt.Chart(flight.to_pandas()).encode(x=alt.X("t:Q", title="Time step"))

    alt_chart = base.mark_line(color="steelblue").encode(
        y=alt.Y(f"{alt_col}:Q", title="Altitude (ft)"),
    )
    gamma_chart = base.mark_line(color="coral").encode(
        y=alt.Y("gamma_air:Q", title="gamma_air (rad)"),
    )

    chart = (
        alt.layer(alt_chart, gamma_chart)
        .resolve_scale(y="independent")
        .properties(
            title=f"Enriched vertical profile -- {sample_id}",
            width=700,
            height=300,
        )
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "03_enriched_profile.html"
    chart.save(output)
    _status(True, f"Enriched profile chart saved to {output}")


def plot_mach_vs_selected(df: pl.DataFrame, figure_dir: Path) -> None:
    """Generate Mach vs mach_sel overlay for a sample flight.

    Args:
        df: Processed dataframe.
        figure_dir: Directory to save the chart.
    """
    import altair as alt

    sample_id = df["flight_id"].unique().sort()[0]
    flight = df.filter(pl.col("flight_id") == sample_id).sort("timestamp")
    flight = flight.with_row_index("t")
    flight_pd = flight.to_pandas()

    mach_col = "Mach" if "Mach" in flight.columns else "mach"

    mach_chart = (
        alt.Chart(flight_pd)
        .mark_line()
        .encode(
            x=alt.X("t:Q", title="Time step"),
            y=alt.Y(f"{mach_col}:Q", title="Mach"),
            color=alt.value("blue"),
        )
    )
    mach_sel_chart = (
        alt.Chart(flight_pd)
        .mark_line(strokeDash=[5, 3])
        .encode(
            x=alt.X("t:Q", title="Time step"),
            y=alt.Y("mach_sel:Q", title="Mach"),
            color=alt.value("red"),
        )
    )

    chart = (mach_chart + mach_sel_chart).properties(
        title=f"Mach vs mach_sel -- {sample_id}",
        width=700,
        height=250,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "03_mach_vs_selected.html"
    chart.save(output)
    _status(True, f"Mach vs mach_sel chart saved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all processing validation checks."""
    print("=" * 60)
    print("  Processing Validation (03)")
    print("=" * 60)

    data_dir, process_dir = _load_config()

    print("\n--- Loading processed data ---")
    df = _load_processed(process_dir)
    if df is None:
        raise SystemExit(1)

    n_flights = df["flight_id"].n_unique()
    n_points = len(df)

    print("\n--- T1: Feature set validation ---")
    t1_ok = validate_feature_set(df)

    print("\n--- T2: Physical consistency ---")
    t2_ok = validate_physics(df)

    print("\n--- T3: Distance monotonicity ---")
    t3_ok = validate_distance_monotonicity(df)

    print("\n--- T4: Dataset split ---")
    t4_ok, split_df = validate_split(process_dir)

    print("\n--- T5: Per-flight files ---")
    if split_df is not None:
        t5_ok = validate_flight_files(process_dir, split_df)
    else:
        t5_ok = _status(False, "Skipped -- split CSV not loaded")

    print("\n--- Plots ---")
    figure_dir = data_dir / "figures"
    plot_enriched_profile(df, figure_dir)
    plot_mach_vs_selected(df, figure_dir)

    # Summary
    all_ok = t1_ok and t2_ok and t3_ok and t4_ok and t5_ok
    print("\n" + "=" * 60)
    icon = "✅" if all_ok else "❌"
    print(f"  {icon} Overall: {'PASS' if all_ok else 'FAIL'}")
    print(f"  📊 {n_flights} flights, {n_points:,} points, {len(df.columns)} columns")
    if n_points > 0:
        pts_per_flight = n_points / n_flights if n_flights > 0 else 0
        print(f"  📊 ~{pts_per_flight:.0f} points/flight avg")
    print("=" * 60)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
