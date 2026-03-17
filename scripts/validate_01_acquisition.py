#!/usr/bin/env python3
"""Validate data acquisition pipeline output (aircraft-list + download).

Checks data integrity, completeness, and cross-file consistency after
running ``fdm aircraft-list`` and ``fdm download``.

Run:
    cd /Users/gabriel/Documents/Code/python/node-fdm-v2
    uv run python scripts/validate_01_acquisition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> tuple[Path, Path, list[str]]:
    """Load pipeline config and return (data_dir, download_dir, typecodes)."""
    sys.path.insert(0, str(PROJECT_ROOT / "packages" / "node-fdm-pipeline" / "src"))
    from node_fdm_pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml(PROJECT_ROOT / "config.yaml")
    return cfg.paths.data_dir, cfg.paths.resolve("download_dir"), cfg.typecodes


def _status(ok: bool, msg: str) -> bool:
    """Print a status line and return the check result."""
    icon = "✅" if ok else "❌"
    print(f"  {icon} {msg}")
    return ok


# ---------------------------------------------------------------------------
# T1 — aircraft_db.csv integrity
# ---------------------------------------------------------------------------


def validate_aircraft_db(data_dir: Path, typecodes: list[str]) -> bool:
    """Validate aircraft_db.csv structure, uniqueness, and typecodes.

    Args:
        data_dir: Root data directory containing aircraft_db.csv.
        typecodes: Expected typecodes from config.

    Returns:
        True if all checks pass.
    """
    import polars as pl

    csv_path = data_dir / "aircraft_db.csv"
    if not csv_path.exists():
        return _status(False, f"aircraft_db.csv not found at {csv_path}")

    db = pl.read_csv(csv_path)

    required = {"icao24", "registration", "typecode", "age", "airline"}
    missing = required - set(db.columns)
    ok = _status(
        not missing,
        f"Columns: {sorted(db.columns)}" + (f" (missing: {missing})" if missing else ""),
    )

    dup_count = len(db) - db["icao24"].n_unique()
    ok &= _status(
        dup_count == 0, f"{db['icao24'].n_unique()} unique ICAO24 ({dup_count} duplicates)"
    )

    actual_tc = sorted(db["typecode"].unique().to_list())
    tc_match = set(actual_tc) <= set(typecodes)
    ok &= _status(tc_match, f"Typecodes: {actual_tc} (expected subset of {typecodes})")

    _status(True, f"{len(db)} aircraft in aircraft_db.csv")
    return ok


# ---------------------------------------------------------------------------
# T2 — Parquet completeness
# ---------------------------------------------------------------------------


def validate_parquets(download_dir: Path) -> tuple[bool, list[Path]]:
    """Check that history/flightlist/extended parquet files exist and are non-empty.

    Args:
        download_dir: Directory containing downloaded parquet files.

    Returns:
        Tuple of (all_ok, list of history parquet paths found).
    """
    import polars as pl

    if not download_dir.exists():
        _status(False, f"Download directory not found: {download_dir}")
        return False, []

    ok = True
    history_files: list[Path] = []

    for kind in ("history", "flightlist", "extended"):
        files = sorted(download_dir.glob(f"{kind}_*.parquet"))
        if not files:
            ok &= _status(False, f"No {kind}_*.parquet found")
            continue

        for f in files:
            df = pl.read_parquet(f)
            is_ok = len(df) > 0
            ok &= _status(is_ok, f"{f.name}: {len(df):,} rows, {len(df.columns)} cols")
            if kind == "history":
                history_files.append(f)

    return ok, history_files


# ---------------------------------------------------------------------------
# T3 — ICAO24 coherence
# ---------------------------------------------------------------------------


def validate_icao24_overlap(data_dir: Path, history_files: list[Path]) -> bool:
    """Compute ICAO24 overlap between aircraft_db.csv and history parquets.

    Args:
        data_dir: Root data directory containing aircraft_db.csv.
        history_files: List of history parquet file paths.

    Returns:
        True if overlap is non-zero.
    """
    import polars as pl

    db_icao = set(pl.read_csv(data_dir / "aircraft_db.csv")["icao24"].to_list())

    history_icao: set[str] = set()
    for f in history_files:
        df = pl.read_parquet(f)
        history_icao |= set(df["icao24"].unique().to_list())

    overlap = db_icao & history_icao
    ratio = len(overlap) / len(db_icao) if db_icao else 0.0

    ok = len(overlap) > 0
    _status(ok, f"ICAO24 overlap: {len(overlap)}/{len(db_icao)} ({ratio:.1%})")
    return ok


# ---------------------------------------------------------------------------
# Plot — Temporal coverage
# ---------------------------------------------------------------------------


def plot_temporal_coverage(history_files: list[Path], figure_dir: Path) -> None:
    """Generate an Altair bar chart of StateVectors per hour.

    Args:
        history_files: List of history parquet file paths.
        figure_dir: Directory to save the chart.
    """
    import altair as alt
    import polars as pl

    frames = [pl.read_parquet(f) for f in history_files]
    history = pl.concat(frames)

    hourly = (
        history.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.len().alias("count"))
        .sort("hour")
    )

    chart = (
        alt.Chart(hourly)
        .mark_bar()
        .encode(
            x=alt.X("hour:O").title("Hour of day"),
            y=alt.Y("count:Q").title("StateVectors"),
            tooltip=["hour", "count"],
        )
        .properties(title="Temporal coverage — StateVectors per hour", width=500)
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "01_temporal_coverage.html"
    chart.save(output)
    _status(True, f"Temporal coverage chart saved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all acquisition validation checks."""
    print("=" * 60)
    print("  Data Acquisition Validation (01)")
    print("=" * 60)

    data_dir, download_dir, typecodes = _load_config()

    print("\n--- T1: aircraft_db.csv integrity ---")
    t1_ok = validate_aircraft_db(data_dir, typecodes)

    print("\n--- T2: Parquet completeness ---")
    t2_ok, history_files = validate_parquets(download_dir)

    print("\n--- T3: ICAO24 coherence ---")
    if history_files:
        t3_ok = validate_icao24_overlap(data_dir, history_files)
    else:
        t3_ok = _status(False, "No history files — skipping ICAO24 check")

    print("\n--- Plot: Temporal coverage ---")
    if history_files:
        figure_dir = data_dir / "figures"
        plot_temporal_coverage(history_files, figure_dir)
    else:
        _status(False, "No history files — skipping plot")

    # Summary
    all_ok = t1_ok and t2_ok and t3_ok
    print("\n" + "=" * 60)
    icon = "✅" if all_ok else "❌"
    print(f"  {icon} Overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
