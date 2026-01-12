#!/usr/bin/env python3
"""Generate golden outputs from legacy pandas code.

Run this with the legacy environment to freeze expected behavior:
    legacy/.venv/bin/python scripts/generate_golden_outputs.py

These outputs are then used to validate the Polars refactoring.
"""

import json
import sys
from pathlib import Path

# Add legacy package to path
LEGACY_DIR = Path(__file__).parent.parent / "legacy"
sys.path.insert(0, str(LEGACY_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden" / "outputs"


def generate_preprocessing_golden(flight_path: Path) -> None:
    """Generate golden output for FlightProcessor."""
    from node_fdm.architectures.opensky_2025.columns import (
        col_alt,
        col_alt_sel,
        col_cas_sel,
        col_gamma,
        col_lat,
        col_long,
        col_long_wind_spd,
        col_mach,
        col_mach_sel,
        col_tas,
        col_temp,
        col_vz,
        col_vz_sel,
    )
    from node_fdm.data.flight_processor import FlightProcessor

    # Load with pandas (legacy)
    df = pd.read_parquet(flight_path)

    # Define model columns (simplified - just state columns for now)
    x_cols = [col_alt, col_tas, col_mach, col_vz, col_gamma]
    u_cols = [col_alt_sel, col_vz_sel, col_mach_sel, col_cas_sel]
    e0_cols = [col_lat, col_long]
    e_cols = [col_temp, col_long_wind_spd]
    dx_cols = [(1.0, col_alt.derivative), (1.0, col_tas.derivative)]

    model_cols = (x_cols, u_cols, e0_cols, e_cols, dx_cols)

    # Process with legacy code
    processor = FlightProcessor(model_cols)
    processed = processor.process_flight(df)

    # Save golden output
    output_name = f"preprocessed_{flight_path.stem}.parquet"
    processed.to_parquet(GOLDEN_DIR / output_name)
    print(
        f"✓ Generated {output_name} "
        f"({len(processed)} rows, {len(processed.columns)} cols)"
    )


def generate_raw_conversion_golden(flight_path: Path) -> None:
    """Generate golden output for raw value extraction (simpler test)."""
    df = pd.read_parquet(flight_path)

    # Just save key columns with their raw values
    key_cols = [
        "altitude",
        "TAS",
        "groundspeed",
        "Mach",
        "vertical_rate",
        "temperature",
    ]
    subset = df[key_cols].copy()

    output_name = f"raw_{flight_path.stem}.parquet"
    subset.to_parquet(GOLDEN_DIR / output_name)
    print(f"✓ Generated {output_name}")


def generate_stats_golden(flight_paths: list[Path]) -> None:
    """Generate golden statistics from raw data."""
    all_dfs = [pd.read_parquet(p) for p in flight_paths]
    combined = pd.concat(all_dfs, ignore_index=True)

    # Compute stats for numeric columns
    stats = {}
    for col in combined.select_dtypes(include=[np.number]).columns:
        vals = combined[col].dropna()
        stats[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "count": int(len(vals)),
        }

    with open(GOLDEN_DIR / "raw_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Generated raw_stats.json ({len(stats)} columns)")


def main() -> None:
    """Generate all golden outputs."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Find fixture flights
    flights = sorted(FIXTURES_DIR.glob("flight_*.parquet"))

    if not flights:
        print("⚠️  No fixture flights found in tests/fixtures/")
        print("   Run: legacy/.venv/bin/python scripts/create_test_fixtures.py")
        return

    print(f"Found {len(flights)} fixture flights\n")

    # Generate raw conversion outputs (no node_fdm dependency)
    print("=== Raw Conversions ===")
    for flight in flights:
        generate_raw_conversion_golden(flight)

    # Generate statistics
    print("\n=== Statistics ===")
    generate_stats_golden(flights)

    # Try to generate preprocessing outputs (requires node_fdm)
    print("\n=== Preprocessing ===")
    try:
        for flight in flights:
            generate_preprocessing_golden(flight)
    except Exception as e:
        print(f"⚠️  Preprocessing failed: {e}")
        print("   This is expected if the legacy code has missing dependencies.")
        print("   Raw conversion outputs are still valid for basic tests.")

    print(f"\n✅ Golden outputs generated in {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
