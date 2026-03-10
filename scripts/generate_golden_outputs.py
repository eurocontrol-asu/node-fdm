#!/usr/bin/env python3
"""Generate golden outputs from the new (v2) code.

Run once to freeze expected behavior:
    uv run python scripts/generate_golden_outputs.py

These outputs validate that the Polars refactoring is numerically
equivalent to the legacy pandas code.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from node_fdm_data.conversions import ft_to_m, ftmin_to_ms, kt_to_ms

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
GOLDEN_DIR = Path(__file__).parent.parent / "tests" / "golden" / "outputs"


def generate_raw_conversion_golden(flight_path: Path) -> None:
    """Generate golden output for raw value extraction."""
    df = pl.read_parquet(flight_path)

    key_cols = [
        "altitude",
        "TAS",
        "groundspeed",
        "Mach",
        "vertical_rate",
        "temperature",
    ]
    subset = df.select(key_cols)

    output_name = f"raw_{flight_path.stem}.parquet"
    subset.write_parquet(GOLDEN_DIR / output_name)
    print(f"✓ Generated {output_name}")


def generate_conversion_golden(flight_path: Path) -> None:
    """Generate golden output for unit conversions."""
    df = pl.read_parquet(flight_path)

    converted = df.select(
        ft_to_m("altitude").alias("altitude_m"),
        kt_to_ms("TAS").alias("tas_ms"),
        kt_to_ms("groundspeed").alias("gs_ms"),
        ftmin_to_ms("vertical_rate").alias("vz_ms"),
    )

    output_name = f"converted_{flight_path.stem}.parquet"
    converted.write_parquet(GOLDEN_DIR / output_name)
    print(f"✓ Generated {output_name}")


def generate_stats_golden(flight_paths: list[Path]) -> None:
    """Generate golden statistics from raw data."""
    all_dfs = [pl.read_parquet(p) for p in flight_paths]
    combined = pl.concat(all_dfs)

    numeric_cols = [
        col
        for col in combined.columns
        if combined[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]

    stats: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        values = combined[col].drop_nulls()
        stats[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "count": len(values),
        }

    with open(GOLDEN_DIR / "raw_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Generated raw_stats.json ({len(stats)} columns)")


def main() -> None:
    """Generate all golden outputs."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    flights = sorted(FIXTURES_DIR.glob("flight_*.parquet"))

    if not flights:
        print("⚠️  No fixture flights found in fixtures/")
        print("   Run: uv run python scripts/create_test_fixtures.py")
        return

    print(f"Found {len(flights)} fixture flights\n")

    print("=== Raw Conversions ===")
    for flight in flights:
        generate_raw_conversion_golden(flight)

    print("\n=== Unit Conversions ===")
    for flight in flights:
        generate_conversion_golden(flight)

    print("\n=== Statistics ===")
    generate_stats_golden(flights)

    print(f"\n✅ Golden outputs generated in {GOLDEN_DIR}")


if __name__ == "__main__":
    main()
