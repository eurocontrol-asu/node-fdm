#!/usr/bin/env python3
"""Create synthetic flight data for testing.

Creates test fixtures with the column names expected by the OpenSky 2025
architecture.  Uses Polars for consistency with the v2 codebase.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

REQUIRED_COLUMNS = [
    "distance_along_track_m",
    "adep_dist",
    "ades_dist",
    "longitude",
    "latitude",
    "altitude",
    "TAS",
    "CAS",
    "groundspeed",
    "Mach",
    "vertical_rate",
    "gamma_air",
    "selected_mcp",
    "vz_sel",
    "mach_sel",
    "cas_sel",
    "temperature",
    "long_wind",
]


def create_synthetic_flight(
    n_points: int = 300,
    seed: int = 42,
) -> pl.DataFrame:
    """Create a synthetic flight trajectory with realistic values."""
    rng = np.random.default_rng(seed)

    n_climb = n_points // 3
    n_cruise = n_points // 3
    n_descent = n_points - n_climb - n_cruise

    alt_profile = np.concatenate(
        [
            np.linspace(1000, 35000, n_climb),
            np.full(n_cruise, 35000),
            np.linspace(35000, 1000, n_descent),
        ]
    )
    altitude = alt_profile + rng.normal(0, 50, n_points)

    vertical_rate = np.gradient(altitude) * 60
    vertical_rate = np.clip(vertical_rate, -3000, 3000)

    tas = 150 + 200 * (altitude / 35000) + rng.normal(0, 5, n_points)
    cas = tas * (1 - altitude / 100000)
    groundspeed = tas + rng.normal(-10, 20, n_points)
    mach = tas / 600

    gamma_air = np.arctan2(vertical_rate / 60, groundspeed * 1852 / 3600)

    latitude = 45.0 + np.cumsum(rng.normal(0, 0.001, n_points))
    longitude = 2.0 + np.cumsum(rng.normal(0, 0.002, n_points))

    distance_along_track_m = np.cumsum(
        np.sqrt(
            (np.diff(latitude, prepend=latitude[0]) * 111000) ** 2
            + (np.diff(longitude, prepend=longitude[0]) * 111000 * np.cos(np.radians(45))) ** 2
        )
    )

    selected_mcp = np.full(n_points, 35000.0)
    selected_mcp[n_climb + n_cruise :] = 1000

    vz_sel = np.zeros(n_points)
    vz_sel[:n_climb] = 2000
    vz_sel[n_climb + n_cruise :] = -1500

    mach_sel = np.full(n_points, 0.78)
    cas_sel = np.full(n_points, 280.0)

    temperature = 288.15 - 0.0065 * altitude * 0.3048
    long_wind = rng.normal(0, 20, n_points)

    adep_dist = distance_along_track_m / 1852
    ades_dist = (distance_along_track_m.max() - distance_along_track_m) / 1852

    return pl.DataFrame(
        {
            "distance_along_track_m": distance_along_track_m,
            "adep_dist": adep_dist,
            "ades_dist": ades_dist,
            "longitude": longitude,
            "latitude": latitude,
            "altitude": altitude,
            "TAS": tas,
            "CAS": cas,
            "groundspeed": groundspeed,
            "Mach": mach,
            "vertical_rate": vertical_rate,
            "gamma_air": gamma_air,
            "selected_mcp": selected_mcp,
            "vz_sel": vz_sel,
            "mach_sel": mach_sel,
            "cas_sel": cas_sel,
            "temperature": temperature,
            "long_wind": long_wind,
        }
    )


def main() -> None:
    """Create test fixtures."""
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    short = create_synthetic_flight(n_points=300, seed=42)
    short.write_parquet(FIXTURES_DIR / "flight_short.parquet")
    print(f"✓ Created flight_short.parquet ({len(short)} points)")

    medium = create_synthetic_flight(n_points=1800, seed=43)
    medium.write_parquet(FIXTURES_DIR / "flight_medium.parquet")
    print(f"✓ Created flight_medium.parquet ({len(medium)} points)")

    metadata = {
        "flight_short": {
            "points": 300,
            "duration_min": 5,
            "seed": 42,
            "columns": short.columns,
        },
        "flight_medium": {
            "points": 1800,
            "duration_min": 30,
            "seed": 43,
            "columns": medium.columns,
        },
    }

    with open(FIXTURES_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Test fixtures created in {FIXTURES_DIR}")
    print(f"   Columns: {len(REQUIRED_COLUMNS)}")


if __name__ == "__main__":
    main()
