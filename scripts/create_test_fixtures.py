#!/usr/bin/env python3
"""Create synthetic flight data for testing.

This creates test fixtures with the exact column names expected by
the OpenSky 2025 architecture. Uses pandas for compatibility with legacy.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"

# Column names from opensky_2025/columns.py (raw_name attribute)
REQUIRED_COLUMNS = [
    "distance_along_track_m",
    "adep_dist",
    "ades_dist",
    "longitude",
    "latitude",
    "altitude",  # feet
    "TAS",  # knots
    "CAS",  # knots
    "groundspeed",  # knots
    "Mach",
    "vertical_rate",  # ft/min
    "gamma_air",  # radians
    "selected_mcp",  # feet
    "vz_sel",  # ft/min
    "mach_sel",
    "cas_sel",  # knots
    "temperature",  # kelvin
    "long_wind",  # knots
]


def create_synthetic_flight(
    n_points: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic flight trajectory with realistic values."""
    rng = np.random.default_rng(seed)

    # Altitude profile (climb, cruise, descent) in feet
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

    # Vertical rate (ft/min) - derivative of altitude
    vertical_rate = np.gradient(altitude) * 60
    vertical_rate = np.clip(vertical_rate, -3000, 3000)

    # Speed profile (knots)
    tas = 150 + 200 * (altitude / 35000) + rng.normal(0, 5, n_points)
    cas = tas * (1 - altitude / 100000)  # Simplified CAS
    groundspeed = tas + rng.normal(-10, 20, n_points)  # Wind effect

    # Mach number
    mach = tas / 600  # Simplified

    # Flight path angle (radians)
    gamma_air = np.arctan2(vertical_rate / 60, groundspeed * 1852 / 3600)

    # Position
    latitude = 45.0 + np.cumsum(rng.normal(0, 0.001, n_points))
    longitude = 2.0 + np.cumsum(rng.normal(0, 0.002, n_points))

    # Distance
    distance_along_track_m = np.cumsum(
        np.sqrt(
            (np.diff(latitude, prepend=latitude[0]) * 111000) ** 2
            + (
                np.diff(longitude, prepend=longitude[0])
                * 111000
                * np.cos(np.radians(45))
            )
            ** 2
        )
    )

    # Selected values (autopilot targets)
    selected_mcp = np.full(n_points, 35000.0)
    selected_mcp[:n_climb] = 35000
    selected_mcp[n_climb + n_cruise :] = 1000

    vz_sel = np.zeros(n_points)
    vz_sel[:n_climb] = 2000
    vz_sel[n_climb + n_cruise :] = -1500

    mach_sel = np.full(n_points, 0.78)
    cas_sel = np.full(n_points, 280.0)

    # Environment
    temperature = 288.15 - 0.0065 * altitude * 0.3048  # ISA in Kelvin
    long_wind = rng.normal(0, 20, n_points)  # Headwind/tailwind

    # Airport distances (NM)
    adep_dist = distance_along_track_m / 1852
    ades_dist = (distance_along_track_m.max() - distance_along_track_m) / 1852

    return pd.DataFrame(
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

    # Short flight (5 min @ 1Hz)
    short = create_synthetic_flight(n_points=300, seed=42)
    short.to_parquet(FIXTURES_DIR / "flight_short.parquet")
    print(f"✓ Created flight_short.parquet ({len(short)} points)")

    # Medium flight (30 min @ 1Hz)
    medium = create_synthetic_flight(n_points=1800, seed=43)
    medium.to_parquet(FIXTURES_DIR / "flight_medium.parquet")
    print(f"✓ Created flight_medium.parquet ({len(medium)} points)")

    # Metadata
    metadata = {
        "flight_short": {
            "points": 300,
            "duration_min": 5,
            "seed": 42,
            "columns": list(short.columns),
        },
        "flight_medium": {
            "points": 1800,
            "duration_min": 30,
            "seed": 43,
            "columns": list(medium.columns),
        },
    }

    with open(FIXTURES_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Test fixtures created in {FIXTURES_DIR}")
    print(f"   Columns: {len(REQUIRED_COLUMNS)}")


if __name__ == "__main__":
    main()
