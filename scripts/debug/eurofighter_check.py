"""Investigate the suspicious 130 kt Eurofighter trajectory.

Checks:
- Raw data columns (groundspeed unit, missing TAS)
- Distance covered vs duration -> sanity check on speed
- Position track over time
"""

from __future__ import annotations

import math

import pandas as pd
from traffic.data import opensky


def main() -> None:
    icao = "3f8f0f"
    traj = opensky.history(
        start="2025-06-17 08:07", stop="2025-06-17 09:34", icao24=icao
    )
    df = traj.data.copy()
    print(f"rows: {len(df)}")
    print(f"\ncolumns: {list(df.columns)}")
    print(f"\nstats on key columns:")
    for col in ["groundspeed", "altitude", "track", "vertical_rate", "TAS", "IAS",
                "Mach", "vrate_baro_inertial", "selected_altitude"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                print(f"  {col:30s} count={len(s):5d}  min={s.min():>8.2f}  "
                      f"max={s.max():>8.2f}  mean={s.mean():>8.2f}")
            else:
                print(f"  {col:30s} ALL NaN")

    # Sanity: total distance vs duration
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if len(df) < 2:
        return
    R = 6371000.0  # m
    lat = df["latitude"].to_numpy() * math.pi / 180
    lon = df["longitude"].to_numpy() * math.pi / 180
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    a = (
        (dlat / 2).__abs__() * 0  # placeholder for haversine
    )
    # haversine proper:
    import numpy as np
    a = (np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2)
    seg_m = 2 * R * np.arcsin(np.sqrt(a))
    total_m = seg_m.sum()
    dur_s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
    avg_kt = (total_m / dur_s) / 0.514444
    print(f"\ntotal distance: {total_m / 1000:.1f} km, duration: {dur_s/60:.1f} min")
    print(f"=> average GS from positions: {avg_kt:.0f} kt")
    print(f"GS reported by ADS-B: median={df['groundspeed'].median():.0f} kt")

    # Lat/lon bounding box
    print(f"\nposition bbox:")
    print(f"  lat: {df['latitude'].min():.3f} -> {df['latitude'].max():.3f}")
    print(f"  lon: {df['longitude'].min():.3f} -> {df['longitude'].max():.3f}")
    span_lat = (df["latitude"].max() - df["latitude"].min()) * 111
    span_lon = (df["longitude"].max() - df["longitude"].min()) * 111 * math.cos(
        df["latitude"].mean() * math.pi / 180
    )
    print(f"  span: {span_lat:.1f} km N-S x {span_lon:.1f} km E-W")


if __name__ == "__main__":
    main()
