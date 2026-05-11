"""Download an Eurofighter flight history and plot velocity profile."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from traffic.data import opensky


def main() -> None:
    icao = "3f8f0f"  # German Air Force 30+15

    # Find a flight in 2025
    print(f"Finding flights for icao24={icao} (2025)...")
    fl = opensky.flightlist("2025-01-01", "2026-01-01", icao24=[icao])
    if fl is None or len(fl) == 0:
        print("No flights found")
        return
    print(f"Found {len(fl)} flights")
    print(fl[["callsign", "firstseen", "lastseen"]].head())

    # Pick the longest flight
    fl["dur"] = (fl["lastseen"] - fl["firstseen"]).dt.total_seconds()
    fl = fl.sort_values("dur", ascending=False)
    pick = fl.iloc[0]
    start = pick["firstseen"].strftime("%Y-%m-%d %H:%M")
    stop = pick["lastseen"].strftime("%Y-%m-%d %H:%M")
    print(f"\nDownloading trajectory: {pick.callsign}  {start} -> {stop}")

    traj = opensky.history(start=start, stop=stop, icao24=icao)
    if traj is None:
        print("No trajectory")
        return
    df = traj.data.copy()
    print(f"Got {len(df)} points")

    # Compute groundspeed in kt (already kt in OpenSky), and altitude in ft
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t0 = df["timestamp"].min()
    t = (df["timestamp"] - t0).dt.total_seconds() / 60.0  # minutes

    axes[0].plot(t, df["altitude"], color="tab:blue")
    axes[0].set_ylabel("Altitude (ft)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"Eurofighter {pick.callsign} — {pick['firstseen'].date()}")

    axes[1].plot(t, df["groundspeed"], color="tab:red", label="GS")
    if "TAS" in df.columns and df["TAS"].notna().any():
        axes[1].plot(t, df["TAS"], color="tab:green", label="TAS")
    axes[1].set_ylabel("Speed (kt)")
    axes[1].set_xlabel("Time (min)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    out = Path("data/figures/eurofighter_profile.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")
    print(f"\nGS range: {df['groundspeed'].min():.0f} - {df['groundspeed'].max():.0f} kt")
    print(f"Alt range: {df['altitude'].min():.0f} - {df['altitude'].max():.0f} ft")


if __name__ == "__main__":
    main()
