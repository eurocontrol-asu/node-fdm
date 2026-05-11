"""Download AN-225 Mriya flights and plot profile."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from traffic.data import opensky


def main() -> None:
    icao = "508035"  # UR-82060 AN-225 Mriya
    print(f"Querying flightlist for AN-225 (icao24={icao}) over its active years")
    fl = opensky.flightlist("2018-01-01", "2022-03-01", icao24=[icao])
    if fl is None or len(fl) == 0:
        print("No flights")
        return
    print(f"{len(fl)} flights total")
    fl = fl.copy()
    fl["dur"] = (fl["lastseen"] - fl["firstseen"]).dt.total_seconds()
    fl = fl.sort_values("dur", ascending=False)
    print("\nTop 5 longest flights:")
    print(fl[["callsign", "firstseen", "lastseen", "dur"]].head(5).to_string())

    # Pick the longest
    pick = fl.iloc[0]
    start = pick["firstseen"].strftime("%Y-%m-%d %H:%M")
    stop = pick["lastseen"].strftime("%Y-%m-%d %H:%M")
    print(f"\nDownloading {pick['callsign']}: {start} -> {stop} ({pick['dur']/3600:.1f}h)")
    traj = opensky.history(start=start, stop=stop, icao24=icao)
    if traj is None:
        print("No traj")
        return
    df = traj.data.dropna(subset=["altitude", "groundspeed", "timestamp"])
    print(f"{len(df)} points")
    print(f"GS  range: {df['groundspeed'].min():.0f} - {df['groundspeed'].max():.0f} kt  "
          f"(median {df['groundspeed'].median():.0f})")
    print(f"alt range: {df['altitude'].min():.0f} - {df['altitude'].max():.0f} ft")

    t0 = df["timestamp"].min()
    tt = (df["timestamp"] - t0).dt.total_seconds() / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(tt, df["altitude"] / 1000, color="tab:blue", lw=0.8)
    axes[0].set_ylabel("Altitude (×1000 ft)")
    axes[0].set_title(
        f"AN-225 Mriya  |  {pick['callsign']}  |  {pick['firstseen'].date()}"
    )
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(tt, df["groundspeed"], color="tab:red", lw=0.8)
    axes[1].set_ylabel("Groundspeed (kt)")
    axes[1].set_xlabel("Time (min)")
    axes[1].grid(True, alpha=0.3)

    out = Path("data/figures/mriya_profile.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
