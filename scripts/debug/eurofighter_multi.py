"""Download multiple Eurofighter flights and plot a grid of altitude+speed profiles."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from traffic.data import aircraft, opensky


def main() -> None:
    # Pick the 9 Eurofighters with most flights in 2024-2025
    df = pl.from_pandas(aircraft.data[["icao24", "typecode"]])
    ef_icaos = df.filter(pl.col("typecode") == "EUFI")["icao24"].to_list()
    print(f"{len(ef_icaos)} Eurofighters in db")

    t = pl.read_parquet("data/icao_yearly_flights.parquet")
    candidates = (
        t.filter(pl.col("icao24").is_in(ef_icaos) & (pl.col("year") >= 2024))
        .group_by("icao24")
        .agg(pl.col("n_flights").sum())
        .sort("n_flights", descending=True)
        .head(20)
    )
    print(f"top candidates: {len(candidates)}")

    # Get their flight lists, pick the longest single flight per icao
    flights_to_download = []
    for icao in candidates["icao24"].to_list():
        try:
            fl = opensky.flightlist("2024-01-01", "2026-05-08", icao24=[icao])
        except Exception:
            continue
        if fl is None or len(fl) == 0:
            continue
        fl = fl.copy()
        fl["dur"] = (fl["lastseen"] - fl["firstseen"]).dt.total_seconds()
        # Pick the longest
        pick = fl.sort_values("dur", ascending=False).iloc[0]
        if pick["dur"] < 600:  # less than 10 min, skip
            continue
        flights_to_download.append(
            {
                "icao": icao,
                "callsign": pick["callsign"],
                "start": pick["firstseen"].strftime("%Y-%m-%d %H:%M"),
                "stop": pick["lastseen"].strftime("%Y-%m-%d %H:%M"),
                "date": pick["firstseen"].date(),
                "dur_min": pick["dur"] / 60,
            }
        )
        if len(flights_to_download) >= 9:
            break

    print(f"\nDownloading {len(flights_to_download)} flights...")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    for i, f in enumerate(flights_to_download):
        try:
            traj = opensky.history(start=f["start"], stop=f["stop"], icao24=f["icao"])
            d = traj.data
        except Exception as exc:
            print(f"  [{i}] {f['icao']} FAILED: {exc!s:.100}")
            continue
        if d is None or len(d) == 0:
            continue
        d = d.dropna(subset=["altitude", "groundspeed", "timestamp"])
        if len(d) < 30:
            continue
        t0 = d["timestamp"].min()
        tt = (d["timestamp"] - t0).dt.total_seconds() / 60.0
        ax = axes[i]
        ax2 = ax.twinx()
        ax.plot(tt, d["altitude"], "b-", lw=0.8, label="alt")
        ax2.plot(tt, d["groundspeed"], "r-", lw=0.8, label="GS")
        ax.set_title(
            f"{f['icao']} {f['callsign']} {f['date']}\n"
            f"alt {d['altitude'].max():.0f}ft / GS {d['groundspeed'].max():.0f}kt",
            fontsize=9,
        )
        ax.set_xlabel("min")
        ax.set_ylabel("alt (ft)", color="b")
        ax2.set_ylabel("GS (kt)", color="r")
        ax.grid(True, alpha=0.3)
        print(
            f"  [{i}] {f['icao']:8s} {f['callsign']} {f['date']} "
            f"alt_max={d['altitude'].max():>5.0f}ft  GS_max={d['groundspeed'].max():>4.0f}kt  "
            f"({len(d)} pts)"
        )

    out = Path("data/figures/eurofighter_multi.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
