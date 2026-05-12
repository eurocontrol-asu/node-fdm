"""Compare raw history vs preprocessed Delta for a single flight.

Usage:
    uv run python scripts/debug/plot_raw_vs_preprocessed.py 02010d MAC302 20250901 02010d_MAC302_s0
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from deltalake import DeltaTable

icao24, callsign, date, flight_id = sys.argv[1:5]

raw = (
    pl.read_parquet(f"data/raw/history/date={date}/icao24={icao24}/data.parquet")
    .filter(pl.col("callsign").str.strip_chars() == callsign)
    .sort("timestamp")
)
print(f"raw history rows: {raw.height}")

tbl = (
    DeltaTable("data/flights.delta")
    .to_pyarrow_dataset()
    .to_table(
        columns=[
            "meta_flight_id",
            "raw_timestamp",
            "raw_lat_deg",
            "raw_lon_deg",
            "raw_alt_ft",
            "raw_gs_kt",
            "raw_vz_ftmin",
            "fdm_flag_valid",
        ]
    )
)
proc = pl.from_arrow(tbl).filter(pl.col("meta_flight_id") == flight_id).sort("raw_timestamp")
print(f"preprocessed rows: {proc.height}  valid: {proc['fdm_flag_valid'].sum()}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.plot(raw["longitude"], raw["latitude"], "k.", ms=2, alpha=0.5, label="raw history")
ax.plot(proc["raw_lon_deg"], proc["raw_lat_deg"], "r.", ms=2, alpha=0.5, label="preprocessed")
ax.set_xlabel("lon [°]")
ax.set_ylabel("lat [°]")
ax.set_title("Ground track")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(raw["timestamp"], raw["altitude"], "k.", ms=2, alpha=0.5, label="raw altitude")
ax.plot(raw["timestamp"], raw["geoaltitude"], "g.", ms=2, alpha=0.5, label="raw geoaltitude")
ax.plot(proc["raw_timestamp"], proc["raw_alt_ft"], "r.", ms=2, alpha=0.5, label="preprocessed")
ax.set_ylabel("alt [ft]")
ax.set_title("Altitude")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(raw["timestamp"], raw["groundspeed"], "k.", ms=2, alpha=0.5, label="raw")
ax.plot(proc["raw_timestamp"], proc["raw_gs_kt"], "r.", ms=2, alpha=0.5, label="preprocessed")
ax.set_ylabel("GS [kt]")
ax.set_title("Groundspeed")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(raw["timestamp"], raw["vertical_rate"], "k.", ms=2, alpha=0.5, label="raw")
ax.plot(proc["raw_timestamp"], proc["raw_vz_ftmin"], "r.", ms=2, alpha=0.5, label="preprocessed")
ax.set_ylabel("VZ [ft/min]")
ax.set_title("Vertical rate")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle(f"Raw vs preprocessed — {flight_id}")
fig.tight_layout()
out = Path(f"data/figures/raw_vs_preprocessed_{flight_id}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print(f"Saved: {out}")
