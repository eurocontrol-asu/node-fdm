"""Quick latency test: query OpenSky history for 5 specific icao24 on 1 day."""

from __future__ import annotations

import time
from datetime import datetime

from traffic.data import opensky


def main() -> None:
    icao24s = ["3c6589", "4b1815", "406a3c", "a4fa61", "484e6e"]
    start = "2024-11-01"
    stop = "2025-11-01"

    opensky.trino_client.connect()

    print(f"Querying flightlist for {len(icao24s)} icao24 on {start} -> {stop}")
    t0 = time.time()
    fl = opensky.flightlist(start, stop, icao24=icao24s)
    dt = time.time() - t0
    print(f"  flightlist: {len(fl) if fl is not None else 0} rows in {dt:.2f}s")
    if fl is not None and len(fl):
        print(fl[["icao24", "callsign", "firstseen", "lastseen"]].head(20))


if __name__ == "__main__":
    main()
