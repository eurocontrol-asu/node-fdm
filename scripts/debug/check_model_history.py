"""Test scaling: query OpenSky history for ALL icao24 of one borderline model."""

from __future__ import annotations

import time

import polars as pl
from traffic.data import aircraft, opensky


def main(
    model: str = "717-200",
    start: str = "2024-11-01",
    stop: str = "2025-11-01",
) -> None:
    df = pl.from_pandas(
        aircraft.data[["icao24", "model", "operator", "operatoricao"]]
    )
    sub = (
        df.filter(
            (pl.col("model") == model)
            & (
                (pl.col("operator").is_not_null() & (pl.col("operator") != ""))
                | (pl.col("operatoricao").is_not_null() & (pl.col("operatoricao") != ""))
            )
        )
        .select("icao24")
        .unique()
    )
    icaos = sub.to_series().to_list()
    print(f"model={model!r}: {len(icaos)} commercial icao24 in db")
    print(f"window: {start} -> {stop}")

    opensky.trino_client.connect()
    t0 = time.time()
    fl = opensky.flightlist(start, stop, icao24=icaos)
    dt = time.time() - t0
    if fl is None or len(fl) == 0:
        print(f"  no rows in {dt:.2f}s")
        return
    fl_pl = pl.from_pandas(fl)
    seen_icaos = fl_pl["icao24"].n_unique()
    n_flights = len(fl_pl)
    print(f"  {n_flights} flights, {seen_icaos}/{len(icaos)} icao24 seen in {dt:.2f}s")
    print(f"  coverage ratio: {seen_icaos / len(icaos):.0%}")


if __name__ == "__main__":
    main()
