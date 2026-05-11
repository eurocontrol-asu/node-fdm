"""Test aggregated Trino SQL on 500 icao24."""

from __future__ import annotations

import time

import polars as pl
from pyopensky.trino import Trino
from sqlalchemy import text
from traffic.data import aircraft


def main() -> None:
    df = pl.from_pandas(aircraft.data[["icao24", "operator", "operatoricao"]])
    commercial = df.filter(
        (pl.col("operator").is_not_null() & (pl.col("operator") != ""))
        | (pl.col("operatoricao").is_not_null() & (pl.col("operatoricao") != ""))
    )
    icaos = commercial["icao24"].unique().head(20000).to_list()
    print(f"Testing {len(icaos)} icao24")

    in_list = ", ".join(f"'{i}'" for i in icaos)
    sql = text(
        f"""
        SELECT
          icao24,
          year(from_unixtime(firstseen)) AS year,
          COUNT(*) AS n_flights
        FROM flights_data4
        WHERE icao24 IN ({in_list})
          AND day >= 1514764800  -- 2018-01-01
          AND day <  1778630400  -- 2026-05-08
        GROUP BY icao24, year(from_unixtime(firstseen))
        """
    )

    t = Trino()
    t0 = time.time()
    with t.connect() as conn:
        rows = conn.execute(sql).fetchall()
    dt = time.time() - t0
    print(f"\n{len(rows)} (icao, year) pairs returned in {dt:.1f}s")

    res = pl.DataFrame(rows, schema=["icao24", "year", "n_flights"], orient="row")
    seen = res["icao24"].n_unique()
    print(f"distinct icao24 with at least one flight: {seen}/{len(icaos)}")
    print(f"total flights: {res['n_flights'].sum():,}")
    print(res.head(15))


if __name__ == "__main__":
    main()
