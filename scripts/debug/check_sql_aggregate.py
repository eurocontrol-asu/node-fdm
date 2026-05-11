"""Test a raw aggregated Trino SQL query on flights_data4.

Goal: get one icao24's flight count per year via a single SQL query
that aggregates server-side, instead of pulling full flightlist.
"""

from __future__ import annotations

import time

from pyopensky.trino import Trino
from sqlalchemy import text


def main() -> None:
    icao = "3c6589"
    sql = text(
        f"""
        SELECT
          icao24,
          year(from_unixtime(firstseen)) AS year,
          COUNT(*) AS n_flights
        FROM flights_data4
        WHERE icao24 = '{icao}'
          AND day >= 1514764800  -- 2018-01-01 UTC
          AND day <  1778630400  -- 2026-05-08 UTC (today)
        GROUP BY icao24, year(from_unixtime(firstseen))
        ORDER BY year
        """
    )
    print("SQL:")
    print(sql)
    print()

    t = Trino()
    t0 = time.time()
    with t.connect() as conn:
        res = conn.execute(sql)
        rows = res.fetchall()
    print(f"\nDone in {time.time() - t0:.1f}s, {len(rows)} rows:")
    for r in rows:
        print(f"  {r}")


if __name__ == "__main__":
    main()
