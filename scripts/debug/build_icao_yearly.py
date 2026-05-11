"""Build a yearly flight-count table for all icao24 in aircraft.data.

Aggregates server-side via Trino, batched by BATCH_SIZE icao24, years 2018..today.

Output:
    data/icao_yearly_flights.parquet
        columns: icao24, year, n_flights

Usage:
    uv run python scripts/debug/build_icao_yearly.py
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from pyopensky.trino import Trino
from sqlalchemy import text
from traffic.data import aircraft

BATCH_SIZE = 50_000
YEAR_START = 2018
DAY_START_EPOCH = int(datetime(YEAR_START, 1, 1, tzinfo=timezone.utc).timestamp())
DAY_END_EPOCH = int(datetime.now(tz=timezone.utc).timestamp())

OUTPUT = Path("data/icao_yearly_flights.parquet")
CHECKPOINT_DIR = Path("data/icao_yearly_batches")


def query_batch(trino: Trino, icaos: list[str]) -> pl.DataFrame:
    in_list = ", ".join(f"'{i}'" for i in icaos)
    sql = text(
        f"""
        SELECT
          icao24,
          year(from_unixtime(firstseen)) AS year,
          COUNT(*) AS n_flights
        FROM flights_data4
        WHERE icao24 IN ({in_list})
          AND day >= {DAY_START_EPOCH}
          AND day <  {DAY_END_EPOCH}
        GROUP BY icao24, year(from_unixtime(firstseen))
        """
    )
    with trino.connect() as conn:
        rows = conn.execute(sql).fetchall()
    if not rows:
        return pl.DataFrame(
            schema={"icao24": pl.Utf8, "year": pl.Int64, "n_flights": pl.Int64}
        )
    return pl.DataFrame(
        rows, schema=["icao24", "year", "n_flights"], orient="row"
    )


def main() -> None:
    df = pl.from_pandas(aircraft.data[["icao24"]])
    icaos = df["icao24"].drop_nulls().unique().sort().to_list()
    print(f"Total icao24 in aircraft.data: {len(icaos):,}")
    print(f"Window: {YEAR_START}-01-01 -> today")
    print(f"Batch size: {BATCH_SIZE}, batches: {(len(icaos) + BATCH_SIZE - 1) // BATCH_SIZE}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    trino = Trino()
    parts: list[pl.DataFrame] = []
    t_total = time.time()

    for i in range(0, len(icaos), BATCH_SIZE):
        batch_idx = i // BATCH_SIZE
        ckpt = CHECKPOINT_DIR / f"batch_{batch_idx:04d}.parquet"
        if ckpt.exists():
            print(f"[{batch_idx}] cached -> {ckpt.name}")
            parts.append(pl.read_parquet(ckpt))
            continue
        batch = icaos[i : i + BATCH_SIZE]
        t0 = time.time()
        try:
            res = query_batch(trino, batch)
        except Exception as exc:  # noqa: BLE001
            print(f"[{batch_idx}] FAILED ({type(exc).__name__}: {exc!s:.200})")
            continue
        dt = time.time() - t0
        res.write_parquet(ckpt)
        parts.append(res)
        print(
            f"[{batch_idx}] {len(batch):>5} icao -> "
            f"{res['icao24'].n_unique():>5} seen, "
            f"{len(res):>6} pairs, {res['n_flights'].sum():>10,} flights "
            f"in {dt:5.1f}s  ({(time.time() - t_total) / 60:.1f} min total)"
        )

    if not parts:
        raise SystemExit("No batches succeeded")
    out = pl.concat(parts)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUTPUT)
    print(f"\nDone in {(time.time() - t_total) / 60:.1f} min")
    print(f"Wrote {OUTPUT}: {len(out):,} rows, "
          f"{out['icao24'].n_unique():,} distinct icao24, "
          f"{out['n_flights'].sum():,} total flights")


if __name__ == "__main__":
    main()
