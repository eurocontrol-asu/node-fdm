"""Quick coverage test: for borderline models (~100 icao24 in aircraft.data),
how many actually fly on a given day window per OpenSky flightlist?

Run:
    uv run python scripts/debug/check_model_coverage.py
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import polars as pl
from traffic.data import aircraft, opensky


def main(
    start: str = "2025-09-01",
    days: int = 30,
    lo: int = 80,
    hi: int = 150,
    n_models: int = 10,
) -> None:
    df = pl.from_pandas(
        aircraft.data[["icao24", "model", "typecode", "operator", "operatoricao"]]
    )
    # Commercial filter: keep only aircraft with a known operator (airline)
    commercial = df.filter(
        pl.col("model").is_not_null()
        & (pl.col("model") != "")
        & (
            (pl.col("operator").is_not_null() & (pl.col("operator") != ""))
            | (pl.col("operatoricao").is_not_null() & (pl.col("operatoricao") != ""))
        )
    )
    print(f"total icao24 in db: {len(df)}, with operator: {len(commercial)}")
    counts = (
        commercial.group_by("model")
        .agg(pl.col("icao24").n_unique().alias("n_icao24"))
        .filter((pl.col("n_icao24") >= lo) & (pl.col("n_icao24") <= hi))
        .sort("n_icao24")
    )
    print(f"borderline models in [{lo}, {hi}]: {len(counts)}")
    print(counts.head(n_models))

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=days)
    print(f"\nQuerying flightlist {start_dt.date()} -> {end_dt.date()} ...")

    opensky.trino_client.connect()
    t0 = time.time()
    fl_pd = opensky.flightlist(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
    fl = pl.from_pandas(fl_pd).select("icao24").unique()
    print(f"flightlist: {len(fl)} distinct icao24 in {time.time() - t0:.1f}s")

    sample = counts.head(n_models)
    rows = []
    for row in sample.iter_rows(named=True):
        model = row["model"]
        n_ref = row["n_icao24"]
        ref_icao = commercial.filter(pl.col("model") == model).select("icao24").unique()
        seen = ref_icao.join(fl, on="icao24", how="inner")
        rows.append(
            {
                "model": model,
                "n_in_db": n_ref,
                "n_flying": len(seen),
                "ratio": round(len(seen) / n_ref, 2),
            }
        )
    print("\n=== coverage ===")
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
