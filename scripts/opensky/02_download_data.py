# %%
"""02 — Download ADS-B history data from OpenSky Network."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import yaml
from traffic.data import opensky


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    download_dir = data_dir / cfg["paths"]["download_dir"]
    download_dir.mkdir(parents=True, exist_ok=True)

    sampled_ext = pl.read_csv(data_dir / "aircraft_db.csv")
    icao24_list = sampled_ext["icao24"].to_list()

    # %%
    start_date = datetime(2024, 10, 1)
    end_date = datetime(2025, 10, 15)
    step = timedelta(hours=480)

    current = start_date
    while current < end_date:
        date_str = current.strftime("%Y%m%d")
        next_day = current + timedelta(hours=24)

        # --- History ---
        path = download_dir / f"history_{date_str}.parquet"
        if not path.exists():
            print(f"Downloading history data for {current:%Y-%m-%d}")
            t = opensky.history(current, next_day, icao24=icao24_list)
            assert t is not None
            t.to_parquet(path)

        # --- Flight list ---
        path = download_dir / f"flightlist_{date_str}.parquet"
        if not path.exists():
            print(f"Downloading flight list for {current:%Y-%m-%d}")
            ft = opensky.flightlist(current, next_day, icao24=icao24_list)
            assert ft is not None
            ft.to_parquet(path)

        # --- Extended ---
        path = download_dir / f"extended_{date_str}.parquet"
        if not path.exists():
            print(f"Downloading extended data for {current:%Y-%m-%d}")
            ext = opensky.extended(current, next_day, icao24=icao24_list)
            assert ext is not None
            ext.to_parquet(path)

        current += step


if __name__ == "__main__":
    main()
# %%
