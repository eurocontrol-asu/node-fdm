# %%
import os
import yaml
from pathlib import Path

import pandas as pd
from traffic.data import opensky


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
download_dir = data_dir / cfg["paths"]["download_dir"]
os.makedirs(download_dir, exist_ok=True)

sampled_ext = pd.read_csv(data_dir / "aircraft_db.csv")
sampled_ext
# %%

for start in pd.date_range("2024-10-01", "2024-10-02", freq="480h"):  # "2025-10-15"
    path = Path(f"history_{start.strftime('%Y%m%d')}.parquet")

    if not path.exists():
        print(f"Downloading history data for {start.strftime('%Y-%m-%d')}")

        t = opensky.history(
            start,
            start + pd.Timedelta("10min"),  # 24h
            icao24=sampled_ext.icao24.tolist(),
        )

        assert t is not None

        t.to_parquet(download_dir / f"history_{start.strftime('%Y%m%d')}.parquet")

    path = Path(f"flightlist_{start.strftime('%Y%m%d')}.parquet")
    if not path.exists():
        print(f"Downloading flight list for {start.strftime('%Y-%m-%d')}")

        ft = opensky.flightlist(
            start,
            start + pd.Timedelta("10min"),  # 24h
            icao24=sampled_ext.icao24.tolist(),
        )

        assert ft is not None

        ft.to_parquet(download_dir / f"flightlist_{start.strftime('%Y%m%d')}.parquet")

    path = Path(f"extended_{start.strftime('%Y%m%d')}.parquet")

    if not path.exists():
        print(f"Downloading extended data for {start.strftime('%Y-%m-%d')}")

        ext = opensky.extended(
            start,
            start + pd.Timedelta("10min"),  # #24h
            icao24=sampled_ext.icao24.tolist(),
        )

        assert ext is not None

        ext.to_parquet(download_dir / f"extended_{start.strftime('%Y%m%d')}.parquet")

# %%
