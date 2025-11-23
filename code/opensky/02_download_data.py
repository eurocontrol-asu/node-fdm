# %%
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parents[1]))

from config import DATA_DIR, DOWNLOAD_DIR
import pandas as pd
from traffic.data import opensky

sampled_ext = pd.read_csv(DATA_DIR / "aircraft_db.csv")
sampled_ext
# %%

for start in pd.date_range("2024-10-01", "2025-10-15", freq="480h"):
    path = Path(f"history_{start.strftime('%Y%m%d')}.parquet")

    if not path.exists():
        print(f"Downloading history data for {start.strftime('%Y-%m-%d')}")

        t = opensky.history(
            start,
            start + pd.Timedelta("24h"),
            icao24=sampled_ext.icao24.tolist(),
        )

        assert t is not None

        t.to_parquet(DOWNLOAD_DIR / f"history_{start.strftime('%Y%m%d')}.parquet")

    path = Path(f"flightlist_{start.strftime('%Y%m%d')}.parquet")
    if not path.exists():
        print(f"Downloading flight list for {start.strftime('%Y-%m-%d')}")

        ft = opensky.flightlist(
            start,
            start + pd.Timedelta("24h"),
            icao24=sampled_ext.icao24.tolist(),
        )

        assert ft is not None

        ft.to_parquet(DOWNLOAD_DIR / f"flightlist_{start.strftime('%Y%m%d')}.parquet")

    path = Path(f"extended_{start.strftime('%Y%m%d')}.parquet")

    if not path.exists():
        print(f"Downloading extended data for {start.strftime('%Y-%m-%d')}")

        ext = opensky.extended(
            start,
            start + pd.Timedelta("24h"),
            icao24=sampled_ext.icao24.tolist(),
        )

        assert ext is not None

        ext.to_parquet(DOWNLOAD_DIR / f"extended_{start.strftime('%Y%m%d')}.parquet")

# %%
