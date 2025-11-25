# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parents[1]))

from config import DATA_DIR, TYPECODES as typecodes
from traffic.data import opensky

opensky.trino_client.connect()
fl = opensky.flightlist("2025-10-01", "2025-10-02")
fl

# %%
from traffic.data import aircraft

ext = (
    aircraft.data[["icao24", "registration", "typecode", "age"]]
    .merge(fl, on="icao24", how="inner")
    .assign(airline=lambda df: df.callsign.str.slice(0, 3).str.strip())
)
ext

# %%
ext.groupby(["airline", "typecode"]).agg(
    {"icao24": "nunique"}
).reset_index().sort_values("icao24", ascending=False)
# %%

ext.groupby(["typecode"]).agg({"icao24": "nunique"}).reset_index().sort_values(
    "icao24", ascending=False
).query("typecode in @typecodes", engine="python")
# %%
sampled_ext = (
    ext.query("typecode in @typecodes", engine="python")
    .groupby("typecode", group_keys=False)
    .apply(lambda x: x.sample(n=min(100, len(x)), random_state=42))
    .reset_index(drop=True)
)
# %%
typecodes = [
    "A20N",
    "A21N",
    "A319",
    "A320",
    "A321",
    "A333",
    "A359",
    "AT76",
    "B38M",
    "B738",
    "E190",
]


sampled_ext.query("typecode in @typecodes", engine="python").groupby(
    ["typecode", "airline"]
).agg({"icao24": "nunique"}).reset_index().sort_values("icao24", ascending=False)
pivot_df = (
    sampled_ext.query("typecode in @typecodes", engine="python")
    .query("typecode=='A319'")
    .pivot_table(
        index="typecode",
        columns="airline",
        values="icao24",
        aggfunc="nunique",
        fill_value=0,
    )
)
pivot_df


# %%
aircraft_db = sampled_ext[["icao24", "registration", "typecode", "age", "airline"]]
aircraft_db.to_csv(DATA_DIR / "aircraft_db.csv", index=False)
aircraft_db

# %%
sampled_ext.groupby(["typecode"]).agg(
    {"icao24": "nunique", "airline": "nunique"}
).reset_index().sort_values("icao24", ascending=False).query(
    "typecode in @typecodes", engine="python"
)
