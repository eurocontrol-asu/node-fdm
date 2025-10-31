# %%

import pandas as pd
from traffic.core import Traffic

t = Traffic.from_file("processed_20241001.parquet")
t

# %%

stats: pd.DataFrame = t.summary(
    ["icao24", "callsign", "flight_id", "typecode", "duration"]
).eval()

# %%
stats.groupby("typecode").agg(
    dict(duration="sum", icao24="nunique", callsign="nunique")
)

# %%

flight = t["20241001_B789_02204"]
display(flight)  # type: ignore  # noqa: F821
flight.map_leaflet()

# %%

g = flight.drop(
    columns=[
        "bds05",
        "bds18",
        "bds19",
        "bds21",
        "selected_fms",
        "target_source",
    ]
).drop_duplicates()  # .columns

# %%

g.chart().encode(y="value:Q", color="key:N").transform_fold(
    ["altitude", "selected_mcp"], as_=["key", "value"]
)

# %%

g.chart().encode(y="value:Q", color="key:N").transform_fold(
    ["groundspeed", "IAS", "TAS"], as_=["key", "value"]
)
