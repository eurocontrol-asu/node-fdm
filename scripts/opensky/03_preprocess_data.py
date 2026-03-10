# %%
# ls history_*.parquet | parallel -j 20 uv run 03_preprocess_data.py {}
"""03 — Preprocess raw ADS-B data (EHS decoding, filtering, resampling).

This script uses the ``traffic`` library, which operates on pandas
DataFrames internally.  All pandas usage is isolated inside
traffic-specific helper classes; no ``import pandas`` appears at
module level.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import click
import polars as pl
import yaml
from traffic.core import Flight, Traffic
from traffic.data import airports

cfg = yaml.safe_load(Path("config.yaml").read_text())

data_dir = Path(cfg["paths"]["data_dir"])
download_dir = data_dir / cfg["paths"]["download_dir"]
preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]
preprocess_dir.mkdir(parents=True, exist_ok=True)


class ExtendedDecoder:
    """Decode EHS BDS registers from raw Mode-S data.

    Uses traffic's pandas-based Flight API internally.
    """

    def __init__(self, rawdata: object = None) -> None:
        self.rawdata = rawdata

    def __call__(self, flight: Flight) -> Flight | None:
        if flight.duration < timedelta(minutes=4):
            return None
        decoded = flight.query_ehs(self.rawdata)
        for bds in ("bds40", "bds50", "bds60"):
            if bds not in decoded.data.columns:
                return None

        # Expand BDS columns (traffic returns pandas DataFrames)
        import pandas as pd  # traffic interop only

        exp60 = decoded.data["bds60"].apply(pd.Series)
        exp50 = decoded.data["bds50"].apply(pd.Series).drop(columns=["groundspeed", "track"])
        exp40 = decoded.data["bds40"].apply(pd.Series)
        result = pd.concat(
            [
                decoded.data.drop(columns=["bds40", "bds50", "bds60"]),
                exp40,
                exp50,
                exp60,
            ],
            axis=1,
        )
        drop_cols = [
            "metadata",
            "squawk",
            "bds20",
            "bds17",
            "bds18",
            "bds19",
            "bds21",
            "bds45",
            "bds10",
            "bds44",
            "bds30",
            0,
            "bds",
            "serials",
            "alert",
            "spi",
            "geoaltitude",
            "vrate_barometric",
            "vrate_inertial",
            "barometric_setting",
            "selected_fms",
            "target_source",
            "df",
            "frame",
            "onground",
        ]
        decoded = Flight(
            result.drop(columns=drop_cols, errors="ignore").convert_dtypes(dtype_backend="pyarrow")
        )
        return decoded


class DistanceADEPADES:
    """Compute distances to departure and arrival airports."""

    def __init__(self, flights: pl.DataFrame) -> None:
        # traffic's distance() needs pandas, convert for lookup
        import pandas as pd  # traffic interop only

        self._flights_pd = (
            flights.to_pandas() if isinstance(flights, pl.DataFrame) else pd.DataFrame(flights)
        )

    def __call__(self, flight: Flight) -> Flight:
        candidate = self._flights_pd.query(
            "icao24 == @flight.icao24 and "
            "@flight.start < lastseen and @flight.stop > firstseen and "
            "departure.notnull() and arrival.notnull()"
        )
        if candidate.shape[0] == 0:
            return flight
        adep = candidate.iloc[0].departure
        ades = candidate.iloc[0].arrival
        try:
            flight = flight.distance(airports[adep], column_name="adep_dist")
            flight = flight.distance(airports[ades], column_name="ades_dist")
        except Exception:
            pass
        return flight


class FlightIdNamer:
    """Generate deterministic flight IDs from date + typecode."""

    def __init__(self, date: str) -> None:
        self.date = date

    def format(s, self: Flight, idx: int) -> str:  # noqa: N805
        """Mimic ``str.format`` — fed with .format(self=, idx=)."""
        return f"{s.date}_{self.typecode}_{idx:05}"


@click.command()
@click.argument("history", type=click.Path(exists=True, path_type=Path))
@click.option("--workers", type=int, default=1)
def main(history: Path, workers: int) -> None:
    date = history.stem.split("_")[1]
    extended = download_dir / f"extended_{date}.parquet"
    flightlist = download_dir / f"flightlist_{date}.parquet"
    processed = preprocess_dir / f"processed_{date}.parquet"

    if processed.exists():
        print(f"{processed} already exists, skipping processing.")
        return

    t = Traffic.from_file(history)
    print(f"Processing {history} with {len(t)} flights.")
    assert t is not None

    # Read supporting data with Polars, convert for traffic interop
    ext_pl = pl.read_parquet(extended)
    fl_pl = pl.read_parquet(flightlist)
    aircraft_pl = pl.read_csv(data_dir / "aircraft_db.csv")

    # Filter by extended message count (traffic needs pandas)

    ext_pd = ext_pl.to_pandas()
    t_ext = Traffic(
        t.data.query(
            "icao24 in @icao24",
            local_dict={
                "icao24": ext_pd.groupby(["icao24"])
                .count()
                .query("rawmsg > 50")
                .reset_index()
                .icao24.to_list()
            },
        )
    )
    assert t_ext is not None

    aircraft_pd = aircraft_pl.select("icao24", "registration", "typecode").to_pandas()

    t_filtered = (
        t_ext.iterate_lazy(iterate_kw={"by": "1h"})
        .pipe(ExtendedDecoder(ext_pd))
        .filter()
        .resample("1s", how=None)
        .pipe(DistanceADEPADES(fl_pl))
        .filter("aggressive")
        .resample("4s")
        .drop(
            columns=["track_unwrapped", "heading_unwrapped", "lastcontact"],
            errors="ignore",
        )
        .merge(aircraft_pd)
        .assign_id(FlightIdNamer(date))
        .eval(desc="Processing", max_workers=workers)
    )

    try:
        t_filtered = t_filtered.drop_duplicates()
    except Exception:
        pass

    t_filtered.to_parquet(processed)


if __name__ == "__main__":
    main()
# %%
