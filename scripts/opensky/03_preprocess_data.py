# %%
# ls history_*.parquet | parallel -j 20 uv run 03_preprocess_data.py {}
import os
import yaml
from pathlib import Path

import click

import pandas as pd
from traffic.core import Flight, Traffic
from traffic.data import airports


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
download_dir = data_dir / cfg["paths"]["download_dir"]
preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]

os.makedirs(preprocess_dir, exist_ok=True)


class ExtendedDecoder:
    def __init__(self, rawdata: pd.DataFrame | None = None) -> None:
        self.rawdata = rawdata

    def __call__(self, flight: Flight) -> None | Flight:
        if flight.duration < pd.Timedelta("4min"):
            return None
        decoded = flight.query_ehs(self.rawdata)
        if "bds40" not in decoded.data.columns:
            return None
        if "bds50" not in decoded.data.columns:
            return None
        if "bds60" not in decoded.data.columns:
            return None
        exp60 = decoded.data["bds60"].apply(pd.Series)
        exp50 = (
            decoded.data["bds50"]
            .apply(pd.Series)
            .drop(columns=["groundspeed", "track"])
        )
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
        decoded = Flight(
            result.drop(
                columns=[
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
                ],
                errors="ignore",
            ).convert_dtypes(dtype_backend="pyarrow")
        )
        return decoded


class DistanceADEPADES:
    def __init__(self, flights: pd.DataFrame) -> None:
        self.flights = flights

    def __call__(self, flight: Flight) -> Flight:
        candidate = self.flights.query(
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
    def __init__(self, date: str) -> None:
        self.date = date

    def format(s, self: Flight, idx: int) -> str:
        """This method mimics the behaviour of the str.format method.

        It will be fed with .format(self=, idx=) so we have to keep the
        second argument as self, which is WEIRD.
        """
        # print(
        #     "Processed flight:",
        #     self.callsign,
        #     f"{s.date}_{self.typecode}_{idx:05}",
        # )
        return f"{s.date}_{self.typecode}_{idx:05}"


pd.set_option("future.no_silent_downcasting", True)


@click.command()
@click.argument("history", type=click.Path(exists=True, path_type=Path))
@click.option("--workers", type=int, default=1)
def main(history: Path, workers: int) -> None:
    date = history.stem.split("_")[1]
    extended = Path(download_dir / f"extended_{date}.parquet")
    flightlist = Path(download_dir / f"flightlist_{date}.parquet")
    processed = Path(preprocess_dir / f"processed_{date}.parquet")

    if processed.exists():
        print(f"{processed} already exists, skipping processing.")
        return

    t = Traffic.from_file(history)
    print(f"Processing {history} with {len(t)} flights.")
    assert t is not None

    ext = pd.read_parquet(extended)
    fl = pd.read_parquet(flightlist)
    aircraft = pd.read_csv("aircraft_db.csv")

    icao24 = (  # noqa: F841
        ext.groupby(["icao24"])
        .count()
        .query("rawmsg > 50")
        .reset_index()
        .icao24.to_list()
    )
    t_ext = Traffic(t.data.query("icao24 in @icao24"))
    assert t_ext is not None

    t_filtered = (
        t_ext.iterate_lazy(iterate_kw=dict(by="1h"))
        .pipe(ExtendedDecoder(ext))
        .filter()
        .resample("1s", how=None)
        .pipe(DistanceADEPADES(fl))
        .filter("aggressive")
        .resample("4s")
        .drop(
            columns=["track_unwrapped", "heading_unwrapped", "lastcontact"],
            errors="ignore",
        )
        .merge(aircraft[["icao24", "registration", "typecode"]])
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
