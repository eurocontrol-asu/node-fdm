# %%
"""01 — Build aircraft database from OpenSky Network."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml
from traffic.data import aircraft, opensky


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    typecodes = cfg["typecodes"]

    # --- Query OpenSky for one day of flights ---
    opensky.trino_client.connect()
    fl_pd = opensky.flightlist("2025-10-01", "2025-10-02")

    # Convert traffic (pandas) outputs to Polars immediately
    fl = pl.from_pandas(fl_pd)
    acft_db = pl.from_pandas(aircraft.data[["icao24", "registration", "typecode", "age"]])

    # %%
    ext = acft_db.join(fl, on="icao24", how="inner").with_columns(
        airline=pl.col("callsign").str.slice(0, 3).str.strip_chars()
    )

    # --- Stats ---
    print(
        ext.group_by("airline", "typecode")
        .agg(pl.col("icao24").n_unique())
        .sort("icao24", descending=True)
    )

    print(
        ext.group_by("typecode")
        .agg(pl.col("icao24").n_unique())
        .sort("icao24", descending=True)
        .filter(pl.col("typecode").is_in(typecodes))
    )

    # %%
    # --- Sample up to 100 flights per type ---
    sampled_ext = (
        ext.filter(pl.col("typecode").is_in(typecodes))
        .group_by("typecode")
        .map_groups(lambda g: g.sample(n=min(100, len(g)), seed=42))
    )

    # %%
    aircraft_db = sampled_ext.select("icao24", "registration", "typecode", "age", "airline")
    aircraft_db.write_csv(data_dir / "aircraft_db.csv")
    print(f"✅ Saved aircraft_db.csv ({len(aircraft_db)} rows)")


if __name__ == "__main__":
    main()
# %%
