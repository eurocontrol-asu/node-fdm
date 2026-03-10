# %%
"""11 — Explore preprocessed dataset interactively."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml
from traffic.core import Traffic


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]

    date = "20241001"
    t = Traffic.from_file(preprocess_dir / f"processed_{date}.parquet")
    print(t)

    # %%
    stats = t.summary(["icao24", "callsign", "flight_id", "typecode", "duration"]).eval()

    # Convert traffic stats to Polars for analysis
    stats_pl = pl.from_pandas(stats)

    print(
        stats_pl.group_by("typecode").agg(
            pl.col("duration").sum(),
            pl.col("icao24").n_unique(),
            pl.col("callsign").n_unique(),
        )
    )


if __name__ == "__main__":
    main()
# %%
