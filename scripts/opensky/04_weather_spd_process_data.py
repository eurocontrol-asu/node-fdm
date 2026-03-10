# %%
"""04 — Attach ERA5 weather data, compute flight parameters, and split."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from node_fdm_data.preprocessing.opensky import flight_processing
from node_fdm_data.split import split_by_icao


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]
    process_dir = data_dir / cfg["paths"]["process_dir"]
    process_dir.mkdir(parents=True, exist_ok=True)

    era5_cache_dir = data_dir / cfg["paths"]["era5_cache_dir"]
    era5_cache_dir.mkdir(parents=True, exist_ok=True)

    era5_features = cfg["era5_features"]  # noqa: F841 — future: pass to ERA5 attachment

    # TODO: attach ERA5 data via fastmeteo.ArcoEra5 before processing

    # --- Process each preprocessed file ---
    for file in sorted(preprocess_dir.iterdir()):
        if not file.suffix == ".parquet":
            continue
        print(f"Processing {file}")
        df = pl.read_parquet(file).lazy()
        processed = flight_processing(df).collect()
        processed.write_parquet(process_dir / file.name)

    # --- Create train/val/test split ---
    split_df = split_by_icao(process_dir)
    split_df.write_csv(process_dir / "dataset_split.csv")
    print(f"✅ Split saved ({len(split_df)} flights)")


if __name__ == "__main__":
    main()
# %%
