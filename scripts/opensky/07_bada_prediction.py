# %%
"""07 — Run BADA 4.2 baseline predictions."""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl
import yaml
from joblib import Parallel, delayed
from pyBADA.bada4 import Bada4Aircraft
from tqdm import tqdm

from node_fdm_bada.aircraft_mapping import get_bada_identifier
from node_fdm_bada.predictor import process_single_flight
from node_fdm_data.preprocessing.opensky import flight_processing
from node_fdm_data.processor import FlightProcessor


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    process_dir = data_dir / cfg["paths"]["process_dir"]
    bada_4_2_dir = data_dir / cfg["bada"]["bada_4_2_dir"]
    bada_dir = data_dir / cfg["paths"]["bada_dir"]
    bada_dir.mkdir(parents=True, exist_ok=True)

    typecodes = cfg["typecodes"]
    warnings.filterwarnings("ignore")

    split_df = pl.read_csv(process_dir / "dataset_split.csv")
    processor = FlightProcessor(steps=[flight_processing])

    for acft in typecodes:
        print(f"\n🛫 Predicting for aircraft: {acft}")
        try:
            bada_name = get_bada_identifier(acft)
            AC = Bada4Aircraft("4.2", filePath=bada_4_2_dir, acName=bada_name)
            data_df = split_df.filter(pl.col("aircraft_type") == acft)
            test_df = data_df.filter(pl.col("split") == "test")

            output_dir = bada_dir / acft
            output_dir.mkdir(parents=True, exist_ok=True)

            filepaths = test_df["filepath"].to_list()

            _ = Parallel(n_jobs=cfg["computing"]["default_cpu_count"], backend="loky")(
                delayed(process_single_flight)(fp, AC, processor, output_dir)
                for fp in tqdm(filepaths, desc=acft)
            )
            print(f"✅ Finished BADA predictions for {acft}")
        except KeyError:
            print(f"{acft} not available in BADA 4.2")


if __name__ == "__main__":
    main()
# %%
