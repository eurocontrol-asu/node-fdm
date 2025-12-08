# %%
import os
import yaml
from pathlib import Path

from node_fdm.architectures.opensky_2025.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import sys

sys.path.append(str(Path.cwd().parents[0]))

from pybada_predictor.aircraft_mapping import BADA_4_2_MAPPING
from pybada_predictor.predictor import process_single_flight

from pyBADA.bada4 import Bada4Aircraft

import warnings

cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
process_dir = data_dir / cfg["paths"]["process_dir"]
bada_4_2_dir = data_dir / cfg["paths"]["bada_4_2_dir"]

bada_dir = data_dir / cfg["paths"]["bada_dir"]
os.makedirs(bada_dir, exist_ok=True)

typecodes = cfg["typecodes"]

warnings.filterwarnings("ignore")


split_df = pd.read_csv(process_dir / "dataset_split.csv")

processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)


for acft in typecodes:
    print(f"\n🛫 Predicting for aircraft: {acft}")
    try:
        AC = Bada4Aircraft("4.2", filePath=bada_4_2_dir, acName=BADA_4_2_MAPPING[acft])
        data_df = split_df[split_df.aircraft_type == acft]
        test_df = data_df[data_df.split == "test"]

        output_dir = bada_dir / acft
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = list(test_df.itertuples(index=False))

        results = Parallel(
            n_jobs=cfg["computing"]["default_cpu_count"], backend="loky"
        )(
            delayed(process_single_flight)(row, AC, processor, output_dir)
            for row in tqdm(rows, total=len(rows), desc=f"{acft}")
        )
        print(f"✅ Finished BADA predictions for {acft}")
    except KeyError:
        print(acft, "not avalaible in bada 4.2")

# %%
