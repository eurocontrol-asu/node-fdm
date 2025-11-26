# %%
import os
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from importlib.resources import files

from node_fdm.predictor import NodeFDMPredictor
from node_fdm.architectures.opensky_2025.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS  # , DX_COLS2
from node_fdm.architectures.opensky_2025.columns import col_cas


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
process_dir = data_dir / cfg["paths"]["process_dir"]
models_dir = data_dir / cfg["paths"]["models_dir"]
predict_dir = data_dir / cfg["paths"]["predicted_dir"]
os.makedirs(predict_dir, exist_ok=True)

typecodes = cfg["typecodes"]

split_df = pd.read_csv(process_dir / "dataset_split.csv")

processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)

local_models = False


for acft in typecodes:
    print(f"\n🛫 Predicting for aircraft: {acft}")

    if local_models:
        model_path = models_dir / f"opensky_{acft}"
    else:
        model_path = files("models.opensky_2025").joinpath(f"opensky_{acft}")

    if not model_path.exists():
        print(f"⚠️  Model not found for {acft}: {model_path}")
        continue

    predictor = NodeFDMPredictor(MODEL_COLS, model_path, dt=4.0, device="cuda:0")

    data_df = split_df[split_df.aircraft_type == acft]
    test_df = data_df[data_df.split == "test"]

    output_dir = predict_dir / acft
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{acft}"):
        flight_path = Path(row.filepath)
        flight_id = flight_path.stem

        f = processor.process_flight(pd.read_parquet(flight_path))

        out_path = output_dir / f"{flight_id}.parquet"

        pred_df = predictor.predict_flight(f, add_cols=[col_cas])
        pred_df.to_parquet(out_path, index=False)

    print(f"✅ Finished predictions for {acft}")

# %%
