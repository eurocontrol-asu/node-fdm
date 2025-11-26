# %%

import os
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from importlib.resources import files


from node_fdm.predictor import NodeFDMPredictor
from node_fdm.architectures.qar.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.qar.model import (
    MODEL_COLS,
    col_ff,
    col_n1,
    col_aoa,
    col_pitch,
)


cfg = yaml.safe_load(open("./config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
models_dir = data_dir / cfg["paths"]["models_dir"]
predicted_dir = data_dir / cfg["paths"]["predicted_dir"]
os.makedirs(predicted_dir, exist_ok=True)

split_df = pd.read_csv(data_dir / "dataset_split.csv")

acft = "A320"
local = False

processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)

print(f"\n🛫 Predicting for aircraft: {acft}")
if local:
    model_path = models_dir / f"qar_{acft}"
else:
    model_path = files("models").joinpath(f"qar_{acft}")


if not model_path.exists():
    print(f"⚠️  Model not found for {acft}: {model_path}")


predictor = NodeFDMPredictor(MODEL_COLS, model_path, dt=4.0, device="cuda:0")


data_df = split_df[split_df.aircraft_type == acft]
test_df = data_df[data_df.split == "test"]

output_dir = predicted_dir / acft
output_dir.mkdir(parents=True, exist_ok=True)

for _, row in tqdm(test_df.iloc[10:].iterrows(), total=len(test_df), desc=f"{acft}"):
    flight_path = Path(row.filepath)
    flight_id = flight_path.stem

    f = processor.process_flight(pd.read_parquet(flight_path))

    # Prédiction
    pred_df = predictor.predict_flight(f, add_cols=[col_ff, col_n1, col_aoa, col_pitch])
    break
print(f"✅ Finished predictions for {acft}")
