# %%
import sys
from pathlib import Path

root_path = Path.cwd().parents[1]  
sys.path.append(str(root_path))

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import MODELS_DIR, PROCESS_DIR, PREDICT_DIR
from node_fdm.predictor import NodeFDMPredictor
from node_fdm.architectures.qar.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.qar.model import MODEL_COLS, col_ff, col_n1, col_aoa, col_pitch

split_df = pd.read_csv(PROCESS_DIR / "dataset_split.csv")

processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)
acft= "A320"
print(f"\n🛫 Predicting for aircraft: {acft}")
model_path = MODELS_DIR / f"qar_{acft}"

if not model_path.exists():
    print(f"⚠️  Model not found for {acft}: {model_path}")


predictor = NodeFDMPredictor(
    MODEL_COLS,
    model_path, 
    dt=4.0, 
    device="cuda:0")


data_df = split_df[split_df.aircraft_type == acft]
test_df = data_df[data_df.split == "test"]

output_dir = PREDICT_DIR / acft
output_dir.mkdir(parents=True, exist_ok=True)

for _, row in tqdm(test_df.iloc[10:].iterrows(), total=len(test_df), desc=f"{acft}"):
    flight_path = Path(row.filepath)
    flight_id = flight_path.stem

    f = processor.process_flight(pd.read_parquet(flight_path))

    # Prédiction
    pred_df = predictor.predict_flight(f, add_cols=[col_ff, col_n1, col_aoa, col_pitch])
    break
print(f"✅ Finished predictions for {acft}")



# %%
import matplotlib.pyplot as plt

for col in pred_df.columns:
    plt.plot(pred_df[col])
    plt.plot(f[col[5:]])
    plt.title(col)
    plt.show()
# %%
pred_df
# %%
flight_path
# %%
[el for el in pd.read_parquet(flight_path).columns if "DIST" in el]
# %%
plt.plot(-pred_df["pred_mass_kg"].diff(1) / 4 )
plt.plot(-f["mass_kg"].diff(1)/4)
plt.plot(pred_df[col])
plt.plot(f[col[5:]])
plt.title(col)
plt.show()
# %%
plt.plot(pred_df["pred_mass_kg"])
plt.plot(f["mass_kg"])
plt.title(col)
plt.show()

# %%
