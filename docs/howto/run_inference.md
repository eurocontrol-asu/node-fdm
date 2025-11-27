# 🔮 Run Inference

Once a model is trained, the **`NodeFDMPredictor`** allows you to perform trajectory rollouts (simulations) on new data. It reconstructs the flight path by integrating the Neural ODE over time.

---

## 📋 Requirements

To run inference, the predictor needs three things:

1.  **Architecture Definition**: The `MODEL_COLS` and `flight_processing` hooks to prepare data exactly as the model expects.
2.  **Model Artifacts**: A directory containing `meta.json` (for scaling/hyperparams) and the layer weights (`*.pt`).
3.  **Input Data**: A processed parquet file or DataFrame compatible with the architecture.

---

## ⚡ Single Flight Inference

This script demonstrates how to load a trained model and predict a single trajectory.

```python title="scripts/predict_single.py"
import json
import pandas as pd
import yaml
from pathlib import Path

from node_fdm.predictor import NodeFDMPredictor
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures import mapping

# 1. Setup Paths & Config
# Assumes running from repo root
with open("scripts/opensky/config.yaml") as f:
    cfg = yaml.safe_load(f)

paths = cfg["paths"]
process_dir = Path(paths["data_dir"]) / paths["process_dir"]
models_dir = Path(paths["data_dir"]) / paths["models_dir"]

# 2. Identify the Architecture
# We read meta.json first to know WHICH architecture to load
target_model_folder = models_dir / "opensky_2025_A320"  # Adjust folder name
meta_path = target_model_folder / "meta.json"

if not meta_path.exists():
    raise FileNotFoundError(f"Missing meta.json in {target_model_folder}")

meta = json.loads(meta_path.read_text())
arch_name = meta["architecture_name"]
time_step = meta.get("step", 4.0)

# 3. Load Architecture Modules
# Dynamically fetch columns and hooks based on the name found in meta.json
_, model_cols, hooks = mapping.get_architecture_from_name(arch_name)
custom_processing_fn, _ = hooks

# 4. Prepare Input Data
# Load a raw parquet file and apply the architecture's preprocessing
flight_path = process_dir / "A320" / "sample_flight.parquet"
raw_df = pd.read_parquet(flight_path)

processor = FlightProcessor(
    model_cols, 
    custom_processing_fn=custom_processing_fn
)
# Returns a standardized Flight object
processed_flight = processor.process_flight(raw_df)

# 5. Initialize Predictor
predictor = NodeFDMPredictor(
    model_cols=model_cols,
    model_path=target_model_folder,
    dt=time_step,
    device="cuda:0",  # Use "cpu" if GPU is unavailable
)

# 6. Run Prediction
# Integrates the ODE and returns a DataFrame with predictions
pred_df = predictor.predict_flight(processed_flight)

print(f"Prediction complete. Shape: {pred_df.shape}")
print(pred_df.head())
```

---

## 📊 Output Format

The `predict_flight` method returns a DataFrame containing:
* **Original Columns**: All input columns required by the model.
* **Predicted Columns**: Columns prefixed with `pred_` (e.g., `pred_alt`, `pred_tas`, `pred_lat`, `pred_lon`).

!!! tip "Visualization"
    You can immediately plot `alt` vs `pred_alt` to verify the model's accuracy on this specific flight.

---

## 🔄 Batch Processing

For processing an entire test set, it is inefficient to re-initialize the `NodeFDMPredictor` loop inside the loop. Instead, initialize it once and iterate over your file list.

```python
# Load Split File
split_df = pd.read_csv(process_dir / "dataset_split.csv")
test_files = split_df[split_df.split == "test"].filepath.tolist()

results = []

for rel_path in test_files:
    # Construct full path
    fpath = process_dir / rel_path
    
    # Process
    raw_df = pd.read_parquet(fpath)
    flight_obj = processor.process_flight(raw_df)
    
    # Predict (Reuse the predictor instance!)
    pred = predictor.predict_flight(flight_obj)
    
    # Save or Analyze
    output_path = fpath.with_name(f"{fpath.stem}_pred.parquet")
    pred.to_parquet(output_path)
```