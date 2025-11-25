# Run inference on flights (generic)

`NodeFDMPredictor` needs:
- The architecture’s `MODEL_COLS` and (optionally) its `flight_processing` function.
- A trained checkpoint directory containing `meta.json` and layer weights.
- Processed flights that match the architecture’s preprocessing.

Example (adapt the architecture import and paths):
```python
import json
from pathlib import Path
import pandas as pd

from node_fdm.predictor import NodeFDMPredictor
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures import mapping
from config import PROCESS_DIR, MODELS_DIR

# Load meta to recover the architecture name
model_path = MODELS_DIR / "opensky_A320"   # replace with your model folder
meta = json.loads((model_path / "meta.json").read_text())
arch_name = meta["architecture_name"]

# Get architecture-specific modules
_, model_cols, custom_fn = mapping.get_architecture_from_name(arch_name)
custom_processing_fn, _ = custom_fn

# Prepare one processed flight
flight_path = Path(PROCESS_DIR / "A320" / "20241001_A320_00001.parquet")
processor = FlightProcessor(model_cols, custom_processing_fn=custom_processing_fn)
f = processor.process_flight(pd.read_parquet(flight_path))

# Load predictor
predictor = NodeFDMPredictor(
    model_cols=model_cols,
    model_path=model_path,
    dt=meta.get("step", 4.0),
    device="cuda:0",  # set to "cpu" if no GPU
)

# Predict full trajectory
pred_df = predictor.predict_flight(f)
print(pred_df.head())
```

`predict_flight` returns `pred_<column>` for each state/environment output defined by the architecture.

Batch runs: adapt `code/opensky/06_flight_prediction.py` or build a small loop over your test split, reusing `FlightProcessor` + `NodeFDMPredictor`.
