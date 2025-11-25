# Run inference on flights

`NodeFDMPredictor` rolls out trajectories with a trained model. You need processed flights (same columns as training) and a checkpoint directory (e.g., `models/opensky_A320/`).

```python
from pathlib import Path
import pandas as pd
from node_fdm.predictor import NodeFDMPredictor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS
from node_fdm.architectures.opensky_2025.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from config import PROCESS_DIR, MODELS_DIR

# Prepare one processed flight
flight_path = Path(PROCESS_DIR / "A320" / "20241001_A320_00001.parquet")
processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)
f = processor.process_flight(pd.read_parquet(flight_path))

# Load model
predictor = NodeFDMPredictor(
    model_cols=MODEL_COLS,
    model_path=MODELS_DIR / "opensky_A320",
    dt=4.0,          # integration step in seconds
    device="cuda:0", # set to "cpu" if no GPU
)

# Predict full trajectory
pred_df = predictor.predict_flight(f)
pred_df.head()
```

`predict_flight` returns a DataFrame with `pred_<column>` for each state/env output (altitude, gamma, TAS, CAS, Mach, VZ, altitude difference).

For batch processing on the test split, reuse `code/opensky/06_flight_prediction.py`; it loads `dataset_split.csv`, iterates over flights, and writes parquet files into `data/predicted_flights/<TYPECODE>/`.
