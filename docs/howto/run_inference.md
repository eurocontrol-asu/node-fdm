# 🔮 Run Inference

Once a model is trained, the **`NodeFDMPredictor`** allows you to perform trajectory rollouts by integrating the Neural ODE over time.

---

## 📋 Requirements

1.  **Model Artifacts**: A directory containing `meta.json` and layer weights (`*.pt`).
2.  **Input Data**: Processed flight data as NumPy arrays.

---

## ⚡ Single Flight Inference

```python title="scripts/predict_single.py"
from pathlib import Path

import numpy as np
from node_fdm.predictor import NodeFDMPredictor, ModelMeta

# 1. Load model metadata
model_dir = Path("models/opensky_A320")
meta = ModelMeta.from_json(model_dir / "meta.json")
print(f"Architecture: {meta.architecture_name}")

# 2. Initialize predictor
predictor = NodeFDMPredictor(
    model_path=model_dir,
    device="cpu",  # or "cuda:0"
)

# 3. Run prediction
# x_init: initial state [n_x]
# u_seq: control inputs [seq_len, n_u]
# e_seq: environment inputs [seq_len, n_e]
result = predictor.predict_flight(x_init, u_seq, e_seq)
# result: dict[str, np.ndarray] keyed by state column names
```

---

## 📊 Output Format

The `predict_flight` method returns a dictionary mapping column names to predicted arrays:

```python
result = predictor.predict_flight(x_init, u_seq, e_seq)
# {'distance_m': array([...]),
#  'altitude_ft': array([...]),
#  'gamma_rad': array([...]),
#  'tas_kt': array([...])}
```

!!! tip "Visualization"
    Plot ground truth vs. predicted values to verify accuracy:
    ```python
    import matplotlib.pyplot as plt
    plt.plot(ground_truth["altitude_ft"], label="Ground Truth")
    plt.plot(result["altitude_ft"], label="Predicted")
    plt.legend()
    ```

---

## 🔄 Batch Processing

Initialize the predictor once and iterate over flights:

```python
from pathlib import Path

predictor = NodeFDMPredictor(model_path=Path("models/opensky_A320"), device="cpu")

for flight_path in test_flight_paths:
    # Load and prepare flight data...
    result = predictor.predict_flight(x_init, u_seq, e_seq)
    # Save or analyze result
```

---

## 🚀 Next Steps

* **[Contribute](../contribute/)**: Help improve the framework.
