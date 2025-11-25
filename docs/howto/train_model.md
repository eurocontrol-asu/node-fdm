# Train a Neural ODE model

Prerequisites: processed flights and `dataset_split.csv` produced by `04_weather_spd_process_data.py`.

## Minimal training script (single aircraft)
```python
from pathlib import Path
import pandas as pd
from node_fdm.ode_trainer import ODETrainer
from config import PROCESS_DIR, MODELS_DIR

split_df = pd.read_csv(PROCESS_DIR / "dataset_split.csv")
acft = "A320"
data_df = split_df[split_df.aircraft_type == acft]

model_config = dict(
    architecture_name="opensky_2025",
    model_name=f"opensky_{acft}",
    step=4,           # seconds between samples
    shift=60,         # stride when sliding windows
    seq_len=60,       # 4 minutes per sample
    lr=1e-3,
    weight_decay=1e-4,
    model_params=[3, 2, 48],   # see architecture/model.py
    loading_args=(False, False),
    batch_size=512,
    num_workers=4,
)

trainer = ODETrainer(
    data_df=data_df,
    model_config=model_config,
    model_dir=MODELS_DIR,
    num_workers=model_config["num_workers"],
    load_parallel=True,
)

trainer.train(
    epochs=10,
    batch_size=model_config["batch_size"],
    val_batch_size=10_000,
    method="euler",   # or "rk4"
    alpha_dict=None,  # defaults to 1.0 for each monitored column
)
```

Outputs in `models/opensky_<TYPECODE>/`:
- `meta.json` with architecture name, stats, hyperparameters.
- `trajectory.pt` and `data_ode.pt` checkpoints (modular per layer).
- `training_losses.csv` and `training_curve.png`.

## Tips
- Remove the hard-coded `acft = "A320"` override in `code/opensky/05_training.py` to loop over all types in `config.TYPECODES`.
- Reduce `batch_size` or `num_workers` on small GPUs/CPUs.
- To resume from checkpoints, set `loading_args=(True, True)` in `model_config`.
- Tune `alpha_dict` in `ODETrainer.train` to rebalance losses across monitored columns (`X_COLS + E_COLS`).
