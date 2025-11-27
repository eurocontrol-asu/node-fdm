# Train a Neural ODE model (generic)

Prerequisites:

- A processed dataset compatible with the target architecture (columns and preprocessing must match).
- A file list DataFrame with at least `filepath` and `split` (`train` / `val` / `test`).

## Minimal training script
Set the architecture name you want to use (e.g., `opensky_2025`, `qar`, or your custom one registered in `architectures.mapping.valid_names`).
```python
import pandas as pd
from node_fdm.ode_trainer import ODETrainer
import yaml
from pathlib import Path

cfg = yaml.safe_load(open("config.yaml"))  # run inside scripts/opensky or scripts/qar
paths = cfg["paths"]
process_dir = Path(paths["data_dir"]) / paths["process_dir"]
models_dir = Path(paths["data_dir"]) / paths["models_dir"]

acft = "A320"  # or any grouping key you use
arch = "opensky_2025"  # replace with your architecture

split_df = pd.read_csv(process_dir / "dataset_split.csv")
data_df = split_df[split_df.aircraft_type == acft]

model_config = dict(
    architecture_name=arch,
    model_name=f"{arch}_{acft}",
    step=4,           # sample period (seconds)
    shift=60,         # stride for sliding windows
    seq_len=60,       # window length (steps)
    lr=1e-3,
    weight_decay=1e-4,
    model_params=[3, 2, 48],   # architecture-specific (see model.py)
    loading_args=(False, False),  # (load, load_loss)
    batch_size=512,
    num_workers=4,
)

trainer = ODETrainer(
    data_df=data_df,
    model_config=model_config,
    model_dir=models_dir,
    num_workers=model_config["num_workers"],
    load_parallel=True,
)

trainer.train(
    epochs=10,
    batch_size=model_config["batch_size"],
    val_batch_size=10_000,
    method="euler",   # or "rk4"
    alpha_dict=None,  # defaults to 1.0 per monitored column
)
```

Outputs in `models/<arch>_<group>/`:

- `meta.json` with architecture name, stats, hyperparameters.
- Layer checkpoints (e.g., `trajectory.pt`, `data_ode.pt`).
- `training_losses.csv` and `training_curve.png`.

## Tips

- For multiple architectures, run the loop per `architecture_name` and per data subset that matches its preprocessing/columns.
- `model_params` is defined by each architecture’s `model.py`; keep it in sync with your custom architecture.
- To resume training, set `loading_args=(True, True)`.
- Use `alpha_dict` in `trainer.train` to rebalance losses across monitored columns (`X_COLS + E_COLS`).
