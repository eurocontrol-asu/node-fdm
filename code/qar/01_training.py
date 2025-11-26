# %%
import os
import yaml
import pandas as pd
from pathlib import Path

from node_fdm.ode_trainer import ODETrainer
from node_fdm.architectures.qar.model import E2_COLS, E3_COLS


cfg = yaml.safe_load(open("./config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
models_dir = data_dir / cfg["paths"]["models_dir"]
os.makedirs(models_dir, exist_ok=True)

split_df = pd.read_csv(data_dir / "dataset_split.csv")


model_config = dict(
    architecture_name="qar",
    step=4,
    shift=60,
    lr=1e-3,
    weight_decay=1e-4,
    seq_len=60,
    num_workers=4,
    model_params=[2, 1, 24],
    loading_args=(False, False),
    batch_size=500,
)

acft = "A320"
model_config["model_name"] = "qar_%s" % acft
data_df = split_df[split_df.aircraft_type == acft]

trainer = ODETrainer(
    data_df,
    model_config,
    models_dir,
    num_workers=4,
    train_val_num=(10, 10),
    load_parallel=True,
)

alpha_dict = {col: 1.0 for col in trainer.x_cols}

for col in E2_COLS + E3_COLS:
    alpha_dict[col] = 1.0

trainer.train(
    epochs=1000,
    batch_size=model_config["batch_size"],
    val_batch_size=10000,
    method="euler",
    alpha_dict=alpha_dict,
)

# %%
