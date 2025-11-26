# %%
import os
import yaml
from pathlib import Path
import pandas as pd
from node_fdm.ode_trainer import ODETrainer


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
process_dir = data_dir / cfg["paths"]["process_dir"]

models_dir = data_dir / cfg["paths"]["models_dir"]
os.makedirs(models_dir, exist_ok=True)

typecodes = cfg["typecodes"]


# %%


split_df = pd.read_csv(process_dir / "dataset_split.csv")


model_config = dict(
    architecture_name="opensky_2025",
    step=4,
    shift=60,
    lr=1e-3,
    weight_decay=1e-4,
    seq_len=60,
    num_workers=4,
    model_params=[3, 2, 48],
    loading_args=(False, False),
    batch_size=512,
)

# %%

for acft in typecodes:
    model_config["model_name"] = "opensky_%s" % acft
    data_df = split_df[split_df.aircraft_type == acft]

    trainer = ODETrainer(
        data_df,
        model_config,
        models_dir,
        num_workers=4,
        train_val_num=(5000, 5000),
        load_parallel=False,
    )

    alpha_dict = {col: 1.0 for col in trainer.x_cols}

    n_step_per_epoch = len(trainer.train_dataset) // 512
    coeff = 50 / n_step_per_epoch

    trainer.train(
        epochs=800 * coeff,
        batch_size=model_config["batch_size"],
        val_batch_size=10000,
        method="euler",
        alpha_dict=alpha_dict,
    )
    break

# %%
