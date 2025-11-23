# %%
import sys
from pathlib import Path

root_path = Path.cwd().parents[1] 
sys.path.append(str(root_path))


import pandas as pd
from pathlib import Path
from node_fdm.ode_trainer import ODETrainer
from config import PROCESS_DIR, MODELS_DIR
from node_fdm.architectures.qar.model import E2_COLS, E3_COLS

split_df = pd.read_csv(PROCESS_DIR / "dataset_split.csv")

model_config = dict(
    
    architecture_name="qar",
    step=4,
    shift=60,
    lr=1e-3,
    weight_decay=1e-4,
    seq_len=60,
    num_workers=4,
    model_params=[3, 2, 48],
    loading_args = (False, False),
    batch_size = 1000
)


acft = "A320"
model_config["model_name"] = "qar_%s" % acft
data_df = split_df[split_df.aircraft_type == acft]

trainer = ODETrainer(
    data_df,
    model_config,
    MODELS_DIR,
    num_workers=4,
    train_val_num=(1000, 500),
    load_parallel=True
)






#%%
alpha_dict = {
    col: 1.0 for col in trainer.x_cols + E2_COLS+ E3_COLS
}

trainer.train(
    epochs=10, 
    batch_size=model_config['batch_size'],
    val_batch_size=10000,
    method="euler",
    alpha_dict=alpha_dict
)


# %%
import matplotlib.pyplot as plt
f = trainer.train_dataset.get_full_flight(0)
f = f[-1]


for col in f.columns:
    print(col, f[col].isna().sum()>0)
    plt.plot(f[col])
    plt.title(col)
    plt.show()
# %%
from node_fdm.architectures.qar.columns import col_ff
f[col_ff].isna().sum()
# %%

# %%
trainer.stats_dict
# %%
