# %%
import sys
from pathlib import Path

root_path = Path.cwd().parents[1]
sys.path.append(str(root_path))

from config import TYPECODES, PROCESS_DIR
import pandas as pd
from pathlib import Path
from node_fdm.architectures.opensky_2025.model import MODEL_COLS
from node_fdm.data.dataset import SeqDataset

split_df = pd.read_csv(PROCESS_DIR / "dataset_split.csv")

for acft in TYPECODES:

    data_df = split_df[split_df.aircraft_type == acft]
    train_files = data_df[data_df.split == "train"].filepath.tolist()
    validation_files = data_df[data_df.split == "val"].filepath.tolist()
    test_files = data_df[data_df.split == "test"].filepath.tolist()

    train_dataset = SeqDataset(
        train_files,
        MODEL_COLS,
        seq_len=60,
        shift=60,
    )
    val_dataset = SeqDataset(
        validation_files,
        MODEL_COLS,
        seq_len=60,
        shift=60,
    )

    test_dataset = SeqDataset(test_files, MODEL_COLS, seq_len=60, shift=60)

    print(
        acft,
        len(train_files),
        round(len(train_dataset) * 60 * 4 / 3600, 2),
        len(validation_files),
        round(len(val_dataset) * 60 * 4 / 3600, 2),
        len(test_files),
        round(len(test_dataset) * 60 * 4 / 3600, 2),
        sep=" & ",
        end=" \\\\\n",
    )
# %%
