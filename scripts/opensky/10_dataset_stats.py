# %%
"""10 — Compute dataset statistics (flight counts, hours per split)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

from node_fdm.loader import get_train_val_data
from node_fdm_data.preprocessing.opensky import flight_processing, segment_filtering
from node_fdm_data.schemas.opensky import DX_COLS, E0_COLS, U_COLS, X_COLS


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    process_dir = data_dir / cfg["paths"]["process_dir"]

    typecodes = cfg["typecodes"]
    split_df = pl.read_csv(process_dir / "dataset_split.csv")
    dx_col_names = [col for _, col in DX_COLS]

    for acft in typecodes:
        data_df = split_df.filter(pl.col("aircraft_type") == acft)
        train_files = data_df.filter(pl.col("split") == "train")["filepath"].to_list()
        val_files = data_df.filter(pl.col("split") == "val")["filepath"].to_list()
        test_files = data_df.filter(pl.col("split") == "test")["filepath"].to_list()

        train_ds, val_ds = get_train_val_data(
            data_df=data_df,
            x_cols=X_COLS,
            u_cols=U_COLS,
            e_cols=E0_COLS,
            dx_cols=dx_col_names,
            seq_len=60,
            shift=60,
            preprocessing_fn=flight_processing,
            segment_filter_fn=segment_filtering,
            train_limit=None,
            val_limit=None,
        )

        # Approximate test dataset size
        test_count = len(test_files)

        print(
            acft,
            len(train_files),
            round(len(train_ds) * 60 * 4 / 3600, 2),
            len(val_files),
            round(len(val_ds) * 60 * 4 / 3600, 2),
            test_count,
            sep=" & ",
            end=" \\\\\n",
        )


if __name__ == "__main__":
    main()
# %%
