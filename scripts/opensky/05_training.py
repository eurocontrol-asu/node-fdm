# %%
"""05 — Train Neural ODE models for each aircraft type."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

# Trigger architecture auto-registration
import node_fdm.architectures.opensky  # noqa: F401
from node_fdm.loader import get_train_val_data
from node_fdm.trainer import ODETrainer, TrainingConfig
from node_fdm_data.preprocessing.opensky import flight_processing, segment_filtering
from node_fdm_data.schemas.opensky import DX_COLS, E0_COLS, U_COLS, X_COLS


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    process_dir = data_dir / cfg["paths"]["process_dir"]
    models_dir = data_dir / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    typecodes = cfg["typecodes"]

    split_df = pl.read_csv(process_dir / "dataset_split.csv")
    dx_col_names = [col for _, col in DX_COLS]

    # %%
    for acft in typecodes:
        config = TrainingConfig(
            architecture_name="opensky_2025",
            model_name=f"opensky_{acft}",
            model_params=(3, 2, 48),
            step=4.0,
            shift=60,
            lr=1e-3,
            weight_decay=1e-4,
            seq_len=60,
            batch_size=512,
            epochs=800,
            method="euler",
            num_workers=4,
        )

        data_df = split_df.filter(pl.col("aircraft_type") == acft)

        train_ds, val_ds = get_train_val_data(
            data_df=data_df,
            x_cols=X_COLS,
            u_cols=U_COLS,
            e_cols=E0_COLS,
            dx_cols=dx_col_names,
            seq_len=config.seq_len,
            shift=config.shift,
            preprocessing_fn=flight_processing,
            segment_filter_fn=segment_filtering,
            train_limit=5000,
            val_limit=5000,
        )

        trainer = ODETrainer(
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_dir=models_dir,
        )

        trainer.train()
        break  # Train one model for testing


if __name__ == "__main__":
    main()
# %%
