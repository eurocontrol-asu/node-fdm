# %%
"""01 — Train Neural ODE model on QAR data."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import yaml

# Trigger architecture auto-registration
import node_fdm.architectures.qar  # noqa: F401
from node_fdm.loader import get_train_val_data
from node_fdm.trainer import ODETrainer, TrainingConfig
from node_fdm_data.preprocessing.qar import flight_processing
from node_fdm_data.schemas.qar import DX_COLS, E0_COLS, U_COLS, X_COLS


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    models_dir = data_dir / cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    split_df = pl.read_csv(data_dir / "dataset_split.csv")
    dx_col_names = [col for _, col in DX_COLS]

    acft = "A320"

    config = TrainingConfig(
        architecture_name="qar",
        model_name=f"qar_{acft}",
        model_params=(2, 1, 24),
        step=4.0,
        shift=60,
        lr=1e-3,
        weight_decay=1e-4,
        seq_len=60,
        batch_size=500,
        epochs=1000,
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
        segment_filter_fn=None,
        train_limit=10,
        val_limit=10,
    )

    # Loss weights for all state + extended variables
    trainer = ODETrainer(
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_dir=models_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()
# %%
