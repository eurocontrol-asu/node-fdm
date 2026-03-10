# %%
"""06 — Predict flight trajectories using trained Neural ODE models."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from tqdm import tqdm

from node_fdm.predictor import NodeFDMPredictor
from node_fdm_data.preprocessing.opensky import flight_processing
from node_fdm_data.processor import FlightProcessor
from node_fdm_data.schemas.opensky import E0_COLS, U_COLS, X_COLS


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    data_dir = Path(cfg["paths"]["data_dir"])
    process_dir = data_dir / cfg["paths"]["process_dir"]
    models_dir = data_dir / cfg["paths"]["models_dir"]
    predict_dir = data_dir / cfg["paths"]["predicted_dir"]
    predict_dir.mkdir(parents=True, exist_ok=True)

    typecodes = cfg["typecodes"]

    split_df = pl.read_csv(process_dir / "dataset_split.csv")

    processor = FlightProcessor(steps=[flight_processing])

    local_models = False

    for acft in typecodes:
        print(f"\n🛫 Predicting for aircraft: {acft}")

        if local_models:
            model_path = models_dir / f"opensky_{acft}"
        else:
            model_path = files("node_fdm.models.pretrained_models.opensky_2025").joinpath(
                f"opensky_{acft}"
            )

        if not model_path.exists():
            print(f"⚠️  Model not found for {acft}: {model_path}")
            continue

        predictor = NodeFDMPredictor(model_path=Path(model_path), device="cuda:0")

        data_df = split_df.filter(pl.col("aircraft_type") == acft)
        test_df = data_df.filter(pl.col("split") == "test")

        output_dir = predict_dir / acft
        output_dir.mkdir(parents=True, exist_ok=True)

        for row in tqdm(test_df.iter_rows(named=True), total=len(test_df), desc=acft):
            flight_path = Path(row["filepath"])
            flight_id = flight_path.stem

            raw = pl.read_parquet(flight_path)
            processed = processor.process(raw).collect()

            # Extract arrays for predictor
            x_init = processed.select(X_COLS).to_numpy().astype(np.float32)
            u_seq = processed.select(U_COLS).to_numpy().astype(np.float32)
            e_seq = processed.select(E0_COLS).to_numpy().astype(np.float32)

            predictions = predictor.predict_flight(x_init, u_seq, e_seq)

            # Build output DataFrame
            pred_df = pl.DataFrame({f"pred_{k}": v for k, v in predictions.items()})
            pred_df.write_parquet(output_dir / f"{flight_id}.parquet")

        print(f"✅ Finished predictions for {acft}")


if __name__ == "__main__":
    main()
# %%
