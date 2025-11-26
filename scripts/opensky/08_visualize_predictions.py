# %%
import os
import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from node_fdm.architectures.opensky_2025.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS
from ..pybada_predictor.utils import cas_to_mach, tas_to_cas


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
process_dir = data_dir / cfg["paths"]["process_dir"]
predict_dir = data_dir / cfg["paths"]["predicted_dir"]
bada_dir = data_dir / cfg["paths"]["bada_dir"]


processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)
split_df = pd.read_csv(process_dir / "dataset_split.csv")
acft = "A320"

data_df = split_df[split_df.aircraft_type == acft]
test_df = data_df[data_df.split == "test"]
node_pred_dir = predict_dir / acft
output_dir = bada_dir / acft


for idx, row in test_df.iterrows():
    flight_path = row.filepath
    print(flight_path)
    if not os.path.exists(output_dir / flight_path.split("/")[-1]):
        continue
    f = pd.read_parquet(flight_path)
    f0 = processor.process_flight(f)
    f2 = pd.read_parquet(node_pred_dir / flight_path.split("/")[-1])
    f = f.join(f2)
    f2 = pd.read_parquet(output_dir / flight_path.split("/")[-1])
    f = f.join(f2)
    f["bada_cas_ms"] = tas_to_cas(f["bada_tas_ms"], f["bada_alt_std_m"], f["temp_k"])
    f["bada_mach"] = cas_to_mach(f["bada_cas_ms"], f["bada_alt_std_m"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()

    # -------- Colonne 1 --------
    # Altitude
    axes[0].plot(f.alt_std_m, color="r", label="True")
    axes[0].plot(f.pred_alt_std_m, color="b", label="Pred")
    axes[0].plot(f.bada_alt_std_m, color="g", label="Bada")
    axes[0].plot(f.alt_sel_m, "--", color="k", lw=0.5, label="Selected")
    axes[0].set_title("Altitude [m]")

    # TAS
    axes[1].plot(f.tas_ms, color="r", label="True")
    axes[1].plot(f.pred_tas_ms, color="b", label="Pred")
    axes[1].plot(f.bada_tas_ms, color="g", label="Bada")
    axes[1].set_title("True Airspeed [m/s]")

    # Gamma
    axes[2].plot(f.gamma_rad, color="r", label="True")
    axes[2].plot(f.pred_gamma_rad, color="b", label="Pred")
    axes[2].plot(f.bada_gamma_rad, color="g", label="Bada")
    axes[2].set_title("Flight path angle γ [rad]")

    # -------- Mise en forme --------
    for ax in axes:
        ax.grid(True, lw=0.3, linestyle="--", alpha=0.7)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    f.to_parquet(data_dir / "example.parquet", index=False)
    break

# %%
