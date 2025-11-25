# %%
import sys
from pathlib import Path

root_path = Path.cwd().parents[1]
sys.path.append(str(root_path))

import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from config import PROCESS_DIR, PREDICT_DIR, BADA_DIR
from node_fdm.architectures.opensky_2025.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS


processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)
split_df = pd.read_csv(PROCESS_DIR / "dataset_split.csv")
acft = "A319"
data_df = split_df[split_df.aircraft_type == acft]
test_df = data_df[data_df.split == "test"]
output_dir = PREDICT_DIR / acft
for idx, row in test_df.iloc[:1].iterrows():
    flight_path = row.filepath
    f = processor.process_flight(pd.read_parquet(flight_path))
    f2 = pd.read_parquet(output_dir / flight_path.split("/")[-1])
    f = f.join(f2)

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    axes = axes.flatten()

    # -------- Colonne 1 --------
    # Altitude
    axes[0].plot(f.alt_std_m, color="r", label="True")
    axes[0].plot(f.pred_alt_std_m, color="b", label="Pred")
    axes[0].plot(f.alt_sel_m, "--", color="k", lw=0.5, label="Selected")
    axes[0].set_title("Altitude [m]")

    # TAS
    axes[3].plot(f.tas_ms, color="r", label="True")
    axes[3].plot(f.pred_tas_ms, color="b", label="Pred")
    axes[3].set_title("True Airspeed [m/s]")

    # -------- Colonne 2 --------
    # Mach
    axes[1].plot(f.mach, color="r", label="True")
    axes[1].plot(f.pred_mach, color="b", label="Pred")
    axes[1].plot(f.mach_sel, "--", color="k", lw=0.5, label="Selected")
    axes[1].set_title("Mach number")

    # CAS
    axes[4].plot(f.cas_ms, color="r", label="True")
    axes[4].plot(f.pred_cas_ms, color="b", label="Pred")
    axes[4].plot(f.cas_sel_ms, "--", color="k", lw=0.5, label="Selected")
    axes[4].set_title("Calibrated Airspeed [m/s]")

    # -------- Colonne 3 --------
    # Gamma
    axes[2].plot(f.gamma_rad, color="r", label="True")
    axes[2].plot(f.pred_gamma_rad, color="b", label="Pred")
    axes[2].set_title("Flight path angle γ [rad]")

    # Vertical speed
    axes[5].plot(f.vz_ms, color="r", label="True")
    axes[5].plot(f.pred_vz_ms, color="b", label="Pred")
    axes[5].plot(f.vz_sel_ms, "--", color="k", lw=0.5, label="Selected")
    axes[5].set_title("Vertical speed [m/s]")

    # -------- Mise en forme --------
    for ax in axes:
        ax.grid(True, lw=0.3, linestyle="--", alpha=0.7)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
# %%
import matplotlib.pyplot as plt
from pybada_predictor.utils import cas_to_mach, tas_to_cas

acft = "A319"
node_pred_dir = PREDICT_DIR / acft
output_dir = BADA_DIR / acft
for idx, row in test_df.iloc[0:50].iterrows():
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

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    axes = axes.flatten()

    # -------- Colonne 1 --------
    # Altitude
    axes[0].plot(f.alt_std_m, color="r", label="True")
    axes[0].plot(f.pred_alt_std_m, color="b", label="Pred")
    axes[0].plot(f.bada_alt_std_m, color="g", label="Bada")
    axes[0].plot(f.alt_sel_m, "--", color="k", lw=0.5, label="Selected")
    axes[0].set_title("Altitude [m]")

    # TAS
    axes[3].plot(f.tas_ms, color="r", label="True")
    axes[3].plot(f.pred_tas_ms, color="b", label="Pred")
    axes[3].plot(f.bada_tas_ms, color="g", label="Bada")
    axes[3].set_title("True Airspeed [m/s]")

    # -------- Colonne 2 --------
    # Mach
    axes[1].plot(f.mach, color="r", label="True")
    axes[1].plot(f.pred_mach, color="b", label="Pred")
    axes[1].plot(f.bada_mach, color="g", label="Bada")
    axes[1].plot(f.mach_sel, "--", color="k", lw=0.5, label="Selected")
    axes[1].set_title("Mach number")

    # CAS
    axes[4].plot(f.cas_ms, color="r", label="True")
    axes[4].plot(f.pred_cas_ms, color="b", label="Pred")
    axes[4].plot(f.bada_cas_ms, color="g", label="Bada")
    axes[4].plot(f.cas_sel_ms, "--", color="k", lw=0.5, label="Selected")
    axes[4].set_title("Calibrated Airspeed [m/s]")

    # -------- Colonne 3 --------
    # Gamma
    axes[2].plot(f.gamma_rad, color="r", label="True")
    axes[2].plot(f.pred_gamma_rad, color="b", label="Pred")
    axes[2].plot(f.bada_gamma_rad, color="g", label="Bada")
    axes[2].set_title("Flight path angle γ [rad]")

    # Vertical speed
    axes[5].plot(f.vz_ms, color="r", label="True")
    axes[5].plot(f.pred_vz_ms, color="b", label="Pred")
    axes[5].plot(f.bada_vz_ms, color="g", label="Bada")
    axes[5].plot(f.vz_sel_ms, "--", color="k", lw=0.5, label="Selected")
    axes[5].set_title("Vertical speed [m/s]")

    # -------- Mise en forme --------
    for ax in axes:
        ax.grid(True, lw=0.3, linestyle="--", alpha=0.7)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

# %%

from config import DATA_DIR

f.to_parquet(DATA_DIR / "example2.parquet", index=False)
# %%


# %%
import sys
from pathlib import Path

# Détermine le chemin vers la racine du projet
root_path = Path.cwd().parents[0]  # si ton notebook est dans notebooks/
sys.path.append(str(root_path))

import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from config import PROCESS_DIR, PREDICT_DIR, BADA_DIR
from node_fdm.architectures.opensky_2025.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS


processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)
split_df = pd.read_csv(PROCESS_DIR / "dataset_split.csv")
acft = "A319"
data_df = split_df[split_df.aircraft_type == acft]
test_df = data_df[data_df.split == "test"]
output_dir = PREDICT_DIR / acft
for idx, row in test_df.iloc[:1].iterrows():
    flight_path = row.filepath
    f = processor.process_flight(pd.read_parquet(flight_path))
    f2 = pd.read_parquet(output_dir / flight_path.split("/")[-1])
    f = f.join(f2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    # -------- Colonne 1 --------
    # Altitude
    # TAS
    # -------- Colonne 2 --------
    # Mach
    axes[0].plot(f.mach, color="r", label="True")
    axes[0].plot(f.mach_sel, "--", color="k", lw=0.5, label="Selected")
    axes[0].set_title("Mach number")

    # CAS
    axes[1].plot(f.cas_ms, color="r", label="True")
    axes[1].plot(f.cas_sel_ms, "--", color="k", lw=0.5, label="Selected")
    axes[1].set_title("Calibrated Airspeed [m/s]")

    axes[2].plot(f.vz_ms, color="r", label="True")
    axes[2].plot(f.vz_sel_ms, "--", color="k", lw=0.5, label="Selected")
    axes[2].set_title("Vertical speed [m/s]")

    # -------- Mise en forme --------
    for ax in axes:
        ax.grid(True, lw=0.3, linestyle="--", alpha=0.7)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
# %%
