# %%
import sys
from pathlib import Path

root_path = Path.cwd().parents[1]  
sys.path.append(str(root_path))

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from config import MODELS_DIR, PROCESS_DIR, PREDICT_DIR, TYPECODES
from node_fdm.predictor import NodeFDMPredictor

from node_fdm.architectures.opensky_2025.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS#, DX_COLS2

split_df = pd.read_csv(PROCESS_DIR / "dataset_split.csv")

processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)



# Boucle sur chaque type d'avion
for acft in TYPECODES:
    print(f"\n🛫 Predicting for aircraft: {acft}")
    model_path = MODELS_DIR / f"opensky_{acft}"

    if not model_path.exists():
        print(f"⚠️  Model not found for {acft}: {model_path}")
        continue

    predictor = NodeFDMPredictor(MODEL_COLS, model_path, dt=4.0, device="cuda:0")

    # Sélectionne les vols test
    data_df = split_df[split_df.aircraft_type == acft]
    test_df = data_df[data_df.split == "test"]

    # Crée le dossier de sortie
    output_dir = PREDICT_DIR / acft
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{acft}"):
        flight_path = Path(row.filepath)
        flight_id = flight_path.stem
        print(flight_path)
        # Prépare les données
        f = processor.process_flight(pd.read_parquet(flight_path))

        # Fichier de sortie
        out_path = output_dir / f"{flight_id}.parquet"

        # Prédiction
        pred_df = predictor.predict_flight(f)
        pred_df.to_parquet(out_path, index=False)

    print(f"✅ Finished predictions for {acft}")

