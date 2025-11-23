# %%
import sys
from pathlib import Path

root_path = Path.cwd().parents[1]
sys.path.append(str(root_path))




from config import PROCESS_DIR, BADA_DIR, DEFAULT_CPU_COUNT # PYBADA_DIR
from node_fdm.architectures.opensky_2025.flight_process import flight_processing
from node_fdm.data.flight_processor import FlightProcessor
from node_fdm.architectures.opensky_2025.model import MODEL_COLS

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from config import BADA_4_2_DIR, TYPECODES
from pybada_predictor.aircraft_mapping import BADA_4_2_MAPPING

# sys.path.append(PYBADA_DIR) #ADD YOUR PYBADA if local clone 
from pyBADA.bada4 import Bada4Aircraft


from pybada_predictor.predictor import process_single_flight

import warnings
warnings.filterwarnings("ignore")


split_df = pd.read_csv(PROCESS_DIR / "dataset_split.csv")

processor = FlightProcessor(MODEL_COLS, custom_processing_fn=flight_processing)


for acft in TYPECODES:
    print(f"\n🛫 Predicting for aircraft: {acft}")
    try:
        AC = Bada4Aircraft("4.2", filePath=BADA_4_2_DIR, acName=BADA_4_2_MAPPING[acft])
        data_df = split_df[split_df.aircraft_type == acft]
        test_df = data_df[data_df.split == "test"]

        output_dir = BADA_DIR / acft
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = list(test_df.iloc[:10].itertuples(index=False)) 

        results = Parallel(n_jobs=DEFAULT_CPU_COUNT, backend="loky")(
            delayed(process_single_flight)(row, AC, processor, output_dir)
            for row in tqdm(rows, total=len(rows), desc=f"{acft}")
        )
        print(f"✅ Finished BADA predictions for {acft}")
    except KeyError:
        print(acft, "not avalaible in bada 4.2")

# %%
