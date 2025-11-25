# %%
# ls history_*.parquet | parallel -j 20 uv run 03_preprocess_data.py {}
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parents[1]))  # ajoute node-fdm/

from config import PREPROCESS_DIR, PROCESS_DIR, ERA5_CACHE_DIR, ERA5_FEATURES


import os
from pathlib import Path
from preprocessing.meteo_and_parameters import process_files
from preprocessing.split import make_global_split_csv

from node_fdm.architectures.opensky_2025.flight_process import selected_param_config

from fastmeteo.source import ArcoEra5


arco_grid = ArcoEra5(local_store=ERA5_CACHE_DIR, features=ERA5_FEATURES)

for file in os.listdir(PREPROCESS_DIR):
    file_path = PREPROCESS_DIR / file
    print(file_path)
    process_files(arco_grid, file_path, PROCESS_DIR, selected_param_config)
    break

make_global_split_csv(PROCESS_DIR)
