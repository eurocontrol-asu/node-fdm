# %%
import os
import yaml
from pathlib import Path
from fastmeteo.source import ArcoEra5
from preprocessing.meteo_and_parameters import process_files
from preprocessing.split import make_global_split_csv

from node_fdm.architectures.opensky_2025.flight_process import selected_param_config


cfg = yaml.safe_load(open("config.yaml"))

data_dir = Path(cfg["paths"]["data_dir"])
preprocess_dir = data_dir / cfg["paths"]["preprocess_dir"]
process_dir = data_dir / cfg["paths"]["process_dir"]
os.makedirs(process_dir, exist_ok=True)

era5_cache_dir = data_dir / cfg["paths"]["era5_cache_dir"]
os.makedirs(era5_cache_dir, exist_ok=True)

era5_features = cfg["era5_features"]


os.makedirs(process_dir, exist_ok=True)


arco_grid = ArcoEra5(local_store=era5_cache_dir, features=era5_features)

for file in os.listdir(preprocess_dir):
    file_path = preprocess_dir / file
    print(file_path)
    process_files(arco_grid, file_path, process_dir, selected_param_config)

make_global_split_csv(process_dir)

# %%
