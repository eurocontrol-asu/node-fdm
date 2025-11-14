from pathlib import Path

DEFAULT_CPU_COUNT = 35

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
DOWNLOAD_DIR = DATA_DIR / "downloaded_parquet" # FOLDER TO BE CREATED
PREPROCESS_DIR = DATA_DIR / "preprocessed_parquet" # FOLDER TO BE CREATED
PROCESS_DIR = DATA_DIR / "processed_flights" # FOLDER TO BE CREATED
PREDICT_DIR = DATA_DIR / "predicted_flights" # FOLDER TO BE CREATED
BADA_DIR = DATA_DIR / "bada_flights" # FOLDER TO BE CREATED
MODELS_DIR = ROOT_DIR / "models"
FIGURE_DIR = ROOT_DIR / "figures"

ERA5_CACHE_DIR = DATA_DIR / "era5_cache"
ERA5_FEATURES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
]


TYPECODES = [ 
    "A320", 
    'A20N', 
    'A21N',
    'A319',
    'A321',
    'A333', 
    'A359', 
    'AT76', 
    'B38M', 
    'B738',
    'E190'
]

BADA_4_2_DIR = Path("TODO") # COMPLETE WITH YOUR BADA DIRECTORY PATH
