# Configure project paths and options

All runtime settings are now defined per pipeline:
- `code/opensky/config.yaml` — OpenSky 2025 pipeline
- `code/qar/config.yaml` — QAR pipeline
- (Template) `../configs/opensky_2025.yaml` — copy/adapt for new setups

Key fields (OpenSky example):
```yaml
paths:
  data_dir: "/path/to/data"
  download_dir: "downloaded_parquet"
  preprocess_dir: "preprocessed_parquet"
  process_dir: "processed_flights"
  predicted_dir: "predicted_flights"
  bada_dir: "bada_flights"
  models_dir: "models"
  figure_dir: "figures"
  era5_cache_dir: "era5_cache"

era5_features:
  - u_component_of_wind
  - v_component_of_wind
  - temperature

typecodes:
  - A320
  - A20N
  # ...

bada:
  bada_4_2_dir: "/path/to/BADA/4.2.1"
```

Tips:
- Keep `data_dir` absolute; subfolders are resolved relative to it.
- Adjust `typecodes` once in the relevant `config.yaml` instead of editing scripts.
- Ensure directories exist before running downloads/preprocessing (`mkdir -p ...`).
- Set `bada.bada_4_2_dir` if you plan to run `07_bada_prediction.py`.
- For QAR, only the needed paths are kept (`data_dir`, `predicted_dir`, `bada_dir`, `models_dir`, `figure_dir`) plus `typecodes` and `computing.default_cpu_count`.
