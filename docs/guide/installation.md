# Installation

## Prerequisites
- Python 3.11+
- Access to OpenSky data (Trino credentials) if you plan to run the full pipeline.
- Optional: BADA 4.2 files (set `BADA_4_2_DIR` in `config.py`) for the baseline.

## Install the package and extras
```bash
pip install -e .[all]            # core + traffic/fastmeteo/click/tqdm/matplotlib
pip install -e .[bada]           # adds pyBADA if you have the data
# Docs (if you want to build this site locally)
pip install mkdocs-material mkdocstrings[python]
```

## Project directories
`config.py` points everything under `data` by default (works for any architecture as long as your preprocessing writes here):
- `data/downloaded_parquet` : raw OpenSky history/flightlist/extended downloads
- `data/preprocessed_parquet` : decoded + resampled flights (4s)
- `data/processed_flights` : ERA5-enriched segments ready for training
- `data/predicted_flights` : Neural ODE predictions
- `data/bada_flights` : BADA baseline predictions
- `models/` : trained checkpoints per aircraft type

Create the folders before running scripts:
```bash
mkdir -p data/downloaded_parquet data/preprocessed_parquet \
        data/processed_flights data/predicted_flights data/bada_flights \
        data/era5_cache
```

## Quick check
```bash
python - <<'PY'
import torch
import node_fdm
print("Torch:", torch.__version__)
print("node_fdm import OK")
PY
```
