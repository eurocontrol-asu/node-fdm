# ⚙️ Installation

This page explains how to install **node-fdm**, configure optional dependencies, and set up the directories used by the data pipelines.


### 🧩 Prerequisites

Before installing, ensure you have:

- **Python 3.11+**
- **OpenSky Trino access** (required only if you plan to run the full OpenSky 2025 pipeline)
- **Optional:** BADA 4.2 model files  
  Set their location in `config.py`:
  ```python
  BADA_4_2_DIR = "/path/to/BADA_4.2/"
  ```

### 📦 Install the Package

Install `node-fdm` in editable mode with all optional dependencies:

```bash
pip install -e .[all]      # core + traffic + fastmeteo + click + tqdm + matplotlib
```

If you want the BADA baseline and have the BADA 4.2 dataset:

```bash
pip install -e .[bada]     # enables pyBADA support
```


### 📁 Project Directories

`config.py` centralises paths and points everything to the `data/` directory by default.  
These folders are used across the pipelines (OpenSky / ERA5 / training / inference):

- `data/downloaded_parquet` — raw OpenSky flight history, flightlist, and extended datasets  
- `data/preprocessed_parquet` — decoded & resampled flights (4 s)  
- `data/processed_flights` — ERA5-enriched segments ready for training  
- `data/predicted_flights` — Neural ODE predictions  
- `data/bada_flights` — BADA baseline outputs  
- `data/era5_cache` — local cache for meteorological fields

Create all required directories:

```bash
mkdir -p data/downloaded_parquet data/preprocessed_parquet \
         data/processed_flights data/predicted_flights \
         data/bada_flights data/era5_cache
```

### ✔️ Quick Check

Verify that `node_fdm` imports correctly:

```bash
python - <<'PY'
import torch
import node_fdm
print("Torch:", torch.__version__)
print("node_fdm import OK")
```

If both lines print successfully, your installation is complete.
