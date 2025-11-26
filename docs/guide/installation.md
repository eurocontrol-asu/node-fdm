# ⚙️ Installation

This page explains how to install **node-fdm**, configure optional dependencies, and set up the directories used by the data pipelines.


### 🧩 Prerequisites

Before installing, ensure you have:

- **Python 3.11+**
- **OpenSky Trino access** (required only if you plan to run the full OpenSky 2025 pipeline)
- **Optional:** BADA 4.2 model files  
  Set their location in the relevant `config.yaml` (`scripts/opensky/config.yaml`).

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

Each pipeline has its own `config.yaml` (`scripts/opensky/config.yaml`, `scripts/qar/config.yaml`) where you define:

- `paths.data_dir` — root for all data artifacts  
- `paths.download_dir`, `preprocess_dir`, `process_dir`, `predicted_dir`, `bada_dir`, `models_dir`, `figure_dir`  
- `paths.era5_cache_dir` — local cache for meteorological fields  
- `bada.bada_4_2_dir` — required if you run the BADA baseline (`07_bada_prediction.py`)


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
