# ⚙️ Installation

This page explains how to install **node-fdm-v2**, configure optional dependencies, and set up the directory structure required for the data pipelines.

---

## 🧩 Prerequisites

* **Python 3.12+**
* **[uv](https://docs.astral.sh/uv/)** package manager
* **OpenSky Trino access** (required only for the full OpenSky data collection pipeline)

!!! warning "BADA 4.2 Model Files"
    Support for the **BADA 4.2** physical model is optional but recommended for benchmarking.

    * You must obtain the model files separately (due to licensing).
    * You will need to set their location in the relevant `config.yaml`.

---

## 📦 Install the Package

=== "Contributor (Recommended)"

    Clone the repository and install all packages via UV:

    ```bash
    git clone https://github.com/eurocontrol-asu/node-fdm-v2.git
    cd node-fdm-v2
    uv sync
    ```

    This installs all three packages (`node-fdm-data`, `node-fdm`, `node-fdm-bada`) in editable mode.

=== "Single Package"

    Install individual packages if you only need a subset:

    ```bash
    uv pip install -e packages/node-fdm-data   # Data layer only
    uv pip install -e packages/node-fdm         # Models + training
    uv pip install -e packages/node-fdm-bada    # BADA baseline
    ```

=== "BADA Support"

    The `pybada` wrapper has restrictive dependencies:
    ```bash
    pip install pybada --ignore-requires-python --no-deps
    pip install simplekml 'xlsxwriter>=3.2.5'
    ```

---

## 📁 Configuration & Directories

**node-fdm-v2** relies on a YAML configuration file to locate data and artifacts. Create a `config.yaml` at your project root (see **[Configure Project](../../howto/configure_params/)**).

| Parameter | Description | Requirement |
| :--- | :--- | :--- |
| `paths.data_dir` | Root directory for all data artifacts | **Required** |
| `paths.era5_cache_dir` | Local cache for meteorological fields | **Required** |
| `bada.bada_4_2_dir` | Path to BADA 4.2 model files | Optional |
| `paths.models_dir` | Directory for trained models | Auto-managed |

---

## ✔️ Verification

Run this quick check to verify the installation:

```python
import torch
import polars as pl
import node_fdm
import node_fdm_data
import sys

print(f"Python version: {sys.version.split()[0]}")
print(f"Torch version:  {torch.__version__}")
print(f"Polars version: {pl.__version__}")
print("✅ All packages imported successfully")
```

---

!!! success "Next Step"
    Once installation is verified, head to the **[Core Concepts](../concepts/)** to understand how node-fdm works.
