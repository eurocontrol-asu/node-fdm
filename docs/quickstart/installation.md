# ⚙️ Installation

This page explains how to install **node-fdm**, configure optional dependencies, and set up the directory structure required for the data pipelines.

---

## 🧩 Prerequisites

Before installing, ensure your environment meets the following requirements:

* **Python 3.11+**
* **OpenSky Trino access** (Required only if you plan to run the full *OpenSky 2025* data collection pipeline).

!!! warning "BADA 4.2 Model Files"
    Support for the **BADA 4.2** physical model is optional but recommended for benchmarking.
    
    * You must obtain the model files separately (due to licensing).
    * You will need to set their location in the relevant `config.yaml` later.

---

## 📦 Install the Package

Choose the installation method that matches your needs.

=== "Standard User (PyPI)"

    Recommended for running existing pipelines and training models.

    **1. Core Installation**
    Install the core library:
    ```bash
    pip install node-fdm
    ```

    **2. Optional Dependencies (Recommended)**
    To install support for traffic data processing, fast meteorology, and visualization:
    ```bash
    # Quotes are often required for shell compatibility
    pip install 'node-fdm[all]'
    ```

    **3. BADA Baseline Support (Optional)**
    The `pybada` wrapper has restrictive dependencies. Use these specific commands to force installation:
    ```bash
    pip install pybada --ignore-requires-python --no-deps
    pip install simplekml 'xlsxwriter>=3.2.5'
    ```

=== "Contributor (Source)"

    Recommended if you plan to modify the code or create custom architectures.

    **1. Clone the repository**
    ```bash
    git clone [https://github.com/eurocontrol-asu/node-fdm.git](https://github.com/eurocontrol-asu/node-fdm.git)
    cd node-fdm
    ```

    **2. Editable Installation**
    Install the package in editable mode along with all development dependencies:
    ```bash
    pip install -e .[all]
    ```

---

## 📁 Configuration & Directories

**node-fdm** relies on configuration files to locate data and artifacts. You must configure these paths before running a pipeline.

!!! tip "Where to configure"
    Edit the configuration file specific to your target pipeline:
    
    * 📂 **OpenSky:** `scripts/opensky/config.yaml`
    * 📂 **QAR:** `scripts/qar/config.yaml`

| Parameter | Description | Requirement |
| :--- | :--- | :--- |
| `paths.data_dir` | The root directory for all data artifacts. | **Required** |
| `paths.era5_cache_dir` | Local cache directory for meteorological fields. | **Required** |
| `bada.bada_4_2_dir` | Path to the folder containing BADA 4.2 files. | Optional |
| `paths.download_dir` | Destination for raw downloaded data. | Auto-managed |
| `paths.models_dir` | Directory where trained models are saved. | Auto-managed |

---

## ✔️ Verification

Run this quick check to verify that `node_fdm` and `torch` are correctly installed and importable.

```bash
python - <<'PY'
import torch
import node_fdm
import sys

print(f"Python version: {sys.version.split()[0]}")
print(f"Torch version:  {torch.__version__}")
print("✅ node_fdm import successful")
PY
```

---

!!! success "Next Step"
    Once installation is verified, head to the **[Core Concepts](../concepts/)** to understand how node-fdm works.