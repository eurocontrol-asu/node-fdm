# ⚙️ Configure Project Paths and Options

Runtime settings are centralized in YAML files, defining project paths, data scope, and feature flags per pipeline.

---

## 📂 Configuration Files

All configuration settings are defined per pipeline. Edit the file corresponding to your use case:

* 📡 **OpenSky 2025**: `scripts/opensky/config.yaml`
* ✈️ **QAR (Private)**: `scripts/qar/config.yaml`

---

## 📝 Configuration Structure Example

This example shows the primary fields in the OpenSky configuration.

```yaml title="scripts/opensky/config.yaml"
paths:
  data_dir: "/path/to/data"
  download_dir: "downloaded_parquet"
  preprocess_dir: "preprocessed_parquet"
  # ... (more directories)
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

---

## 🔑 Key Parameters and Best Practices

| Section | Parameter | Type | Best Practice / Description |
| :--- | :--- | :--- | :--- |
| **Paths** | **`data_dir`** | Path | **Crucial:** Keep this path **absolute**. All subfolders (`download_dir`, `models_dir`, etc.) are resolved relative to this root. |
| **Paths** | `era5_cache_dir` | Path | Path for local cache of meteorological fields. Setting this prevents re-downloading large files. |
| **Scope** | `typecodes` | List | **Single Source:** Adjust aircraft type scope here, not by modifying pipeline scripts. |
| **BADA** | `bada_4_2_dir` | Path | Set this **only if** you plan to run baseline evaluation (`07_bada_prediction.py`). |
| **ERA5** | `era5_features` | List | Defines the specific meteorological fields (wind components, temperature) to be used as exogenous inputs. |

!!! info "QAR Pipeline Variations"
    The **QAR** configuration is minimal. It typically only retains essential paths (`data_dir`, `predicted_dir`, `models_dir`), `typecodes`, and options for parallel processing (`computing.default_cpu_count`).

!!! warning "Directory Existence"
    Ensure your main directories exist **before** running data downloads or preprocessing scripts.
    ```bash
    mkdir -p /path/to/data/
    ```

---

## 🚀 Next Steps

* **[Create an Architecture](../create_architecture/)**: Now that paths are configured, learn how to build the model's core components.