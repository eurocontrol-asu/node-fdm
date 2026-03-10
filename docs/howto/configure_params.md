# ⚙️ Configure Project Paths and Options

Runtime settings are centralized in YAML files, defining project paths, data scope, and feature flags per pipeline.

---

## 📂 Configuration Files

All configuration settings are defined per pipeline:

* 📡 **OpenSky 2025**: `scripts/opensky/config.yaml`
* ✈️ **QAR (Private)**: `scripts/qar/config.yaml`

---

## 📝 Configuration Structure Example

```yaml title="scripts/opensky/config.yaml"
paths:
  data_dir: "/path/to/data"
  download_dir: "downloaded_parquet"
  preprocess_dir: "preprocessed_parquet"
  era5_cache_dir: "era5_cache"

era5_features:
  - u_component_of_wind
  - v_component_of_wind
  - temperature

typecodes:
  - A320
  - A20N

bada:
  bada_4_2_dir: "/path/to/BADA/4.2.1"
```

---

## 🔑 Key Parameters

| Section | Parameter | Type | Description |
| :--- | :--- | :--- | :--- |
| **Paths** | `data_dir` | Path | **Crucial:** Keep this path **absolute**. All subfolders are resolved relative to this root |
| **Paths** | `era5_cache_dir` | Path | Local cache for meteorological fields. Prevents re-downloading |
| **Scope** | `typecodes` | List | Adjust aircraft type scope here, not in scripts |
| **BADA** | `bada_4_2_dir` | Path | Set only if running baseline evaluation |
| **ERA5** | `era5_features` | List | Meteorological fields for exogenous inputs |

!!! warning "Directory Existence"
    Ensure your main directories exist **before** running scripts:
    ```bash
    mkdir -p /path/to/data/
    ```

---

## 🚀 Next Steps

* **[Create an Architecture](../create_architecture/)**: Now that paths are configured, learn how to build the model's core components.
