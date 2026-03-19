# ⚙️ Configure Project Paths and Options

Runtime settings are centralized in YAML files, defining project paths, data scope, and feature flags per pipeline.

---

## 📂 Configuration Files

Create a `config.yaml` at your project root. All `fdm` CLI commands accept this file as their first argument.

---

## 📝 Configuration Structure Example

```yaml title="config.yaml"
paths:
  data_dir: "/path/to/data"
  download_dir: "downloaded_parquet"
  delta_table: "flights.delta"
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

# Selected-parameter filter thresholds (defaults shown below).
# Legacy v1 values for reference: mach.tol=0.002, cas.tol=1.0, vz.min_abs_value=50
selected_params:
  mach:
    tol: 0.0005            # v1: 0.002
    min_len: 120
    alt_threshold: 15000
    smooth_window: 30
    use_alt: true
  cas:
    tol: 0.75              # v1: 1.0
    min_len: 20
    smooth_window: 20
    smooth_method: savgol
  vz:
    tol: 25
    min_len: 25
    min_abs_value: 75      # v1: 50
    smooth_window: 15
    smooth_method: savgol
  alt:
    tol: 25
    min_len: 5
    min_abs_value: 25
    smooth_window: 5
    smooth_method: savgol
  gamma:
    tol: 0.002
    min_len: 15
    smooth_window: 5
    smooth_method: savgol
```

---

## 🔑 Key Parameters

| Section | Parameter | Type | Description |
| :--- | :--- | :--- | :--- |
| **Paths** | `data_dir` | Path | **Crucial:** Keep this path **absolute**. All subfolders are resolved relative to this root |
| **Paths** | `delta_table` | Path | Delta Table written by `fdm download` (partitioned by `typecode`/`date`) |
| **Paths** | `era5_cache_dir` | Path | Local cache for meteorological fields. Prevents re-downloading |
| **Scope** | `typecodes` | List | Adjust aircraft type scope here, not in scripts |
| **BADA** | `bada_4_2_dir` | Path | Set only if running baseline evaluation |
| **ERA5** | `era5_features` | List | Meteorological fields for exogenous inputs |
| **Selected Params** | `selected_params` | Dict | Per-parameter filter thresholds for flight processing (mach, cas, vz, alt, gamma) |

!!! warning "Directory Existence"
    Ensure your main directories exist **before** running scripts:
    ```bash
    mkdir -p /path/to/data/
    ```

---

## 🚀 Next Steps

* **[Create an Architecture](../create_architecture/)**: Now that paths are configured, learn how to build the model's core components.
