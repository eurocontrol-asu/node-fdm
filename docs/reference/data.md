# 💽 Data Pipeline API

The `node_fdm.data` namespace handles the transformation of raw flight records into training-ready tensors.

Its primary responsibilities include applying architecture-specific preprocessing hooks via the **Flight Processor**, normalizing features using robust statistics, and managing efficient sequence loading from disk through the **Dataset** and **Loader** utilities.

---

## 📘 Class Reference

### Flight Processor

::: node_fdm.data.flight_processor
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Dataset

::: node_fdm.data.dataset
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Loader

::: node_fdm.data.loader
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true