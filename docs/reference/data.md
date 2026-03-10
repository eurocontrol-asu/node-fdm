# 💽 Data Pipeline API

The `node_fdm_data` package handles flight data processing, unit conversions, atmospheric modeling, and dataset splitting.

Built on **Polars** for high-performance data manipulation.

---

## 📘 Module Reference

### Conversions

Unit conversion functions exposed as Polars expressions.

::: node_fdm_data.conversions
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### ISA Atmosphere

International Standard Atmosphere model functions.

::: node_fdm_data.isa
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Column Schemas

Predefined column groups for each architecture.

::: node_fdm_data.schemas
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Flight Processor

Pipeline for flight data preparation and augmentation.

::: node_fdm_data.processor
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Dataset Splitting

Stratified splitting by ICAO aircraft type.

::: node_fdm_data.split
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true
