# 💽 Data Pipeline API

The `node_fdm_data` package handles flight data processing, unit conversions, atmospheric modeling, lateral dynamics, and dataset splitting.

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

::: node_fdm_data.physics.isa
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Meteorology

Haversine distance, TAS recomputation from wind + groundspeed, Mach/CAS derivation.

::: node_fdm_data.meteo
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Lateral Dynamics

Turn detection, orthodromic/rhumb bearing, drift angle, and lateral wind.

::: node_fdm_data.lateral
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Segments

Constant-segment detection and selected-parameter estimation (Mach, CAS, vertical rate).

::: node_fdm_data.segments
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

### Preprocessing — OpenSky

Architecture-specific processing: derived columns, cumulative distance, distance-jump cropping.

::: node_fdm_data.preprocessing.opensky
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Preprocessing — QAR

QAR-specific signal processing: noise filter, mode stabilization, engine reduction.

::: node_fdm_data.preprocessing.qar
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Preprocessing — BDS Speed Cleaning

Multi-pass Hampel filter, ERA-deviation cap, and short-gap interpolation for BDS (Mode-S) airspeed signals.

::: node_fdm_data.preprocessing.clean_speeds
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
