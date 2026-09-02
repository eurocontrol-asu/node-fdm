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

### Fleet enrichment status

`weather_status(cohort, day)` inspects one fleet cohort's Delta table for an enrichment day. It returns `(complete, row_count, max_null_fraction)`: missing ERA5 columns or an absent day produce `(False, 0, 0.0)`, while a day that fails enriched-frame validation preserves its row count and returns `(False, row_count, 0.0)`.

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

### Preprocessing — Mode Labelling

Attaches the per-sample `fdm_mode_label` column (TURN + 12 vert×long classes) consumed by the Cui 2019 per-mode loss weighting. Resolution priority `TURN > vertical (ALT > VZ > GAMMA > UNKVERT) > longitudinal (MACH > CAS > UNK)`; the longitudinal regime is computed per `meta_flight_id` from constant sub-runs (length ≥ 2) of `fdm_mach_sel` and `fdm_cas_sel_kt`.

::: node_fdm_data.preprocessing.label_modes
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
