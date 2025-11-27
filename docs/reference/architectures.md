# 🏗️ Architectures API

The `node_fdm.architectures` namespace contains the blueprint definitions for specific flight dynamics problems.

It serves two main purposes: **Registration** (mapping string names to Python objects) and **Implementation** (defining the column groups, preprocessing logic, and model stacks for specific datasets like OpenSky or QAR).

---

## 🧩 Registry

The mapping module is the central lookup table. It allows the `ODETrainer` and `Predictor` to instantiate the correct classes based on a configuration string.

::: node_fdm.architectures.mapping
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

---

## 📡 OpenSky 2025

This is the reference implementation for public ADS-B data. It defines a physics-informed architecture capable of handling noisy surveillance data.

### Columns Definition
Defines the input/output variables (State, Control, Environment) and their units.

::: node_fdm.architectures.opensky_2025.columns
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Processing Hooks
Functions to clean, smooth, and augment raw ADS-B data.

::: node_fdm.architectures.opensky_2025.flight_process
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Model Stack
The assembly of the Neural ODE, connecting physics layers with the learned derivative layer.

::: node_fdm.architectures.opensky_2025.model
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Trajectory Layer
A specialized physics layer that computes derived kinematic variables.

::: node_fdm.architectures.opensky_2025.trajectory_layer
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true