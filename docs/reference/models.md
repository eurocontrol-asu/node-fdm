# 🧠 Models & Layers API

The `node_fdm.layers` namespace provides the PyTorch `nn.Module` components that form the architecture stack.

---

## 📘 Layer Reference

### Structured Layer
The core Neural ODE component that learns state derivatives from input features.

::: node_fdm.layers.structured
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Trajectory Layer
A deterministic physics layer that computes derived kinematic variables.

::: node_fdm.layers.trajectory
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

---

## 📦 Dataset

The `FlightDataset` handles sequence construction and normalization.

::: node_fdm.dataset
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

---

## 🛩️ BADA Baseline

The `node_fdm_bada` package provides physical model baselines.

### Airspeed Conversions

::: node_fdm_bada.airspeed
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

### Aircraft Mapping

::: node_fdm_bada.mapping
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true
