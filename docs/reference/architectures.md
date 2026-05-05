# 🏗️ Architectures API

The `node_fdm.architectures` namespace contains the typed specifications for flight dynamics problems.

It serves two main purposes: **Registration** (mapping string names to `ArchitectureSpec` objects) and **Implementation** (defining the column groups, preprocessing logic, and layer stacks).

---

## 🧩 Registry

The registry module provides `register()` and `get()` for architecture discovery.

::: node_fdm.architectures.registry
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true

---

## 📡 OpenSky 2025

The reference implementation for public ADS-B data. Defines a physics-informed architecture for noisy surveillance data.

::: node_fdm.architectures.adsb_2025
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true
