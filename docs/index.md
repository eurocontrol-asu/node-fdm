<p align="center">
  <img src="images/logo.jpg" alt="Neural Ordinary Differential Equation Flight Dynamics Model" width="45%">
</p>

<p align="center">
  <em>A physics-guided Neural Ordinary Differential Equation (Neural ODE) framework for aircraft flight dynamics.</em>
</p>

---

## 🎯 node-fdm: At a Glance

**node-fdm** is a Python library designed for **learning** and **simulation of aircraft flight dynamics**.

It couples the efficiency of **Neural Ordinary Differential Equations (Neural ODE)** with **physical laws** from aeronautics to:

* Reconstruct **coherent aircraft trajectories** from data (ADS-B or QAR).
* Simulate aircraft behavior through **physically aware** latent dynamics.
* Offer ready-to-use (**OpenSky 2025**, **QAR**) and customizable architectures.
* Enable **benchmarking** against established physical models such as **BADA**.

This documentation will guide you through installation, core concepts, running pipelines, and extending the framework.

---

## 🚀 Quick Start & Navigation

Start from installation, then follow the end-to-end pipelines mirrored in the repository layout.

### 🌟 Start here
- **[Installation](/guide/installation/)**: set up Python, optional extras, and editable installs.
- **[Quickstart](/guide/quickstart/)**: full workflow overview for any architecture plus the OpenSky 2025 example.

### 🧪 Run the pipelines
- **[Configure parameters](/howto/configure_params/)**: edit `scripts/opensky/config.yaml` or `scripts/qar/config.yaml` to set paths, typecodes, and hyperparameters.
- **[Train a model](/howto/train_model/)**: launch `opensky_2025` or `qar` training via `ODETrainer` and monitor checkpoints.
- **[Run inference](/howto/run_inference/)**: load saved models with `NodeFDMPredictor`, roll out trajectories, and export predictions.

### 🛠️ Extend or customise
- **[Create an architecture](/howto/create_architecture/)**: clone the OpenSky/QAR templates, declare columns, hooks, and layer stacks.
- **[Core concepts](/concepts/)**: learn about column groups, processing hooks, and architecture registration.

### 📚 API reference
- **[Architectures](/reference/architectures/)**, **[Data](/reference/data/)**, **[Models](/reference/models/)**, **[Trainer](/reference/ode_trainer/)**, **[Predictor](/reference/predictor/)**, **[Package index](/reference/node_fdm/)**.

---

## 📌 Legal Notice

This project is distributed under the **EUPL-1.2** license with specific EUROCONTROL amendments (see `AMENDMENT_TO_EUPL_license.md`).

It is intended **for research purposes only** and must not be used as a regulatory or operational tool under any circumstances.
