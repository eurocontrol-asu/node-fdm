<p align="center">
  <img src="./docs/images/logo.jpg" alt="Neural Ordinary Differential Equation Flight Dynamics Model" width="45%">
</p>
<p align="center">
  <em>Neural Ordinary Differential Equation (Neural ODE) model for aircraft flight dynamics</em>
</p>

---

### ✈️ Overview
**node-fdm** implements a physics-guided Neural ODE framework for learning and simulating aircraft flight dynamics.  
It combines **data-driven learning** with **physical consistency**, enabling the reproduction of vertical motion and energy exchanges during flight.

---

### ⚖️ Important Legal & Usage Notes
- Distributed under the **EUPL-1.2** licence, with exceptions detailed in `AMENDMENT_TO_EUPL_license.md`, reflecting EUROCONTROL’s status as an international organisation.  
- <ins>**This repository is provided for research purposes only and does not constitute a regulatory framework.**</ins>  
  EUROCONTROL disclaims any responsibility for misuse or operational application of these models.

---


### 📁 Repository Structure

```text
├── code/                     # Pipelines and scripts (OpenSky 2025 & QAR) with per-dataset configs
│   ├── opensky/              # OpenSky 2025 data download/preprocess/train/infer (config.yaml inside)
│   └── qar/                  # QAR training and inference scripts (config.yaml inside)
├── node_fdm/                 # Core Neural ODE library
│   ├── architectures/        # Dataset-specific definitions (opensky_2025, qar)
│   ├── data/                 # Datasets, loaders, and flight processing
│   ├── models/               # Neural ODE modules and production wrappers
│   ├── ode_trainer.py        # Training loop and schedulers
│   └── predictor.py          # Inference entry point
├── preprocessing/            # Shared preprocessing utilities (splits, meteo/parameter enrichment)
├── pybada_predictor/         # BADA baseline predictor and aircraft mapping utilities
├── utils/                    # Data helpers, learning blocks, and physics utilities
├── models/                   # Pretrained checkpoints (OpenSky 2025 fleet + QAR A320 artifacts)
├── docs/                     # MkDocs documentation (guide, how-to, reference, logo)
├── tests/                    # Test scaffolding for shared utils
├── mkdocs.yml                # Documentation site configuration
├── pyproject.toml            # Packaging configuration and dependencies
├── LICENCE.md                # EUPL-1.2 licence (see AMENDMENT_TO_EUPL_license.md)
└── AMENDMENT_TO_EUPL_license.md
```

**Configuration**  
- Each pipeline ships its own `config.yaml` under `code/opensky/` and `code/qar/`
- Edit dataset paths, model names, and training hyperparameters directly in the relevant subproject config before running scripts.

---

### 🎨 Use Case:

- #### OpenSky Symposium 2025 — ADS-B Models

*Jarry, G. & Olive, X. (2025). "Generation of Vertical Profiles with Neural Ordinary Differential Equations Trained on Open Trajectory Data," Journal of Open Aviation Science, Proceedings of the 13th OpenSky Symposium.*

This repository enables **full reproducibility** of the study. All code used to **download and preprocess the data**, **train the models**, **perform trajectory inference**, and **generate the figures** presented in the paper is provided here.

- #### SESAR Innovation Days 2025 — QAR Model

*Jarry, G., Dalmau, R., Olive, X., & Very, P. (2025). "A Neural ODE Approach to Aircraft Flight Dynamics Modelling,"*  
*Proceedings of the SESAR Innovation Days 2025, arXiv:2509.23307.*

For the QAR-based model, the repository provides the **full training pipeline**,  the **complete model implementation**, the **inference scripts**, and the **final trained model weights**.  

⚠️ Due to proprietary restrictions, the **QAR datasets themselves cannot be released**; only the model, code, and weights are included.


---

### 🚧 Work in Progress

This repository is under active development. Future updates will include:

- Improve **Mode S feature reconstruction** to reduce errors in training and evaluation  
- Extend to **lateral dynamics** for full trajectory generation and enhanced speed modelling  
- Incorporate stronger **physical constraints** through physics-based loss regularization  
- Train models to **complete ADS-B data** or **generate trajectories** directly from flight plans  

---

### 🤝 Community — Adding a New Architecture

Community feedback and contributions are welcome to help advance the model’s robustness and applicability.

Want to extend the library with another dataset or modelling choice? Use the existing `opensky_2025` and `qar` packages as templates:

1) **Copy a package skeleton**: duplicate `node_fdm/architectures/opensky_2025` (minimal) or `node_fdm/architectures/qar` (multi-layer example) into a new folder `node_fdm/architectures/<your_arch>`, keep the same file names (`columns.py`, `flight_process.py`, `model.py`, extra layers as needed).
2) **Declare columns**: in `columns.py`, define your state (`X_COLS`), control (`U_COLS`), exogenous (`E0_COLS`), and extra outputs, following the existing column objects. Keep derivative columns for ODE targets.
3) **Build layers**: in `model.py`, wire your layers (e.g., `TrajectoryLayer`, `EngineLayer`, `StructuredLayer`) and expose `ARCHITECTURE` and `MODEL_COLS`. The OpenSky model shows a two-layer baseline; QAR illustrates stacking several derived layers.
4) **Preprocess & filter**: implement `flight_processing` and `segment_filtering` in `flight_process.py` to clean/augment your raw data (see altitude diffs in OpenSky vs. smoothing and engine reduction in QAR).
5) **Register the name**: add your architecture key to `valid_names` in `node_fdm/architectures/mapping.py` so loaders and training scripts can find it.
6) **Test a small run**: train or run inference on a tiny slice to validate column ordering and tensor shapes before opening a pull request; sample pipelines live in `code/` (e.g., OpenSky scripts).

If you contribute back, include a short note on your data assumptions, any proprietary constraints, and a minimal script/notebook that exercises the new architecture.
