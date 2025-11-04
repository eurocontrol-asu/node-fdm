<p align="center">
  <img src="./logo.jpg" alt="Neural Ordinary Differential Equation Flight Dynamics Model" width="45%">
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
├── code/                     # Scripts to download, preprocess, train and evaluate models
├── node_fdm/                 # Core Neural ODE implementation and training utilities
├── preprocessing/            # Data preparation and meteorological enrichment modules
├── pybada_predictor/         # Basine performance models using BADA
├── models/                   # Trained model checkpoints (per aircraft type)
├── data/                     # Input and output data (raw, preprocessed, ERA5 cache)
├── figures/                  # Generated figures from the paper (performance & trajectories)
├── utils/                    # Helper functions for metrics, data handling and physics
└── config.py                 # Global configuration file
```

---

### 🎨 Use Case: OpenSky Symposium 2025 Model

 *Jarry, G. & Olive, X. (2025). "Generation of Vertical Profiles with Neural Ordinary Differential Equations Trained on Open Trajectory Data," Journal of Open Aviation Science, Proceedings of the 13th OpenSky Symposium.*

This repository enables **full reproducibility** of the study.  All code used to **download and preprocess the data**, **train the models**, **perform trajectory inference**, and **generate the figures** presented in the paper is provided here.  

---

### 🚧 Work in Progress

This repository is under active development. Future updates will include:

- Improve **Mode S feature reconstruction** to reduce errors in training and evaluation  
- Extend to **lateral dynamics** for full trajectory generation and enhanced speed modelling  
- Incorporate stronger **physical constraints** through physics-based loss regularization  
- Integrate the **QAR-based Neural ODE model** from :

  *Jarry, G., Dalmau, R., Olive, X., & Very, P. (2025). "A Neural ODE Approach to Aircraft Flight Dynamics Modelling,"*  
  *Proceedings of the SESAR Innovation Days 2025, arXiv:2509.23307*

- Train models to **complete ADS-B data** or **generate trajectories** directly from flight plans  


Community feedback and contributions are welcome to help advance the model’s robustness and applicability.