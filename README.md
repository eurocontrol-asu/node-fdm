<div align="center">
  <img src="docs/images/logo.jpg" alt="Neural Ordinary Differential Equation Flight Dynamics Model" width="450">

  <br />
  <br />

  <em>A physics-guided Neural Ordinary Differential Equation (Neural ODE) framework for aircraft flight dynamics simulation and learning.</em>

  <br />
  <br />

  <a href="https://github.com/eurocontrol-asu/node-fdm/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/node-fdm/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://coveralls.io/github/eurocontrol-asu/node-fdm?branch=main"><img src="https://coveralls.io/repos/github/eurocontrol-asu/node-fdm/badge.svg?branch=main" alt="Coverage"></a>
  <a href="https://pypi.org/project/node-fdm/"><img src="https://img.shields.io/pypi/v/node-fdm" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/typed-strict-blue" alt="Typed">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://eurocontrol-asu.github.io/node-fdm/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>

  <br />
  <br />

  <p>
    <a href="#-overview">Overview</a> •
    <a href="#-quick-install">Installation</a> •
    <a href="#-repository-structure">Structure</a> •
    <a href="#-use-cases--publications">Publications</a> •
    <a href="#-contributing">Contributing</a>
  </p>
</div>

---

### 📚 Documentation

**Full documentation, tutorials, and API reference available at:** 👉 **[eurocontrol-asu.github.io/node-fdm](https://eurocontrol-asu.github.io/node-fdm/)**

---

### ✈️ Overview

**node-fdm** implements a physics-guided **Neural Ordinary Differential Equation (Neural ODE)** framework for learning and simulating aircraft flight dynamics. It combines **data-driven learning** with **physical consistency**, enabling the reproduction of vertical motion and energy exchanges during flight.

It allows researchers to:
* 📉 **Reconstruct** coherent trajectories from sparse data.
* 🎮 **Simulate** aircraft behavior using learned latent dynamics.
* 📊 **Benchmark** new models against industry standards like BADA.

---

### ⚖️ Legal & Usage

> [!IMPORTANT]
> **Research Use Only**
>
> * This repository is provided **for research purposes only** and does not constitute a regulatory framework or operational tool.
> * EUROCONTROL disclaims any responsibility for misuse or operational application.
> * Distributed under the **EUPL-1.2** license, with exceptions detailed in `AMENDMENT_TO_EUPL_license.md`.

---

### ⚡ Quick Install

**node-fdm** is available on PyPI. Requires **Python 3.11+**.

#### 1. Standard Installation
For core modeling and training functionalities:
```bash
pip install node-fdm
```

#### 2. Full Installation (Recommended)
Includes support for traffic processing, fast meteorology, and visualization tools:
```bash
pip install 'node-fdm[all]'
```

#### 3. BADA Baseline Support
Support for the BADA physical model is optional. `pybada` has restrictive dependencies, so install it separately:
```bash
pip install pybada --ignore-requires-python --no-deps
pip install simplekml 'xlsxwriter>=3.2.5'
```

> **Developers**: Clone the repository and install in editable mode via `pip install -e .[all]`.

---

### 📁 Repository Structure

```text
├── src/node_fdm/             # 📦 CORE LIBRARY
│   ├── architectures/        #    - Registry of model definitions (opensky_2025, qar)
│   ├── data/                 #    - Datasets, loaders, and flight processing
│   ├── models/               #    - Neural ODE modules, checkpoints, wrappers
│   └── utils/                #    - Data helpers, learning blocks, physics utils
├── scripts/                  # 🚀 PIPELINES
│   ├── opensky/              #    - OpenSky 2025: End-to-end public data pipeline
│   └── qar/                  #    - QAR: training/inference scripts
├── docs/                     # 📚 DOCUMENTATION (MkDocs)
├── mkdocs.yml                #    - Site configuration
├── tests/                    # 🧪 Unit/integration tests
└── AMENDMENT_TO_EUPL_license.md
```

---

### 🎨 Use Cases & Publications

This framework supports the following research publications.

#### 1. OpenSky Symposium 2025 (ADS-B)
*Jarry, G. & Olive, X. (2025). "Generation of Vertical Profiles with Neural Ordinary Differential Equations Trained on Open Trajectory Data," Journal of Open Aviation Science, Proceedings of the 13th OpenSky Symposium.*

<details>
<summary><strong>👇 Click to copy BibTeX</strong></summary>

```bibtex
@inproceedings{jarry2025profiles,
  author = {Jarry, Gabriel and Olive, Xavier},
  title = {Generation of Vertical Profiles with Neural Ordinary Differential Equations Trained on Open Trajectory Data},
  booktitle = {Proceedings of the 13th OpenSky Symposium},
  journal = {Journal of Open Aviation Science},
  year = {2025},
  note = {Under review}
}
```
</details>

This repository enables **full reproducibility** of this study (data download, preprocessing, training, and figure generation).

#### 2. SESAR Innovation Days 2025 (QAR)
*Jarry, G., Dalmau, R., Olive, X., & Very, P. (2025). "A Neural ODE Approach to Aircraft Flight Dynamics Modelling," arXiv:2509.23307.*

<details>
<summary><strong>👇 Click to copy BibTeX</strong></summary>

```bibtex
@misc{jarry2025neural,
  title={A Neural ODE Approach to Aircraft Flight Dynamics Modelling},
  author={Gabriel Jarry and Ramon Dalmau and Xavier Olive and Philippe Very},
  year={2025},
  eprint={2509.23307},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  note = {Proceedings of the SESAR Innovation Days 2025}
}
```
</details>

Includes the **full training pipeline** and **model implementation**.
*⚠️ Note: Raw QAR datasets are proprietary and cannot be released.*

---

### 🤝 Contributing

Community contributions are welcome! See the **[Contribution Guide](https://eurocontrol-asu.github.io/node-fdm/howto/contribute/)** for the full details.

---

### 🚧 Roadmap

This repository is under active development. We are focusing on the following strategic improvements:

| Focus Area | Objective |
| :--- | :--- |
| **Model Scope** | Extend to **lateral dynamics** (turn rates, bank angles) for full 4D trajectory generation. |
| **Data Quality** | Improve **Mode S feature reconstruction** to reduce errors in training and evaluation. |
| **Physical Consistency** | Incorporate stronger **physical constraints** through physics-based loss regularization. |
| **Operationalization** | Train models to **complete ADS-B data** or **generate trajectories** directly from flight plans. |
