---
title: Home
hide:
  - navigation
  - toc
---

<p align="center">
  <img src="images/logo.jpg" alt="Neural ODE Flight Dynamics" width="450">
</p>

<p align="center" style="font-size: 1.2em; color: #555;">
  A physics-guided <strong>Neural Ordinary Differential Equation (Neural ODE)</strong> framework for aircraft flight dynamics.
</p>

<p align="center">
  <a href="quickstart/installation/" class="md-button md-button--primary">Get Started</a>
  <a href="howto/train_model/" class="md-button">View Examples</a>
</p>

---

## 🎯 At a Glance

**node-fdm** bridges the gap between deep learning and aeronautics. It allows you to compose **hybrid dynamical models** by stacking physical principles, analytical features, and neural networks.

The diagram below illustrates the standard architecture used for **ADS-B data (OpenSky 2025)**, where an analytical layer pre-processes physical features before feeding them into a neural network:

```mermaid
graph LR
    subgraph Inputs ["System Inputs"]
        direction TB
        X((State x))
        U((Control u))
        E((Context e))
    end

    subgraph Core ["Node-FDM Core (ADS-B Arch)"]
        direction LR
        B1[Analytical Layer]
        B2[Neural Net Layer]
    end

    subgraph Solver ["Temporal Integration"]
        direction TB
        DX((Derivative dx/dt))
        ODE[ODE Solver]
    end

    %% Connexions (Ordre strict pour l'index linkStyle)
    %% Index 0, 1, 2
    X --> B1
    U --> B1
    E --> B1
    %% Index 3
    B1 --> B2
    %% Index 4
    B2 --> DX
    %% Index 5
    DX --> ODE
    
    %% Index 6 : Feedback Loop (Cible pour le style rouge)
    ODE -.->|Loss| X

    %% Styles
    classDef cInput fill:#9ECAE9,stroke:#333,stroke-width:2px,color:black
    classDef cControl fill:#FF9D98,stroke:#333,stroke-width:2px,color:black
    classDef cContext fill:#88D27A,stroke:#333,stroke-width:2px,color:black
    classDef cAnalytics fill:#F2CF5B,stroke:#333,stroke-width:2px,color:black
    classDef cNeural fill:#83BCB6,stroke:#333,stroke-width:2px,color:black
    classDef cDerivative fill:#D6A5C9,stroke:#333,stroke-width:2px,color:black

    class X cInput
    class U cControl
    class E cContext
    class B1,ODE cAnalytics
    class B2 cNeural
    class DX cDerivative

    %% Application du style rouge sur le lien d'index 6
    linkStyle 6 stroke:red,stroke-width:2px,stroke-dasharray: 5 5,color:red
```

### Key Capabilities

!!! quote ""
    * **Reconstruct Trajectories**: Generate coherent flight paths from ADS-B or QAR data.
    * **Physics-Aware**: Simulate behavior using latent dynamics constrained by aeronautical laws.
    * **Ready-to-Use**: Includes architectures for **OpenSky 2025** and **QAR**.
    * **Benchmark Ready**: Compare directly against physical models like **BADA**.

---

## 🚀 Workflow & Navigation

Follow the pipelines mirrored in the repository layout.

<div class="grid cards" markdown>

-   [:material-flag-checkered: **Quickstart**](quickstart/installation/)

    ---

    Get started with the essentials.

    * [Installation](quickstart/installation/)
    * [Core Concepts](quickstart/concepts/)
    * [Pipelines Overview](quickstart/pipeline/)

-   [:material-tools: **How to**](howto/configure_params/)

    ---

    Configure and customize your project.

    * [Configure Project](howto/configure_params/)
    * [Create Architecture](howto/create_architecture/)
    * [Train a Model](howto/train_model/)
    * [Run Inference](howto/run_inference/)


-   [:material-book-open-page-variant: **API Reference**](reference/node_fdm/)

    ---

    Technical documentation for developers.

    * [Overview](reference/node_fdm/)
    * [Architectures](reference/architectures/) 
    * [Trainer](reference/ode_trainer/) & [Predictor](reference/predictor/)
    * [Data](reference/data/) & [Models](reference/models/)

</div>

---

## ⚡ Quick Install

You can install the core package directly via pip:

```bash
pip install node-fdm
# Or for editable research mode:
pip install -e .[dev]
```

---

!!! danger "Legal Notice"
    **This project is intended for research purposes only.**
    
    This project is distributed under the **EUPL-1.2** license with specific EUROCONTROL amendments. It must **not** be used as a regulatory or operational tool under any circumstances. See `AMENDMENT_TO_EUPL_license.md` for details.