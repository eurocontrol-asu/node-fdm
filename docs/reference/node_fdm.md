# 📚 API Overview

The **node_fdm** package is designed with modularity in mind. It separates data handling, physical/neural architectures, and the training/inference engines into distinct namespaces.

---

## 🗺️ Module Map

Here is a high-level view of how the sub-packages interact to form a complete pipeline:

```mermaid
graph LR
    %% Data Flow
    Data[node_fdm.data] --> Trainer
    Data --> Predictor

    %% Logic Flow
    Arch[node_fdm.architectures] -->|Defines| Model[node_fdm.models]
    
    %% Execution Flow
    Model -->|Instantiated by| Trainer[node_fdm.ode_trainer]
    Model -->|Used by| Predictor[node_fdm.predictor]

    %% Styling
    classDef package fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    class Data,Arch,Model,Trainer,Predictor package;
```

---

## 📦 Core Namespaces

| Module | Description | Key Classes |
| :--- | :--- | :--- |
| **[`node_fdm.ode_trainer`](ode_trainer.md)** | **The Training Engine.**<br>Handles the training loop, validation, and PyTorch Lightning integration. | `ODETrainer` |
| **[`node_fdm.predictor`](predictor.md)** | **The Inference Engine.**<br>Wraps trained models to perform trajectory rollouts and simulations. | `NodeFDMPredictor` |
| **[`node_fdm.data`](data.md)** | **Data Pipeline.**<br>Tools for dataset construction, loading, and batching. | `SeqDataset`, `FlightProcessor` |
| **[`node_fdm.architectures`](architectures.md)** | **The Registry.**<br>Contains the built-in definitions (`opensky_2025`, `qar`) and the mapping logic. | `mapping.py`, `columns.py` |
| **[`node_fdm.models`](models.md)** | **Model Wrappers.**<br>The underlying PyTorch modules and ODE integration utilities. | `StructuredLayer`, `PhysicsLayer` |
