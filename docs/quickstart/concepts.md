# 🧠 Core Concepts

This section introduces the fundamental building blocks of **node-fdm**. Understanding these concepts will help you navigate architectures, preprocess data, and extend the framework with your own models.

---

## 🔢 Column Groups

Every architecture organizes its input and output features into standardized **column groups**. These groups define the **information flow** inside the Neural ODE.

```mermaid
graph LR
    subgraph Inputs
        direction TB
        X(X_COLS<br>State)
        U(U_COLS<br>Control)
        E0(E0_COLS<br>Context)
    end

    subgraph Output
        DX(DX_COLS<br>Derivatives)
    end

    Model[Architecture<br>Layers]

    X & U & E0 --> Model --> DX

    %% Styling matches your Index palette
    classDef state fill:#9ECAE9,stroke:#333,stroke-width:2px,color:black;
    classDef control fill:#FF9D98,stroke:#333,stroke-width:2px,color:black;
    classDef context fill:#88D27A,stroke:#333,stroke-width:2px,color:black;
    classDef derivative fill:#D6A5C9,stroke:#333,stroke-width:2px,color:black;
    classDef model fill:#fff,stroke:#555,stroke-width:1px,stroke-dasharray: 5 5;

    class X state;
    class U control;
    class E0 context;
    class DX derivative;
    class Model model;
```

| Group | Variable Type | Description |
| :--- | :--- | :--- |
| **`X_COLS`** | **State** | Flight variables integrated by the ODE (e.g., altitude, speed). |
| **`U_COLS`** | **Control** | Pilot inputs, FMS selections, or active controls. |
| **`E0_COLS`** | **Environmental** | Exogenous inputs like wind, temperature, or static distances. |
| **`E_COLS`** | **Derived** | Intermediate features calculated by physics layers (e.g., Mach number). |
| **`DX_COLS`** | **Derivatives** | The target outputs predicted by the ODE layer (e.g., `dalt`, `dvz`). |

---

## 🏗️ Architecture Stack

An architecture in **node-fdm** is not just a neural network; it is a **stack** of components defined in `model.py`.

!!! quote "The Stack"
    **Architecture = Physics Layers + Neural Layers**

1.  **Physics/Feature Layers**: Deterministic layers that compute derived quantities (e.g., `TrajectoryLayer`, `EngineLayer`).
2.  **Structured Layers**: The Neural ODE components (`StructuredLayer`) that predict the final derivatives (`DX_COLS`).

The `model.py` file defines the explicit order of these layers and how column groups are mapped between them.

---

## 🔧 Processing Hooks

Architectures are **self-contained**: they define not only the model but also how the data must be prepared. This is handled via two specific hooks:

```mermaid
graph LR
    Raw[Raw Data] --> Hook1
    
    subgraph Architecture Definition
        direction TB
        Hook1[[flight_processing]]
        Hook2[[segment_filtering]]
    end
    
    Hook1 -->|Augmented Data| Hook2
    Hook2 -->|Clean Data| Train[Training Set]

    style Hook1 fill:#FFF9C4,stroke:#FBC02D
    style Hook2 fill:#FFF9C4,stroke:#FBC02D
```

* **`flight_processing`**: Augments raw data before training (e.g., computing `alt_diff`, smoothing signals, adding derived physics).
* **`segment_filtering`**: Removes poor-quality segments or invalid training examples based on domain-specific rules.

---

## 📐 Normalization & Statistics

The `SeqDataset` class handles data normalization automatically to ensure **consistent training and inference**.

!!! check "Automated Features"
    * **Statistics**: Computes mean and standard deviation for every column.
    * **Robust Scaling**: Applies outlier-robust scaling (clipped at 99.5%).
    * **Metadata**: Saves all statistics to `meta.json`, ensuring the inference pipeline uses the exact same scaling as the training pipeline.

---
## 🧩 Registration Mechanism

For the `ODETrainer` or `NodeFDMPredictor` to utilize your custom architecture, it must be discoverable via the central registry.

!!! info "Registry Location"
    The mapping is defined in: `node_fdm/architectures/mapping.py`

This file resolves a simple **string identifier** (e.g., `"opensky_2025"`) into a configuration dictionary that links the three essential components of your architecture:

```mermaid
graph LR
    ID[String Name<br>'opensky_2025'] --> Map{mapping.py}
    
    Map -->|Resolves to| Config[Configuration Dict]
    
    Config --> Cols[Columns Definition]
    Config --> Hooks[Processing Hooks]
    Config --> Class[Model Class]

    style Map fill:#f9f,stroke:#333
```

### Registration Example

Inside `mapping.py`, the registration looks like this:

```python title="node_fdm/architectures/mapping.py"
AVAILABLE_ARCHITECTURES = {
    "opensky_2025": {
        "columns": OpenSkyColumns,     # (1)
        "hooks": {                     # (2)
            "flight": process_flight,
            "segment": filter_segment
        },
        "model_class": OpenSkyModel    # (3)
    },
    # Your custom architecture here...
}
```

1.  **Column Definitions**: Defines `X_COLS`, `U_COLS`, etc.
2.  **Processing Hooks**: The functions used to preprocess raw data.
3.  **Layer Stack**: The Python class defining the Neural ODE layers.

---

## 🚀 Next Steps

* **[Pipelines Overview](../pipelines/)**: Now that you understand the core building blocks, see how they fit together in a complete workflow.
