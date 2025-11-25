# 🧠 Core Concepts

This section introduces the fundamental building blocks of **node-fdm**.  Understanding these concepts will help you navigate architectures, preprocess data, and extend the framework with your own models.

### 🔢 Column Groups

Each architecture organises its input and output features into well-defined **column groups**:

- **`X_COLS` — State variables**  
  Flight states used as ODE inputs and outputs.  

- **`U_COLS` — Control variables**  
  Pilot or FMS selections, control inputs, or actuations.

- **`E0_COLS` — Environmental inputs**  
  Exogenous variables such as wind, distances, temperatures, or air density.

- **`E_COLS` — Derived/environmental features**  
  Outputs from physics or feature-extraction layers.

- **`DX_COLS` — Derivatives predicted by the ODE layer**  
  Target derivatives for the Neural ODE (e.g. `dalt`, `dvz`, `dmass`).

These groups define the **information flow** inside an architecture.

### 🏗️ Architectures

An architecture combines physics and data-driven components:

- **Physics/feature layers**  
  e.g., `TrajectoryLayer`, `EngineLayer`, or custom layers computing derived quantities.

- **Neural ODE / Structured layers**  
  e.g., `StructuredLayer`, which predicts the derivatives (`DX_COLS`).

Together, these layers form a **stack** defined in each architecture’s `model.py`, where the order of layers and the mapping between column groups are specified.

### 🔧 Processing Hooks

Each architecture can provide two optional hooks:

- **`flight_processing`**  
  Augments raw data before training (e.g., computing `alt_diff`, smoothing, adding derived physics).

- **`segment_filtering`**  
  Removes poor-quality segments or invalid training examples.

These hooks allow architectures to remain **self-contained**, they define not only the model, but also how the data should be prepared for it.

### 📐 Normalization & Statistics

`SeqDataset` automatically computes:

- mean and standard deviation for each column  
- outlier-robust scaling (clipped at 99.5%)  
- metadata stored in `meta.json` for inference

This ensures **consistent training and inference**, even when switching architectures.


### 🧩 Registration Mechanism

For an architecture to be discoverable, it must be added to:

node_fdm/architectures/mapping.py


This mapping resolves each `architecture_name` to:

- its column groups  
- its layers  
- its processing hooks

Without registration, trainers and predictors cannot locate your architecture.

### 🚀 Creating Your Own Architecture

Use the existing **`opensky_2025`** and **`qar`** folders as templates.  
Mirror their structure when defining:

- column groups (`columns.py`)  
- processing hooks (`flight_process.py`)  
- layer stack (`model.py`)  
- any extra physics or feature layers  


