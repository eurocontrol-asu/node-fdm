# Core concepts

- **Column groups** — Each architecture defines:
  - `X_COLS`: state variables (with derivatives automatically added for ODE targets).
  - `U_COLS`: controls (pilot/FMS selections or actuations).
  - `E0_COLS`: environment inputs (e.g., wind, distances, temperatures).
  - `E_COLS`: derived outputs from physics or feature layers.
  - `DX_COLS`: derivatives the ODE layer predicts.
- **Architectures** — Combine physics/feature layers (e.g., `TrajectoryLayer`) with data-driven ODE layers (e.g., `StructuredLayer`). See each `model.py` to understand the layer stack and column ordering.
- **Processing hooks** — `flight_processing` augments data (e.g., add `alt_diff`) and `segment_filtering` can drop poor-quality segments. Each architecture can define its own.
- **Normalization/stats** — `SeqDataset` computes per-column mean/std (99.5% max) and saves them in `meta.json` for inference.
- **Registration** — Architectures must be listed in `architectures/mapping.py` so trainers/predictors can resolve `architecture_name` to columns, layers, and hooks.

Use `opensky_2025` and `qar` as templates; mirror their structure when creating new architectures.
