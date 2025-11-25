# node_fdm package overview

Core namespaces and their responsibilities (all architectures share these):
- `node_fdm.ode_trainer` — training loop and checkpointing for modular Neural ODEs.
- `node_fdm.predictor` — inference helper to roll out trajectories.
- `node_fdm.data` — dataset construction, preprocessing hooks, loaders.
- `node_fdm.architectures` — architecture registry and built-in definitions (`opensky_2025`, `qar`, add your own).
- `node_fdm.models` — model wrappers and ODE integration utilities.

Use the dedicated pages in this section for full API details. If you need a top-level view of package exports:

::: node_fdm
    options:
      show_root_heading: true
      show_root_full_path: false
      show_source: true
