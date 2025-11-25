# Core concepts

- **State variables (`X_COLS`)** — distance along track, altitude (standard), flight path angle `gamma`, true airspeed `tas`. Derivatives are auto-computed for ODE targets.
- **Controls (`U_COLS`)** — selected MCP/FMS values: altitude, vertical speed, Mach, calibrated airspeed.
- **Environment (`E0_COLS`)** — longitudinal wind, distance to ADEP/ADES, temperature (ERA5).
- **Derived outputs (`E1_COLS`)** — vertical speed, Mach, ground speed, calibrated airspeed, altitude difference to selected altitude.
- **Architecture** — `opensky_2025` stacks a physics-based `TrajectoryLayer` with a data-driven ODE layer (`StructuredLayer`). Column grouping is defined in `node_fdm/architectures/opensky_2025/model.py`.
- **Processing hooks** — `flight_processing` adds `alt_diff` and `segment_filtering` rejects segments with unrealistic distance jumps.
- **Normalization/stats** — `SeqDataset` computes mean/std per column (99.5% max recorded) and stores them in `meta.json` for later inference.

For additional architectures (e.g., `qar`), mirror the same structure: declare columns, build layers, add custom preprocessing, and register the name in `architectures/mapping.py`.
