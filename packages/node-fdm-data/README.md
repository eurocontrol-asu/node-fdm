# node-fdm-data

[![axm-audit](https://img.shields.io/badge/axm--audit-passing-brightgreen)](https://github.com/eurocontrol-asu/axm-fdm-workspace)
[![axm-init](https://img.shields.io/badge/axm--init-compliant-blue)](https://github.com/eurocontrol-asu/axm-fdm-workspace)

Flight data processing, physics, conversions, and schemas for node-fdm.

## Features

- **Physics constants** — ISA atmosphere parameters, unit conversion factors, QAR discrete-signal lookup tables
- **ISA model** — Temperature, pressure, and density as functions of geometric altitude (troposphere + stratosphere)
- **Unit conversions** — 14 pure Polars expression functions (ft→m, kt→m/s, °C→K, deg→rad, etc.)
- **Meteorological computations** — Haversine distance, Mach/CAS derivation, TAS from wind components
- **Column schemas** — OpenSky 2025 and QAR architectures with typed column lists (X, U, E0, E1, DX) and conversion registries

## Installation

```bash
# From the workspace root
uv sync

# Or install standalone
uv pip install -e packages/node-fdm-data
```

## Development

```bash
# Run tests
uv run pytest packages/node-fdm-data/

# Lint
uv run ruff check packages/node-fdm-data/

# Type check
uv run mypy packages/node-fdm-data/src/
```

## License

[EUPL-1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
