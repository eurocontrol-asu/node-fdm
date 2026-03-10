# node-fdm-bada

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)

BADA 4.2 aircraft performance baseline for node-fdm.

## Features

| Module | Description |
|---|---|
| `utils` | CAS↔Mach, TAS→CAS conversions, flight phase inference |
| `aircraft_mapping` | ICAO type code → BADA 4.2 identifier (68 aircraft) |
| `predictor` | `process_single_flight` — pyBADA TCL wrapper for reference trajectories |

## Installation

```bash
# From the workspace root
uv sync

# Or install standalone
uv pip install -e packages/node-fdm-bada
```

> **Note**: The predictor module requires [pyBADA](https://www.eurocontrol.int/model/bada) (proprietary).
> BADA data files are NOT included. Without pyBADA, `process_single_flight` returns `None`.

## Usage

### Airspeed conversions

```python
from node_fdm_bada import cas_to_mach, mach_to_cas, tas_to_cas

# CAS → Mach at cruise altitude
mach = cas_to_mach(cas_ms=128.6, h_m=9144.0)  # ~0.78

# Mach → CAS
cas = mach_to_cas(mach=0.78, h_m=9144.0)       # ~128.6 m/s

# TAS → CAS (requires temperature)
cas = tas_to_cas(tas_ms=230.0, h_m=9144.0, temperature_k=223.0)
```

### Aircraft mapping

```python
from node_fdm_bada import get_bada_identifier

bada_id = get_bada_identifier("A320")   # → "A320-214"
bada_id = get_bada_identifier("B738")   # → "B737-800"
```

### Flight phase inference

```python
from node_fdm_bada import get_phase

phase = get_phase(hp_init=5000.0, hp_target=35000.0, threshold=100.0)
# → "Climb"
```

## Development

```bash
uv run pytest packages/node-fdm-bada/ -q
uv run ruff check packages/node-fdm-bada/
uv run mypy packages/node-fdm-bada/src/
```

## License

[EUPL-1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
