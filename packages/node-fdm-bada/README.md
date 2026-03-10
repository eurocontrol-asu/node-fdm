# node-fdm-bada

BADA 4.2 aircraft performance baseline for node-fdm.

## Features

| Module | Description |
|---|---|
| `utils` | CAS↔Mach, TAS→CAS conversions, flight phase inference |
| `aircraft_mapping` | ICAO type code → BADA 4.2 identifier (68 aircraft) |
| `predictor` | `process_single_flight` — pyBADA TCL wrapper for reference trajectories |

## Installation

```bash
uv pip install -e packages/node-fdm-bada
```

> **Note**: The predictor module requires [pyBADA](https://www.eurocontrol.int/model/bada) (proprietary).
> BADA data files are NOT included. Without pyBADA, `process_single_flight` returns `None`.

## Usage

```python
from node_fdm_bada import cas_to_mach, get_bada_identifier

# CAS → Mach conversion
mach = cas_to_mach(cas_ms=128.6, h_m=9144.0)

# Aircraft mapping
bada_id = get_bada_identifier("A320")  # → "A320-214"
```

## Development

```bash
uv sync --package node-fdm-bada
uv run pytest packages/node-fdm-bada -q
```

## License

EUPL-1.2
