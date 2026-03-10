# node-fdm-data

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)

Flight data processing, physics, conversions, and schemas for node-fdm.

## Features

- **Unit conversions** — 14 pure Polars expressions (`ft_to_m`, `kt_to_ms`, `celsius_to_kelvin`, `deg_to_rad`, …)
- **ISA model** — Temperature, pressure, and density as functions of geometric altitude
- **Physics constants** — ISA atmosphere parameters, unit conversion factors, QAR discrete-signal lookup tables
- **Meteorological computations** — Haversine distance, Mach/CAS derivation, TAS from wind components
- **Column schemas** — OpenSky 2025 and QAR architectures with typed column lists and conversion registries
- **Flight processor** — Configurable `FlightProcessor` pipeline with method-chaining API
- **Preprocessing** — OpenSky (altitude diff, segment filtering) and QAR (Butterworth, smoothing, engine reduction)
- **Dataset splitting** — `split_by_icao` for deterministic train/val/test split (prevents data leakage)

## Installation

```bash
# From the workspace root
uv sync

# Or install standalone
uv pip install -e packages/node-fdm-data
```

## Usage

### Unit conversions (Polars expressions)

```python
import polars as pl
from node_fdm_data.conversions import ft_to_m, kt_to_ms, celsius_to_kelvin

df = pl.DataFrame({
    "altitude_ft": [35000.0, 10000.0],
    "speed_kt": [450.0, 250.0],
    "temp_c": [-56.5, -4.8],
})

result = df.select(
    ft_to_m("altitude_ft").alias("altitude_m"),
    kt_to_ms("speed_kt").alias("speed_ms"),
    celsius_to_kelvin("temp_c").alias("temp_k"),
)
# ┌────────────┬──────────┬────────┐
# │ altitude_m ┆ speed_ms ┆ temp_k │
# ╞════════════╪══════════╪════════╡
# │ 10668.0    ┆ 231.5    ┆ 216.65 │
# │ 3048.0     ┆ 128.6    ┆ 268.35 │
# └────────────┴──────────┴────────┘
```

### ISA atmosphere model

```python
from node_fdm_data.physics import isa_temperature, isa_pressure, isa_density

h = 10668.0  # cruise altitude in metres

T = isa_temperature(h)   # 216.65 K
P = isa_pressure(h)      # 23842 Pa
rho = isa_density(h)     # 0.3836 kg/m³
```

### Column schemas

```python
from node_fdm_data.schemas.opensky import X_COLS, U_COLS, E0_COLS, CONVERSIONS

print(X_COLS)   # ['distance_m', 'altitude_ft', 'gamma_rad', 'tas_kt']
print(U_COLS)   # ['alt_sel_ft', 'mach_sel', 'cas_sel_kt', 'vz_sel_ftmin']
```

### Flight processor pipeline

```python
import polars as pl
from node_fdm_data.processor import FlightProcessor
from node_fdm_data.conversions import ft_to_m

def add_altitude_m(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(ft_to_m("altitude_ft").alias("altitude_m"))

processor = FlightProcessor([]).add_step(add_altitude_m)
result = processor.process(raw_df)
```

### Dataset splitting

```python
from node_fdm_data.split import split_by_icao

split_df = split_by_icao("data/opensky/", ratios=(0.7, 0.15, 0.15), seed=42)
# DataFrame with columns: [file, icao, split]
# split ∈ {"train", "val", "test"} — deterministic by ICAO group
```

## Development

```bash
uv run pytest packages/node-fdm-data/ -q
uv run ruff check packages/node-fdm-data/
uv run mypy packages/node-fdm-data/src/
```

## License

[EUPL-1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
