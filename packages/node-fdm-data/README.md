# node-fdm-data

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)

Flight data processing, physics, conversions, and schemas for node-fdm.

## Features

- **Unit conversions** — 14 pure Polars expressions (`ft_to_m`, `kt_to_ms`, `celsius_to_kelvin`, `deg_to_rad`, …)
- **ISA model** — Temperature, pressure, and density as numpy functions (`isa_temperature`, `isa_pressure`, `isa_density`) plus a Polars expression variant (`isa_pressure_expr`)
- **Speed conversions** — `mach_to_tas`, `cas_to_tas`, and `vz_to_gamma` using ISA model, isentropic compressible-flow relations, and flight-path angle geometry; `mach_to_tas_real` and `cas_to_tas_real` accept an explicit static temperature (e.g. ERA5 reanalysis) instead of deriving it from ISA
- **Physics constants** — ISA atmosphere parameters, unit conversion factors, QAR discrete-signal lookup tables
- **Meteorological computations** — Haversine distance, Mach/CAS derivation, TAS from wind components; Polars expression variants: `haversine_expr`, `compute_mach_expr`, `compute_cas_expr`
- **Column schemas** — OpenSky 2025, QAR, and ADS-B architectures with typed column lists and conversion registries
- **Flight processor** — Configurable `FlightProcessor` pipeline with method-chaining API
- **Segment detection** — `build_selected_params` detects constant-speed/altitude plateaus and produces target columns (`fdm_alt_target_ft`, `fdm_cas_target_kt`, `fdm_tas_target_kt`, `fdm_gamma_target_rad`) plus a `fdm_tas_target_known` boolean mask (True iff TAS target is non-NaN). Mach detection is restricted to altitude plateaus and aberrant low-Mach segments (mean below `mach_min_value`, default 0.5) are dropped. The TAS target is the FMS envelope `min(mach_to_tas_real, cas_to_tas_real)` on overlap, single-source TAS elsewhere, NaN where neither segment covers (no global backward-fill); ERA5 `era_temp_K` is used when present, ISA otherwise. Climb/descent transition CAS is recovered by minimising `Σ (min(mach_to_tas_real, cas_to_tas_real) − TAS_real)²` and walking outward from the cruise boundary while `|CAS_real − CAS_opt| ≤ cas_deviation_kt` (default 5 kt). `fdm_gamma_target_rad` fuses three sources with priority vz→gamma (highest) > gamma_sel > gamma_from_alt=0 / ALT HLD (lowest), NaN-preserving (gaps between segments stay NaN); `GammaFilterConfig` provides sensible defaults including `min_abs_value` to filter near-zero cruise plateaus; altitude plateaus also emit `fdm_gamma_from_alt_rad` (0.0 in level flight, NaN elsewhere)
- **Preprocessing** — SI conversion with precomputed delta columns (`fdm_alt_diff_m`, `fdm_tas_diff_ms`, `fdm_gamma_diff_rad`; diff=0 where target is NaN), temporal derivatives, OpenSky (altitude diff, segment filtering, subsegment detection, position smoothing, fixed-rate resampling via `resample_flight` / `preprocess_flights`) and QAR (Butterworth, smoothing, engine reduction); BDS speed cleaning (`clean_speeds`, `clean_bds_speeds`: multi-pass Hampel + ERA-deviation cap + short-gap interpolation) and BDS/ERA5 merge (`merge_bds_era5`: coalesce cleaned BDS → raw BDS → ERA5 into `ekf_input_*` columns)
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

### Speed conversions (Mach/CAS to TAS)

```python
from node_fdm_data.physics.speed import (
    cas_to_tas,
    cas_to_tas_real,
    mach_to_tas,
    mach_to_tas_real,
    vz_to_gamma,
)

tas = mach_to_tas(0.78, 10_000.0)         # Mach 0.78 at 10 km (ISA) → ~233 m/s
tas = cas_to_tas(128.6, 10_000.0)         # 250 kt CAS at 10 km (ISA) → ~212 m/s
tas = mach_to_tas_real(0.78, 213.0)       # Mach 0.78 at T=213 K (ERA5) → ~228 m/s
tas = cas_to_tas_real(128.6, 10_000.0, 213.0)  # 250 kt CAS at 10 km, T=213 K
gamma = vz_to_gamma(10.0, 250.0)          # 10 m/s climb at 250 m/s TAS → ~0.04 rad
```

### Column schemas

```python
from node_fdm_data.schemas.opensky import X_COLS, U_COLS, E0_COLS, CONVERSIONS

print(X_COLS)   # ['fdm_distance_cum_m', 'raw_alt_ft', 'fdm_gamma_rad', 'era_tas_kt']
print(U_COLS)   # ['fdm_alt_sel_ft', 'fdm_mach_sel', 'fdm_cas_sel_kt', 'fdm_vz_sel_ftmin']
```

### Flight processor pipeline

```python
import polars as pl
from node_fdm_data.processor import FlightProcessor
from node_fdm_data.conversions import ft_to_m

def add_altitude_m(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(ft_to_m("raw_alt_ft").alias("raw_alt_m"))

processor = FlightProcessor([]).add_step(add_altitude_m)
result = processor.process(raw_df)
```

### Dataset splitting

```python
from node_fdm_data.split import split_by_icao

split_df = split_by_icao(df, ratios=(0.7, 0.15, 0.15), seed=42)
# Returns the input DataFrame with an added `meta_split` column.
# meta_split ∈ {"train", "val", "test"} — deterministic by raw_icao24 hash
```

## Development

<!-- 394 tests -->
```bash
uv run pytest packages/node-fdm-data/ -q
uv run ruff check packages/node-fdm-data/
uv run mypy packages/node-fdm-data/src/
```

## License

[EUPL-1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
