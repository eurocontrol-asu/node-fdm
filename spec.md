# 📋 Project Specification — node-fdm-v2

## Overview

- **Project**: `node-fdm-v2` — UV mono-workspace
- **Goal**: Physics-guided Neural ODE framework for aircraft flight dynamics, rebuilt from scratch with modern standards
- **Origin**: Clean room rebuild of [eurocontrol-asu/node-fdm](https://github.com/eurocontrol-asu/node-fdm)
- **Org**: `eurocontrol-asu`
- **Python**: ≥ 3.12

## Packages

| Package (PyPI) | Import | Scope |
|---|---|---|
| `node-fdm-data` | `node_fdm_data` | Flight data processing, physics, conversions, schemas |
| `node-fdm` | `node_fdm` | Neural ODE models, layers, training, prediction |
| `node-fdm-bada` | `node_fdm_bada` | BADA aircraft performance baseline |

## Stack — SOTA 2026

### Core

| Package | Version | Rôle | Pourquoi |
|---|---|---|---|
| `polars` | ≥ 1.0 | DataFrames | Remplace pandas — 10-100× plus rapide, lazy eval, typage strict |
| `numpy` | ≥ 1.26 | Arrays numériques | Interop SciPy/PyTorch |
| `scipy` | ≥ 1.13 | Signal processing | Butterworth filter (QAR preprocessing) |
| `torch` | ≥ 2.6 | Neural networks | Modèles, layers, training |
| `torchdiffeq` | ≥ 0.2.5 | ODE solvers | Neural ODE integration |
| `pydantic` | ≥ 2.0 | Validation + configs | Remplace `dict[str, Any]` partout |
| `structlog` | ≥ 24.0 | Logging structuré | Remplace 15+ `print()` |
| `pyyaml` | ≥ 6.0 | Config YAML | Lecture configs training |

### Rejeté (overkill ou redondant)

| Package | Verdict | Raison |
|---|---|---|
| Pandera | Redondant | Polars a déjà des schémas typés natifs |
| Lightning | Trop opinionated | Training loop Neural ODE est non-standard (odeint) |
| Dagster/Prefect | Overkill | 13 scripts séquentiels ≠ pipeline DAG |
| Hydra | Overkill | 2 fichiers YAML de ~10 lignes |
| msgspec | Redondant | Pydantic v2 (Rust core) est assez rapide |

### Optionnel (extras)

| Package | Extra | Rôle |
|---|---|---|
| `matplotlib` | `[viz]` | Courbes d'entraînement |
| `mlflow` | `[tracking]` | Experiment tracking |
| `joblib` + `tqdm` | `[parallel]` | Chargement parallèle des vols |

---

## Design Decisions

### 1. Colonnes — Polars natif (pas de `Column` class)

**Avant** (legacy) : `Column` dataclass avec registre global `_instances_dict`, `DataFrameWrapper` de 158 lignes, noms dérivés magiques.

**Après** : Strings simples + conversion registry par architecture.

```python
# schemas/opensky.py
X_COLS = ["distance_m", "altitude_ft", "gamma_rad", "tas_kt"]
U_COLS = ["alt_sel_ft", "mach_sel", "cas_sel_kt", "vz_sel_ftmin"]
E0_COLS = ["temperature_K", "long_wind_kt", "adep_dist_nm", "ades_dist_nm"]

CONVERSIONS: dict[str, tuple[Callable, str]] = {
    "altitude": (ft_to_m, "altitude_m"),
    "TAS": (kt_to_ms, "tas_ms"),
}
```

### 2. Conversions d'unités — fonctions pures sur `pl.Expr`

```python
def ft_to_m(col: str) -> pl.Expr:
    return pl.col(col) * 0.3048

def kt_to_ms(col: str) -> pl.Expr:
    return pl.col(col) * 0.514444
```

### 3. Architecture Registry — `Pydantic` + auto-register

**Avant** : `list[Any]` non typées + `importlib.import_module()` dynamique.

**Après** :
```python
class LayerSpec(BaseModel, frozen=True):
    name: str
    layer_class: str  # dotted path, résolu au runtime
    input_cols: list[str]
    output_cols: list[str]
    trainable: bool = True

class ArchitectureSpec(BaseModel, frozen=True):
    name: str
    x_cols: list[str]
    u_cols: list[str]
    e0_cols: list[str]
    e1_cols: list[str]
    dx_cols: list[tuple[int, str]]  # (sign, col_name)
    layers: list[LayerSpec]
    preprocessing_fn: str | None = None  # dotted path
    segment_filter_fn: str | None = None
```

### 4. Training Config — `Pydantic BaseModel`

**Avant** : `model_config: dict[str, Any]`, désérialisé manuellement.

**Après** :
```python
class TrainingConfig(BaseModel):
    architecture_name: str
    model_name: str
    model_params: tuple[int, int, int] = (2, 1, 48)
    seq_len: int = 60
    shift: int = 60
    step: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 800
    method: str = "rk4"
```

### 5. Dataset — Typé avec `FlightSample`

**Avant** : `Dataset[dict[str, Tensor]]` qui retourne un `tuple` (type hint faux).

**Après** :
```python
@dataclass(frozen=True)
class FlightSample:
    x: torch.Tensor   # state [seq_len, n_x]
    u: torch.Tensor   # control [seq_len, n_u]
    e: torch.Tensor   # environment [seq_len, n_e]
    dx: torch.Tensor  # derivatives [seq_len, n_dx]

class FlightDataset(Dataset[FlightSample]):
    def __getitem__(self, idx: int) -> FlightSample: ...
```

### 6. Logging — `structlog` remplace `print()`

```python
log = structlog.get_logger()
log.info("epoch_complete", epoch=epoch, train_loss=loss, val_loss=val_loss)
```

---

## Features — `node-fdm-data`

| Module | Description | Source legacy |
|---|---|---|
| `conversions` | `ft_to_m`, `kt_to_ms`, `celsius_to_kelvin` (fonctions pures `pl.Expr`) | `conversions.py` (simplifié) |
| `physics.constants` | ISA atmosphere, NM, kt, gear/flap dicts | `constants.py` (copié) |
| `physics.isa` | ISA pressure/temperature/density functions | extrait de `meteo_and_parameters.py` |
| `meteo` | `haversine`, `compute_mach_and_cas`, `compute_tas`, `detect_constant_segments`, `process_flight` | `meteo_and_parameters.py` (Polars) |
| `processor` | `FlightProcessor` — pipeline configurable | `flight_processor.py` |
| `split` | `split_by_icao` — train/val/test split | `split.py` (Polars) |
| `schemas.opensky` | `X_COLS`, `U_COLS`, `E0_COLS`, `CONVERSIONS` | `opensky_2025/columns.py` |
| `schemas.qar` | `X_COLS`, `U_COLS`, `E0_COLS`, `CONVERSIONS` | `qar/columns.py` |
| `preprocessing.opensky` | `flight_processing`, `segment_filtering` | `opensky_2025/flight_process.py` |
| `preprocessing.qar` | smoothing, filtering, engine reduction | `qar/flight_process.py` |

## Features — `node-fdm`

| Module | Description | Source legacy |
|---|---|---|
| `models.batch_neural_ode` | `BatchNeuralODE` — ODE wrapper + interpolation | copié + typé |
| `models.fdm` | `FlightDynamicsModel` — assemblage layers typé | réécrit avec `ArchitectureSpec` |
| `models.fdm_prod` | `FlightDynamicsModelProd` — load/eval pretrained | copié + `ModelMeta` Pydantic |
| `layers.blocks` | `MLPBlock`, `Backbone`, `Head` | copié + `__all__` |
| `layers.normalizers` | `InputNormalizer`, `OutputDenormalizer` | copié |
| `layers.structured` | `StructuredLayer` | copié |
| `layers.trajectory` | `TrajectoryLayer` (OpenSky + QAR variants) | copié |
| `layers.engine` | `EngineLayer` (QAR) | copié |
| `architectures.registry` | `REGISTRY`, `register()`, `get()` | réécrit |
| `architectures.opensky` | `OPENSKY_2025` spec auto-registered | réécrit |
| `architectures.qar` | `QAR` spec auto-registered | réécrit |
| `trainer` | `ODETrainer` avec `TrainingConfig` + `structlog` | réécrit |
| `predictor` | `NodeFDMPredictor` | copié + typé |
| `dataset` | `FlightDataset` → `FlightSample` typé, Polars reader | réécrit |
| `loader` | `get_train_val_data` | copié (déjà Polars) |
| `losses` | `get_loss` | copié |
| `callbacks` | `TrainingCallback` protocol + `ConsoleCallback` | nouveau |

## Features — `node-fdm-bada`

| Module | Description | Source legacy |
|---|---|---|
| `predictor` | `process_single_flight` — BADA 4.2 prediction | copié + typé |
| `utils` | ISA conversions, CAS↔Mach | copié |
| `aircraft_mapping` | ICAO type → BADA 4.2 identifier | copié |

---

## Architecture

```
node-fdm-v2/
├── pyproject.toml                     # [tool.uv.workspace]
├── Makefile
├── packages/
│   ├── node-fdm-data/
│   │   ├── pyproject.toml             # polars, numpy, scipy (NO torch)
│   │   ├── src/node_fdm_data/
│   │   │   ├── conversions.py
│   │   │   ├── meteo.py
│   │   │   ├── processor.py
│   │   │   ├── split.py
│   │   │   ├── physics/
│   │   │   ├── schemas/
│   │   │   └── preprocessing/
│   │   └── tests/
│   ├── node-fdm/
│   │   ├── pyproject.toml             # node-fdm-data + torch + torchdiffeq + pydantic
│   │   ├── src/node_fdm/
│   │   │   ├── models/
│   │   │   ├── layers/
│   │   │   ├── architectures/
│   │   │   ├── trainer.py
│   │   │   ├── predictor.py
│   │   │   ├── dataset.py
│   │   │   ├── loader.py
│   │   │   ├── losses.py
│   │   │   └── callbacks.py
│   │   └── tests/
│   └── node-fdm-bada/
│       ├── pyproject.toml             # node-fdm-data (NO torch)
│       ├── src/node_fdm_bada/
│       └── tests/
├── scripts/
│   ├── opensky/                       # 13-step pipeline
│   └── qar/                           # 2-step pipeline
├── fixtures/                          # Golden test data
├── docs/
└── joss/
```

## Summary

| Package | Modules | Action | Notes |
|---|---|---|---|
| `node-fdm-data` | 10 | Rewrite | Polars-first, fonctions pures |
| `node-fdm` | 17 | Mix rewrite/copy | Pydantic configs, typed specs, structlog |
| `node-fdm-bada` | 3 | Copy + type | Minimal changes |
| **Total** | **30** | | |
