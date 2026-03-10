# node-fdm

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue)](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)

Physics-guided Neural ODE models for aircraft flight dynamics.

## Features

| Module | Description |
|---|---|
| `architectures.registry` | Typed architecture registry with `ArchitectureSpec` and `LayerSpec` (Pydantic) |
| `architectures.opensky` | OpenSky 2025 architecture (auto-registered) |
| `architectures.qar` | QAR architecture (auto-registered) |
| `models.fdm` | `FlightDynamicsModel` — layered state derivative computation |
| `models.batch_neural_ode` | `BatchNeuralODE` — ODE wrapper with input interpolation |
| `models.fdm_prod` | `FlightDynamicsModelProd` — load pretrained weights for inference |
| `layers.blocks` | `MLPBlock`, `Backbone`, `Head`, `MultiLayerDict` |
| `layers.normalizers` | `InputNormalizer`, `OutputDenormalizer` |
| `layers.structured` | `StructuredLayer` — normalize → backbone → heads → denormalize |
| `layers.trajectory` | `TrajectoryLayer` — vertical speed, Mach, CAS, groundspeed |
| `layers.engine` | `EngineLayer` — N1 and fuel flow (QAR) |
| `trainer` | `ODETrainer` + `TrainingConfig` with structlog |
| `predictor` | `NodeFDMPredictor` + `ModelMeta` (typed metadata) |
| `dataset` | `FlightDataset` → `FlightSample` (typed tensors) |
| `loader` | `get_train_val_data` — build datasets from split DataFrame |
| `losses` | `get_loss` factory |
| `callbacks` | `TrainingCallback` protocol + `ConsoleCallback` |

## Installation

```bash
# From the workspace root
uv sync

# Or install standalone
uv pip install -e packages/node-fdm
```

## Usage

### Architecture registry

```python
from node_fdm.architectures.registry import get, REGISTRY

# Specs are auto-registered on import
spec = get("opensky_2025")
print(spec.x_cols)   # ['distance_m', 'altitude_ft', 'gamma_rad', 'tas_kt']
print(spec.layers)   # [LayerSpec(name='structured', ...), LayerSpec(name='trajectory', ...)]

# List all registered architectures
print(list(REGISTRY.keys()))  # ['opensky_2025', 'qar']
```

### Training

```python
from pathlib import Path

from node_fdm.trainer import ODETrainer, TrainingConfig

config = TrainingConfig(
    architecture_name="opensky_2025",
    model_name="my_model",
    model_params=(2, 1, 48),   # (backbone_depth, head_depth, neurons)
    seq_len=60,
    shift=60,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=512,
    epochs=800,
    method="rk4",
)

trainer = ODETrainer(
    config=config,
    train_dataset=train_ds,
    val_dataset=val_ds,
    model_dir=Path("models/my_model"),
)
history = trainer.train()
# history: list[dict[str, float]] with train_loss, val_loss per epoch
```

### Prediction

```python
from pathlib import Path
from node_fdm.predictor import NodeFDMPredictor, ModelMeta

# Load model metadata
meta = ModelMeta.from_json(Path("models/my_model/meta.json"))
print(meta.architecture_name)  # 'opensky_2025'

# Run predictions
predictor = NodeFDMPredictor(
    model_path=Path("models/my_model"),
    device="cpu",
)
result = predictor.predict_flight(x_init, u_seq, e_seq)
# result: dict[str, np.ndarray] keyed by state column names
```

### Dataset

```python
from node_fdm.dataset import FlightDataset, FlightSample

# FlightSample is a frozen dataclass
sample = FlightSample(x=x_tensor, u=u_tensor, e=e_tensor, dx=dx_tensor)

# FlightDataset wraps a list of samples
dataset = FlightDataset(samples=[sample, ...])
print(len(dataset))       # number of samples
print(dataset[0].x.shape) # [seq_len, n_x]
```

## Development

```bash
uv run pytest packages/node-fdm/ -q
uv run ruff check packages/node-fdm/
uv run mypy packages/node-fdm/src/
```

## License

[EUPL-1.2](https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
