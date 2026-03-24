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
| `architectures.adsb` | ADS-B v1 architecture (auto-registered) |
| `models.fdm` | `FlightDynamicsModel` — layered state derivative computation |
| `models.batch_neural_ode` | `BatchNeuralODE` — ODE wrapper with input interpolation |
| `models.fdm_prod` | `FlightDynamicsModelProd` — load pretrained weights for inference |
| `layers.blocks` | `MLPBlock`, `Backbone`, `Head`, `MultiLayerDict` |
| `layers.normalizers` | `InputNormalizer`, `OutputDenormalizer` |
| `layers.structured` | `StructuredLayer` — normalize → backbone → heads → denormalize |
| `layers.trajectory` | `TrajectoryLayer` — vertical speed, Mach, CAS, groundspeed, TAS diff, gamma diff |
| `layers.engine` | `EngineLayer` — N1 and fuel flow (QAR) |
| `trainer` | `ODETrainer` + `TrainingConfig` — ODE rollout loss with per-variable `alpha_dict` weighting, model weights + optimizer checkpoint save/load |
| `predictor` | `NodeFDMPredictor` + `ModelMeta` (typed metadata, euler/rk4 integration) |
| `dataset` | `FlightDataset` → `FlightSample` (typed tensors: x, u, e, dx, optional e1) |
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
print(list(REGISTRY.keys()))  # ['opensky_2025', 'qar', 'node_adsb_v1']
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
    # Optional: per-variable loss weights for x_cols (defaults to 1.0 for all)
    alpha_dict={"altitude_ft": 2.0, "tas_kt": 1.5},
)

trainer = ODETrainer(
    config=config,
    train_dataset=train_ds,
    val_dataset=val_ds,
    model_dir=Path("models/my_model"),
)
history = trainer.train()
# history: list[dict[str, float]] with train_loss, val_loss per epoch

# Resume training from a previous checkpoint
trainer.load_model_weights()    # loads layer .pt files (raises if missing)
trainer.load_optimizer_state()  # loads optimizer.pt if present, warns otherwise
history = trainer.train()
```

### Prediction

```python
from pathlib import Path
from node_fdm.predictor import NodeFDMPredictor, ModelMeta

# Load model metadata
meta = ModelMeta.from_json(Path("models/my_model/meta.json"))
print(meta.architecture_name)  # 'opensky_2025'
print(meta.method)             # 'euler' or 'rk4'
print(meta.optimizer_saved)    # True if optimizer.pt was saved

# Run predictions (integration method is read from meta.json)
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

# FlightSample is a frozen dataclass (e1 is optional for extra environment columns)
sample = FlightSample(x=x_tensor, u=u_tensor, e=e_tensor, dx=dx_tensor, e1=e1_tensor)

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
