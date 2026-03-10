# node-fdm-v2

Physics-guided Neural ODE framework for aircraft flight dynamics

<p align="center">
  <a href="https://github.com/eurocontrol-asu/node-fdm-v2/actions/workflows/ci.yml"><img src="https://github.com/eurocontrol-asu/node-fdm-v2/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/license-EUPL--1.2-blue" alt="License">
</p>

---

Clean room rebuild of [eurocontrol-asu/node-fdm](https://github.com/eurocontrol-asu/node-fdm) with modern standards (Polars, Pydantic v2, structlog, typed architecture registry).

## Packages

| Package | Description |
|---|---|
| **[node-fdm-data](packages/node-fdm-data/)** | Flight data processing, physics, unit conversions, column schemas — Polars-first |
| **[node-fdm](packages/node-fdm/)** | Neural ODE models, layers, training, prediction — PyTorch + Pydantic |
| **[node-fdm-bada](packages/node-fdm-bada/)** | BADA 4.2 aircraft performance baseline |

## Architecture

```
node-fdm-v2/
├── packages/
│   ├── node-fdm-data/          # Flight data layer (Polars, NumPy, SciPy)
│   │   ├── conversions         # 14 unit conversion functions (pl.Expr)
│   │   ├── physics/            # ISA atmosphere + constants
│   │   ├── meteo               # Haversine, Mach/CAS, TAS
│   │   ├── schemas/            # OpenSky 2025 + QAR column definitions
│   │   ├── preprocessing/      # Architecture-specific pipelines
│   │   ├── processor           # FlightProcessor (configurable pipeline)
│   │   └── split               # Train/val/test by ICAO group
│   ├── node-fdm/               # Neural ODE models (PyTorch)
│   │   ├── architectures/      # Typed registry + specs (OpenSky, QAR)
│   │   ├── layers/             # MLP blocks, normalizers, trajectory, engine
│   │   ├── models/             # FDM, BatchNeuralODE, FDM Prod
│   │   ├── trainer             # ODETrainer + TrainingConfig (Pydantic)
│   │   ├── predictor           # NodeFDMPredictor + ModelMeta
│   │   ├── dataset             # FlightDataset → FlightSample (typed)
│   │   └── callbacks           # TrainingCallback protocol
│   └── node-fdm-bada/          # BADA 4.2 baseline (no PyTorch)
│       ├── utils               # CAS↔Mach, TAS→CAS, phase inference
│       ├── aircraft_mapping    # ICAO → BADA 4.2 identifier (68 types)
│       └── predictor           # pyBADA TCL wrapper
├── scripts/
│   ├── opensky/                # 13-step data pipeline
│   └── qar/                    # 2-step data pipeline
├── fixtures/                   # Golden test data
└── docs/                       # MkDocs documentation
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/eurocontrol-asu/node-fdm-v2.git
cd node-fdm-v2
uv sync
```

### Training

```python
from pathlib import Path

from node_fdm.architectures.registry import get
from node_fdm.dataset import FlightDataset
from node_fdm.loader import get_train_val_data
from node_fdm.trainer import ODETrainer, TrainingConfig
from node_fdm_data.split import split_by_icao

# 1. Split flights by ICAO type
split_df = split_by_icao("data/opensky/", ratios=(0.7, 0.15, 0.15), seed=42)

# 2. Load architecture spec
spec = get("opensky_2025")

# 3. Build datasets
train_ds, val_ds = get_train_val_data(
    split_df,
    x_cols=spec.x_cols,
    u_cols=spec.u_cols,
    e_cols=spec.e0_cols,
    dx_cols=[col for _, col in spec.dx_cols],
    seq_len=60,
    shift=60,
    preprocessing_fn=None,
    segment_filter_fn=None,
    train_limit=None,
    val_limit=None,
)

# 4. Train
config = TrainingConfig(
    architecture_name="opensky_2025",
    model_name="opensky_v1",
    epochs=200,
    lr=1e-3,
    batch_size=512,
)
trainer = ODETrainer(config, train_ds, val_ds, model_dir=Path("models/"))
history = trainer.train()
```

### Prediction

```python
from pathlib import Path
from node_fdm.predictor import NodeFDMPredictor

predictor = NodeFDMPredictor(model_path=Path("models/opensky_v1"), device="cpu")
result = predictor.predict_flight(x_init, u_seq, e_seq)
# result: dict[str, np.ndarray] with predicted state columns
```

## Stack

| Package | Role |
|---|---|
| **Polars** ≥ 1.0 | DataFrames (replaces pandas) |
| **PyTorch** ≥ 2.6 | Neural network models and training |
| **torchdiffeq** ≥ 0.2.5 | Neural ODE integration |
| **Pydantic** v2 | Config validation (replaces `dict[str, Any]`) |
| **structlog** | Structured logging (replaces `print()`) |
| **NumPy** + **SciPy** | Numerical computation and signal processing |

## Development

```bash
# Install all dependencies
uv sync

# Run all tests
make test

# Lint all packages
make lint

# Type check
make type-check

# Full quality check
make check

# Serve docs
make docs-serve
```

## License

[EUPL-1.2](LICENSE) — © 2026 Eurocontrol
