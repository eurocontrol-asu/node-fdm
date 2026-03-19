# node-fdm-pipeline

CLI pipeline for the **node-fdm** Neural ODE framework — typed commands for flight dynamics data processing, training, and evaluation.

## Installation

```bash
# From workspace root
uv sync

# With visualization support
pip install node-fdm-pipeline[viz]
```

## Usage

```bash
fdm --help
fdm version
fdm identify --config config.yaml
fdm train --arch opensky --config config.yaml --typecode A320
fdm predict --arch opensky --config config.yaml --device cuda:0
fdm evaluate --arch opensky --config config.yaml
```

## Commands

| Command | Description | Status |
|---|---|---|
| `fdm identify` | Segment at gaps, assign flight IDs, join flightlist metadata | ✅ Implemented |
| `fdm derive` | Compute derived physics columns (gamma, wind, distance) — étape 4 | ✅ Implemented |
| `fdm train` | Train Neural ODE models | Placeholder (AXM-363) |
| `fdm predict` | Predict with trained models | Placeholder (AXM-363) |
| `fdm predict-bada` | BADA 4.2 baseline predictions | Placeholder (AXM-363) |
| `fdm evaluate` | Compute error metrics by phase | Placeholder (AXM-363) |
| `fdm process` | Process flight data + split | Placeholder (AXM-362) |
| `fdm dataset-stats` | Dataset split statistics | Placeholder (AXM-364) |
| `fdm visualize` | Prediction comparison plots | Placeholder (AXM-364) |
| `fdm version` | Print version | ✅ Implemented |
