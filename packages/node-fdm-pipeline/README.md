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
fdm preprocess --config config.yaml
fdm identify --config config.yaml
fdm train --arch opensky --config config.yaml --typecode A320 --method rk4 --seq-len 200
fdm resume --model models/node_adsb_v1_A320 --config config.yaml --epochs 200 --lr 1e-4
fdm predict --arch opensky --config config.yaml --device cuda:0
fdm evaluate --arch opensky --config config.yaml
```

## Commands

| Command | Description | Status |
|---|---|---|
| `fdm preprocess` | Resample flights: subsegment detection, position smoothing, fixed-rate resampling | ✅ Implemented |
| `fdm identify` | Segment at gaps, assign flight IDs, join flightlist metadata | ✅ Implemented |
| `fdm derive` | Compute derived physics columns (gamma, wind, distance) — étape 4 | ✅ Implemented |
| `fdm train` | Train Neural ODE models (`--method euler\|rk4`) | Placeholder (AXM-363) |
| `fdm resume` | Resume training from checkpoint (`--model`, `--overwrite`) | ✅ Implemented |
| `fdm predict` | Predict with trained models | Placeholder (AXM-363) |
| `fdm predict-bada` | BADA 4.2 baseline predictions | Placeholder (AXM-363) |
| `fdm evaluate` | Compute error metrics by phase | Placeholder (AXM-363) |
| `fdm process` | Process flight data + split | Placeholder (AXM-362) |
| `fdm split` | Assign train/val/test split column (`meta_split`) by ICAO group | ✅ Implemented |
| `fdm dataset-stats` | Dataset split statistics | Placeholder (AXM-364) |
| `fdm visualize` | Prediction comparison plots | Placeholder (AXM-364) |
| `fdm version` | Print version | ✅ Implemented |

## Development

<!-- 170 tests -->
```bash
uv run pytest packages/node-fdm-pipeline/ -q
uv run ruff check packages/node-fdm-pipeline/
uv run mypy packages/node-fdm-pipeline/src/
```
