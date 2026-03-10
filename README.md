# axm-fdm-workspace

Physics-guided Neural ODE framework for aircraft flight dynamics

<p align="center">
  <a href="https://github.com/eurocontrol-asu/axm-fdm-workspace/actions/workflows/axm-quality.yml"><img src="https://github.com/eurocontrol-asu/axm-fdm-workspace/actions/workflows/axm-quality.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/eurocontrol-asu/axm-fdm-workspace/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/axm-fdm-workspace/gh-pages/badges/axm-audit.json" alt="axm-audit"></a>
  <a href="https://github.com/eurocontrol-asu/axm-fdm-workspace/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/axm-fdm-workspace/gh-pages/badges/axm-init.json" alt="axm-init"></a>
  <a href="https://github.com/eurocontrol-asu/axm-fdm-workspace/actions/workflows/axm-quality.yml"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/eurocontrol-asu/axm-fdm-workspace/gh-pages/badges/coverage.json" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <a href="https://eurocontrol-asu.github.io/axm-fdm-workspace/"><img src="https://img.shields.io/badge/docs-live-brightgreen" alt="Docs"></a>
</p>

---

## Features

Clean room rebuild of [eurocontrol-asu/node-fdm](https://github.com/eurocontrol-asu/node-fdm) with modern standards:

| Package | Description |
|---|---|
| **node-fdm-data** | Flight data processing, physics, conversions, schemas — Polars-first |
| **node-fdm** | Neural ODE models, layers, training, prediction — PyTorch + Pydantic |
| **node-fdm-bada** | BADA 4.2 aircraft performance baseline |

- **Polars-first** data layer (replaces pandas)
- **Pydantic v2** configs and validation (replaces `dict[str, Any]`)
- **Typed architecture registry** with auto-registered specs
- **structlog** structured logging (replaces `print()`)
- **UV workspace** with proper package separation

## Installation

```bash
# Clone and install
git clone https://github.com/eurocontrol-asu/axm-fdm-workspace.git
cd axm-fdm-workspace
uv sync
```

## Architecture

```
axm-fdm-workspace/
├── packages/
│   ├── node-fdm-data/     # Flight data processing, physics
│   ├── node-fdm/          # Neural ODE models + training
│   └── node-fdm-bada/     # BADA performance baseline
├── fixtures/              # Golden test data
├── scripts/               # OpenSky + QAR pipelines
├── docs/                  # Shared documentation
├── mkdocs.yml             # MkDocs Material
└── pyproject.toml         # UV workspace root
```

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
