# CLAUDE.md

## Overview

`node-fdm-v2` is a physics-guided Neural ODE framework for aircraft flight dynamics — a clean-room rebuild of the v1 codebase replacing pandas/raw-dict/scripts with **Polars + Pydantic v2 + structlog + a typed architecture registry**. All pipeline work goes through the `fdm` CLI (cyclopts). The `legacy` branch holds the frozen v1 used in the OpenSky 2025 paper; do not backport to it.

## Workspace Layout

This is a **uv workspace** (`[tool.uv.workspace]`, `members = ["packages/*"]`) with four interdependent packages:

| Package | Depends on | Responsibility |
|---|---|---|
| `node-fdm-data` | — | Polars-first data layer: conversions, physics (ISA), meteo, lateral, schemas (`opensky`, `opensky_v2`, `qar`), preprocessing + `FlightProcessor`, flags, split |
| `node-fdm` | `node-fdm-data` | PyTorch Neural ODE: typed architecture registry, MLP/engine/trajectory layers, `FDM`/`BatchNeuralODE` models, `ODETrainer` + `TrainingConfig` (Pydantic), `NodeFDMPredictor`, `FlightDataset` |
| `node-fdm-bada` | — | BADA 4.2 physical baseline (no PyTorch); pyBADA TCL wrapper + 68-type ICAO→BADA mapping |
| `node-fdm-pipeline` | all three | `fdm` CLI commands, `PipelineConfig` (Pydantic/YAML), architecture resolver |

Install with `uv sync` (pulls the whole workspace).

## Common Commands

All via the root `Makefile`:

```bash
make install        # uv sync
make test           # uv run pytest packages/ -q
make lint           # ruff check + ruff format --check (packages/)
make fmt            # ruff --fix + ruff format (packages/)
make typecheck      # uv run mypy packages/*/src/
make check          # lint + typecheck + test
make audit          # uv run axm-audit audit . --agent
make docs-serve     # mkdocs serve
```

Running a single test:

```bash
uv run pytest packages/node-fdm-data/tests/test_conversions.py::test_ft_to_m -q
```

`pyproject.toml` pins `--import-mode=importlib`; `testpaths` covers each package's `tests/` plus the root `tests/`.

## `fdm` CLI Pipeline

Driven by a single `config.yaml` (`PipelineConfig` Pydantic model). Canonical order — all steps take `--config config.yaml`:

```
aircraft-list → download → preprocess → identify → flag
  → process --arch {opensky|opensky_v2|qar|adsb} → split
  → train → predict / predict-bada → evaluate → visualize
```

The `Makefile` also exposes each step as a target (`make aircraft`, `make download`, … `make split`) and a `make pipeline` that runs the full chain after `make clean-data`. Override `CONFIG=`, `SAMPLE_SIZE=`, `START_DATE=`, `END_DATE=` on the make command line.

Copy `config.example.yaml` → `config.yaml` and set `paths.data_dir` (absolute) and `bada.bada_4_2_dir` before running anything.

## Architecture Registry & Resolver

The `--arch` flag dispatches through two layers that must stay in sync:

- `node_fdm.architectures.registry` — decorator-registered `ArchitectureSpec`s (`opensky_2025`, `opensky_v2`, `qar`, `node_adsb_v1`) defining `x_cols`, `u_cols`, `e0_cols`/`e1_cols`, `dx_cols`.
- `node_fdm_pipeline.resolver.ARCH_BY_NAME` — maps CLI keys (`opensky`, `opensky_v2`, `qar`, `adsb`) to registry names and attaches the matching `preprocessing_fn` / `segment_filter_fn`.

When adding an architecture: register the spec, add it to `ARCH_BY_NAME`, and ensure the CLI `--arch` choices in `node_fdm_pipeline.commands.*` accept the new key.

## Conventions

- Python ≥ 3.12, `from __future__ import annotations` everywhere, explicit `__all__`, `py.typed` markers shipped.
- **mypy strict** across `packages/*/src/`. Per-module overrides in `pyproject.toml` relax strictness for known pain points (`commands.data`, `cli`, `lateral`, `visualize`, and all third-party stubs like `torch`, `polars`, `traffic`, `pyBADA`). Prefer fixing types over adding new overrides.
- Ruff selects `E,F,I,N,UP,B,C4,S,T20,RUF`; `scripts/` are exempted from `T201`/`N999`/`N806`/`S110`, `tests/` from `S1xx`/`S6xx`. `S101` (assert) is globally ignored. Line length 99.
- **No `print()`** in library code — use `structlog.get_logger()` (ruff `T20` enforces).
- DataFrame ops use **Polars** (`pl.Expr`, lazy where possible) — not pandas.
- Configs are **Pydantic v2 models**, not raw dicts. Conversions live in `node_fdm_data.conversions` as `pl.Expr` factories.
- Coverage floor is `fail_under = 80` (`tool.coverage.report`).

## Commits & Branches

- Pre-commit runs trailing-whitespace, yaml/toml checks, ruff (fix + format), mypy, and **conventional-pre-commit** on commit messages. Install once with `uv run pre-commit install`.
- Commit style: `feat(pkg): …`, `fix(pkg): …`. `cliff.toml` governs changelog generation.
- `main` = v2 active development (longitudinal + lateral). `legacy` is frozen — do not target it for new work.

## Scripts & Fixtures

- `scripts/` contains one-off diagnostic/tuning utilities (gradient diagnosis, loader checks, crossover-segment verification). Exempt from most ruff rules and mypy strictness; do not import them from library code.
- `fixtures/` holds golden test data consumed by unit/integration tests. `data/` is user-local pipeline output (gitignored).
