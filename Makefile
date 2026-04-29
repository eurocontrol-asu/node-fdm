.PHONY: install test test-all lint lint-all fmt format typecheck type-check check audit clean docs-serve docs-build

## Install all packages in the workspace
install:
	uv sync

## Run tests for all packages
test:
	uv run pytest packages/ -q

## Alias for test
test-all: test

## Lint all packages
lint:
	uv run ruff check packages/
	uv run ruff format --check packages/

## Alias for lint
lint-all: lint

## Format all packages (fix + format)
fmt:
	uv run ruff check --fix packages/
	uv run ruff format packages/

## Alias for fmt
format: fmt

## Run type checks
typecheck:
	uv run mypy packages/*/src/

## Alias for typecheck
type-check: typecheck

## Run audit (quality gate)
audit:
	uv run axm-audit audit . --agent

## Full quality check
check:
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test

## Serve documentation locally
docs-serve:
	uv run mkdocs serve

## Build documentation
docs-build:
	uv run mkdocs build

# ── Pipeline variables ───────────────────────────────────────────────
CONFIG       ?= config.yaml
SAMPLE_SIZE  ?= 100
START_DATE   ?= 2025-09-01
END_DATE     ?= 2025-09-08

.PHONY: pipeline clean-data aircraft download identify preprocess flag enrich derive segments convert split

## Run full data pipeline from scratch: clean → split
pipeline: clean-data aircraft download identify preprocess flag enrich derive segments convert split

## Remove Delta table, aircraft CSV, and preprocessed parquet
clean-data:
	rm -rf data/flights.delta data/preprocessed_parquet data/aircraft_db.csv data/predicted_flights

## Step 0: Query aircraft list
aircraft:
	uv run fdm aircraft-list --config $(CONFIG) --sample-size $(SAMPLE_SIZE)

## Step 1: Download ADS-B data
download:
	uv run fdm download --config $(CONFIG) --start-date $(START_DATE) --end-date $(END_DATE)

## Step 1.5: Resample to regular grid
preprocess:
	uv run fdm preprocess --config $(CONFIG)

## Step 2: Identify flights
identify:
	uv run fdm identify --config $(CONFIG)

## Step 3: Flag valid rows
flag:
	uv run fdm flag --config $(CONFIG)

## Step 4: Enrich with ERA5 weather
enrich:
	uv run fdm enrich --config $(CONFIG)

## Step 5: Compute derived physics columns
derive:
	uv run fdm derive --config $(CONFIG)

## Step 6: Detect selected-parameter segments
segments:
	uv run fdm segments --config $(CONFIG)

## Step 7: Convert to SI + compute derivatives
convert:
	uv run fdm convert --config $(CONFIG)

## Step 8: Train/val/test split
split:
	uv run fdm split --config $(CONFIG)

## Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info
