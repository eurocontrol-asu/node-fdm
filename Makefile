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

## Run type checks (per-package, then the root config for papers/)
typecheck:
	@for pkg in packages/*/; do \
		echo "==> mypy $$pkg"; \
		(cd $$pkg && uv run mypy src/ tests/) || exit 1; \
	done
	@echo "==> mypy papers/"
	uv run mypy

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

.PHONY: pipeline clean-data aircraft download decode identify preprocess flag enrich derive label-modes segments convert split

## Run full data pipeline from scratch: clean → split
## Note: `download` auto-chains `decode` (raw cache → Delta) by default,
## so the `decode` target is not listed here. Invoke `make decode` standalone
## to rebuild `data/flights.delta` from the existing `data/raw/` cache.
pipeline: clean-data aircraft download identify preprocess flag enrich clean-speeds derive segments label-modes convert split

## Remove Delta table, aircraft CSV, and preprocessed parquet
## (the system-owned `data/raw/` cache is preserved — `rm -rf data/raw/` to wipe it)
clean-data:
	rm -rf data/flights.delta data/preprocessed_parquet data/aircraft_db.csv data/predicted_flights

## Step 0: Query aircraft list
aircraft:
	uv run fdm aircraft-list --config $(CONFIG) --sample-size $(SAMPLE_SIZE)

## Step 1: Download ADS-B data into data/raw/ (auto-chains decode → Delta)
download:
	uv run fdm download --config $(CONFIG) --start-date $(START_DATE) --end-date $(END_DATE)

## Step 1b: Decode data/raw/ → data/flights.delta (no network)
decode:
	uv run fdm decode --config $(CONFIG) --start-date $(START_DATE) --end-date $(END_DATE)

## Step 2: Identify flights (segment by gap, assign meta_flight_id)
identify:
	uv run fdm identify --config $(CONFIG)

## Step 3: Resample to regular grid (overwrites Delta — every downstream step must rerun)
preprocess:
	uv run fdm preprocess --config $(CONFIG)

## Step 4: Flag valid rows
flag:
	uv run fdm flag --config $(CONFIG)

## Step 5: Enrich with ERA5 weather
enrich:
	uv run fdm enrich --config $(CONFIG)

## Step 6: Clean BDS speeds (Hampel + V-shape + zigzag + ERA fill)
##         Produces bds_*_clean and fdm_tas_from_cas_kt; required by derive.
clean-speeds:
	uv run fdm clean-speeds --config $(CONFIG)

## Step 7: Compute derived physics columns
derive:
	uv run fdm derive --config $(CONFIG)

## Step 7b: Label per-sample mode (TURN + 12 vert×long classes)
label-modes:
	uv run fdm label-modes --config $(CONFIG)

## Step 8: Detect selected-parameter segments
segments:
	uv run fdm segments --config $(CONFIG)

## Step 9: Convert to SI + compute derivatives
convert:
	uv run fdm convert --config $(CONFIG)

## Step 10: Train/val/test split
split:
	uv run fdm split --config $(CONFIG)

## Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info
