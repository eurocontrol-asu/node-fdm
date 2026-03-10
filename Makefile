.PHONY: install test test-all lint lint-all format type-check check audit clean docs-serve docs-build

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

## Format all packages
format:
	uv run ruff check --fix packages/
	uv run ruff format packages/

## Run type checks
type-check:
	uv run mypy packages/*/src/

## Run audit (quality gate)
audit:
	uv run axm-audit audit . --agent

## Full quality check
check:
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test

## Serve documentation locally
docs-serve:
	uv run mkdocs serve

## Build documentation
docs-build:
	uv run mkdocs build

## Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info
