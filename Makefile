.PHONY: test-all lint-all format type-check check clean docs-serve docs-build

## Run tests for all packages
test-all:
	uv run pytest packages/ -q

## Lint all packages
lint-all:
	uv run ruff check packages/
	uv run ruff format --check packages/

## Format all packages
format:
	uv run ruff check --fix packages/
	uv run ruff format packages/

## Run type checks
type-check:
	uv run mypy packages/*/src/

## Full quality check
check:
	$(MAKE) lint-all
	$(MAKE) type-check
	$(MAKE) test-all

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
