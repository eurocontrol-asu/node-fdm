.PHONY: install check test format lint audit ci clean docs-serve help

# ─────────────────────────────────────────────────────────────────────────────
# Development Commands
# ─────────────────────────────────────────────────────────────────────────────

install:  ## Install all dependencies
	uv sync --all-groups

check: lint test  ## Run all checks (lint + tests)

lint:  ## Run linter and type checker
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src

format:  ## Format code
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:  ## Run tests with coverage
	uv run pytest

test-fast:  ## Run tests without coverage
	uv run pytest --no-cov -x

test-golden:  ## Run golden tests only
	uv run pytest tests/golden -v -m golden

coverage:  ## Generate HTML coverage report
	uv run pytest --cov-report=html:coverage_html
	@echo "Coverage report: coverage_html/index.html"

audit:  ## Security audit dependencies
	uv run pip-audit

# ─────────────────────────────────────────────────────────────────────────────
# Build & Publish
# ─────────────────────────────────────────────────────────────────────────────

build:  ## Build package
	uv build

# ─────────────────────────────────────────────────────────────────────────────
# Documentation
# ─────────────────────────────────────────────────────────────────────────────

docs-serve:  ## Serve documentation locally
	uv run mkdocs serve

docs-build:  ## Build documentation
	uv run mkdocs build

# ─────────────────────────────────────────────────────────────────────────────
# CI & Cleanup
# ─────────────────────────────────────────────────────────────────────────────

ci: install check  ## Full CI pipeline

clean:  ## Clean build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache coverage_html .coverage coverage.xml
	rm -rf dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
