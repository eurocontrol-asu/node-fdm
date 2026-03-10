# Contributing to axm-fdm-workspace

## Development Setup

```bash
git clone https://github.com/eurocontrol-asu/axm-fdm-workspace.git
cd axm-fdm-workspace
uv sync
uv run pre-commit install
```

## Making Changes

1. Create a branch: `git checkout -b feat/my-change`
2. Make changes in the relevant package under `packages/`
3. Run tests: `make test-all`
4. Run lint: `make lint-all`
5. Commit with conventional commits: `feat(pkg): description`
6. Open a pull request

## Adding a Package

1. Create `packages/my-pkg/` with `pyproject.toml`, `src/`, `tests/`
2. UV auto-discovers members via `packages/*` glob
3. Run `uv sync` to update the lockfile
