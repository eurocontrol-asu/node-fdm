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

```
axm-fdm-workspace/
├── packages/          # Workspace members
├── docs/              # Shared documentation
├── mkdocs.yml         # MkDocs with monorepo plugin
└── pyproject.toml     # UV workspace root
```

## Development

```bash
# Install all dependencies
uv sync

# Run all tests
make test-all

# Lint all packages
make lint-all

# Serve docs
make docs-serve
```

## Adding a new package

```bash
mkdir -p packages/my-package/src/my_package packages/my-package/tests
# Add pyproject.toml to the new package
# It will be auto-discovered by UV workspace
```

## License

[Apache-2.0](LICENSE) — © 2026 eurocontrol-asu
