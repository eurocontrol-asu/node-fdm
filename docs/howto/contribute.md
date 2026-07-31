# Contributing to node-fdm

Thank you for your interest in contributing to **node-fdm**!

We aim to provide a robust, physics-guided framework for flight dynamics research. Whether you are a researcher correcting a physical formula, a data scientist adding a new preprocessing hook, or a developer fixing a bug, your help is welcome.

---

## 🐛 Bug Reports

Please file bug reports on the [GitHub Issue Tracker](https://github.com/eurocontrol-asu/node-fdm-v2/issues).

When filing a report, please include:

* **Description**: A clear summary of the issue
* **Context**: Which pipeline (OpenSky 2025 or QAR) and which package
* **Configuration**: Relevant parts of your `config.yaml`
* **Logs**: Full traceback or structlog output
* **Environment**: Python version, OS, GPU/CUDA availability

> **Note on Data Privacy**: If the error occurs on a private QAR dataset, **do not** upload the data. Try to reproduce with synthetic data or the public OpenSky sample.

---

## 🛠️ Development Workflow

**1. Fork and Clone**
```bash
git clone https://github.com/<your-username>/node-fdm-v2.git
cd node-fdm-v2
```

**2. Install with UV**
```bash
uv sync
```

**3. Run Tests**
```bash
make test
```

---

## 📥 Pull Requests

1.  Create a new branch: `git checkout -b feature/my-new-architecture`
2.  Make changes and commit with conventional messages (e.g., `feat(data): add new schema`)
3.  Push and submit a **Pull Request** against `main`
4.  Reference related Issues (e.g., "Fixes #42")

---

## 🏗️ Contributing New Architectures

New architectures are developed as provider packages, not as changes to the
generic core. See [Create an Architecture](../create_architecture/) for the full
walkthrough.

**Quick steps:**

1. Define schemas and preprocessing in the provider package.
2. Define its `ArchitectureSpec` and any model-specific layers there.
3. Publish a `node_fdm.architectures` entry point.
4. Validate experiments in the paper repository.
5. Promote stable reference models to `node-fdm-models`.

Only submit architecture-related code to `node-fdm` when it is demonstrably
generic across model families.

---

## 📚 Documentation

Documentation is built with **MkDocs Material**. To preview changes:

```bash
make docs-serve
```

Then open `http://127.0.0.1:8000/` in your browser.

---

## 🎨 Style Guide

* **Python**: Follow PEP 8. We use `ruff` for linting and formatting
* **Type Hints**: Strict typing with `from __future__ import annotations`
* **Docstrings**: Google style
* **`__all__`**: Required in every public module
* **Avoid**: Reformatting unrelated code (makes diffs harder to review)

---

## ⚖️ License & Proprietary Data

* **License**: Contributions are accepted under the **EUPL-1.2** license
* **Proprietary Data**: **Never** commit QAR files, BADA model files, or any other proprietary data
