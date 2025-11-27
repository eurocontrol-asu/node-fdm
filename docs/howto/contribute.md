# Contributing to node-fdm

Thank you for your interest in contributing to **node-fdm**!

We aim to provide a robust, physics-guided framework for flight dynamics research. Whether you are a researcher correcting a physical formula, a data scientist adding a new preprocessing hook, or a developer fixing a bug, your help is welcome.

We welcome contributions in several forms:
1.  **Bug Reports & Issues**
2.  **Documentation Improvements**
3.  **New Architectures** (Adapting the model to new aircraft or datasets)
4.  **Core Improvements** (Solvers, Physics Layers)

---

## 🐛 Bug Reports

Please file bug reports on the [GitHub Issue Tracker](https://github.com/eurocontrol-asu/node-fdm/issues).

When filing a report, please include:
* **Description**: A clear summary of the issue.
* **Context**: Which pipeline were you running? (e.g., *OpenSky 2025* or *QAR*).
* **Configuration**: Relevant parts of your `config.yaml` (especially `model_config`).
* **Logs**: The full traceback or error message.
* **Environment**: Your Python version, OS, and whether you are using GPU/CUDA.

> **Note on Data Privacy**: If the error occurs on a private QAR dataset, **do not** upload the data samples. Try to reproduce the issue with synthetic data or the public OpenSky sample if possible.

---

## 🛠️ Development Workflow

To contribute code, you need a local development environment.

**1. Fork and Clone**
Fork the repository on GitHub, then clone your fork locally:
```bash
git clone [https://github.com/](https://github.com/)<your-username>/node-fdm.git
cd node-fdm
```

**2. Install in Editable Mode**
We recommend using a virtual environment (venv, conda, or uv). Install the package with all development dependencies:
```bash
pip install -e .[all]
```

**3. Run Tests**
Ensure the current codebase is stable before making changes.
```bash
pytest tests/
```

---

## 📥 Pull Requests (PR)

1.  Create a new branch for your feature or fix: `git checkout -b feature/my-new-architecture`.
2.  Make your changes and commit them with a clear, descriptive message.
3.  Push to your fork and submit a **Pull Request** against the `main` branch of `eurocontrol-asu/node-fdm`.
4.  In the PR description, reference any related Issues (e.g., "Fixes #42").

---

## 🏗️ Contributing New Architectures

The most common way to extend **node-fdm** is by adding support for a new aircraft type or a new data source. We call these **Architectures**.

Instead of modifying the core engine, you should create a self-contained module in `node_fdm/architectures/`.

**Steps to contribute an architecture:**
1.  **Duplicate a Template**: Copy `opensky_2025` (for sparse data) or `qar` (for rich data) into a new folder.
2.  **Define Columns**: Update `columns.py` to map your specific data inputs to SI units.
3.  **Implement Logic**: Adjust `flight_process.py` (cleaning) and `model.py` (layer stack).
4.  **Register**: Add your architecture to `node_fdm/architectures/mapping.py`.

> 📘 **Documentation Reference**: Please strictly follow the [Create an Architecture](https://eurocontrol-asu.github.io/node-fdm/howto/create_architecture/) guide in the documentation to ensure your contribution is compatible with the `ODETrainer`.

---

## 📚 Documentation

Documentation is built with **MkDocs Material**. If you modify code, please update the docstrings (we use `mkdocstrings` to auto-generate API references).

To preview documentation changes locally:
```bash
pip install mkdocs-material mkdocstrings[python]
mkdocs build --clean
mkdocs serve
```
Then open `http://127.0.0.1:8000` in your browser.

---

## 🎨 Style Guide

We do not want to be overly strict, but consistency helps review.
* **Python**: Follow PEP 8. We recommend running `ruff` or `black` before committing.
* **Type Hints**: Please use Python type hints (`def func(df: pd.DataFrame) -> torch.Tensor:`) for all core functions.
* **Existing Code**: Avoid reformatting unrelated existing code, as this makes diffs harder to read.

---

## ⚖️ License & Proprietary Data

* **License**: Contributions are accepted under the **EUPL-1.2** license.
* **Proprietary Data**: **Never** commit QAR files, BADA model files, or any other proprietary data to the repository. The `.gitignore` is set up to exclude `data/`, but please be vigilant.