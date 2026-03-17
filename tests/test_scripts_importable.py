"""Test that all pipeline scripts are importable without errors.

Scripts with optional dependencies (traffic, pyBADA) are skipped
if those packages are not installed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path for script imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Scripts that require optional external packages
OPTIONAL_DEPS = {
    "scripts.opensky.01_aircraft_list": "traffic",
    "scripts.opensky.02_download_data": "traffic",
    "scripts.opensky.03_preprocess_data": "traffic",
    "scripts.opensky.11_explore_dataset": "traffic",
    "scripts.opensky.07_bada_prediction": "pyBADA",
    "scripts.validate_01_acquisition": "polars",
}

ALL_SCRIPTS = [
    "scripts.opensky.01_aircraft_list",
    "scripts.opensky.02_download_data",
    "scripts.opensky.03_preprocess_data",
    "scripts.opensky.04_weather_spd_process_data",
    "scripts.opensky.05_training",
    "scripts.opensky.06_flight_prediction",
    "scripts.opensky.07_bada_prediction",
    "scripts.opensky.08_visualize_predictions",
    "scripts.opensky.09_performance_aggregation",
    "scripts.opensky.10_dataset_stats",
    "scripts.opensky.11_explore_dataset",
    "scripts.opensky.12_performance",
    "scripts.opensky.13_example_flight",
    "scripts.qar.01_training",
    "scripts.qar.02_inference",
    "scripts.validate_01_acquisition",
]


def _has_dependency(dep: str) -> bool:
    """Check whether an optional dependency is available."""
    try:
        importlib.import_module(dep)
    except ImportError:
        return False
    return True


@pytest.mark.parametrize("module_name", ALL_SCRIPTS)  # type: ignore[misc]
def test_script_importable(module_name: str) -> None:
    """Each script can be imported without errors."""
    dep = OPTIONAL_DEPS.get(module_name)
    if dep and not _has_dependency(dep):
        pytest.skip(f"Optional dependency '{dep}' not installed")

    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        # Skip if the import error is from an optional dep we didn't check
        err_msg = str(exc)
        optional_pkgs = (
            "traffic",
            "pyBADA",
            "fastmeteo",
            "altair",
            "tqdm",
            "joblib",
            "click",
            "matplotlib",
        )
        if any(pkg in err_msg for pkg in optional_pkgs):
            pytest.skip(f"Optional dependency not installed: {exc}")
        raise


def test_no_pandas_import() -> None:
    """No script should contain 'import pandas' at module level."""
    scripts_dir = PROJECT_ROOT / "scripts"
    violations = []

    for py_file in scripts_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        content = py_file.read_text()
        in_docstring = False
        for i, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            # Track triple-quote docstrings
            if '"""' in stripped or "'''" in stripped:
                count = stripped.count('"""') + stripped.count("'''")
                if count % 2 == 1:
                    in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            # Skip comments
            if stripped.startswith("#"):
                continue
            if "import pandas" in stripped:
                # Allow local imports inside functions (indented)
                if line.startswith("        ") or line.startswith("    "):
                    continue
                violations.append(f"{py_file.name}:{i}: {stripped}")

    assert not violations, "Module-level 'import pandas' found in:\n" + "\n".join(violations)
